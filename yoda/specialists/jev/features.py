"""Build a frozen OpenJev feature table for one specialist channel.

The contract matches every other feature source: hand back a ``(T, N, F)`` cube
plus column names, so the gate, copula, policy and optimizer are identical
whichever backend produced it.

Each channel sees **only** its own point-in-time information set
(:mod:`yoda.specialists.jev.states`), sends **one** ``/v1/systemone`` request
carrying ``model``, ``state`` and ``questions``, and stores the validated
record. Nothing is ever silently zero-filled: a failure is recorded with an
explicit ``inference_status`` so a bad row can be found, counted and re-run.

Backtesting consumes the frozen table and never calls OpenJev online.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from yoda.common.alignment import AlignedPanel
from yoda.common.logger import logger
from yoda.config.settings import Config
from yoda.specialists.jev.cache import JevCache, cache_key
from yoda.specialists.jev.client import JevClient, resolve_model
from yoda.specialists.jev.questions import (
    CHANNELS,
    InferenceStatus,
    InvalidResponse,
    neutral,
)
from yoda.specialists.jev.states import build_state_frame

NEWS_COUNT_SCALE = 10.0  # headlines per asset-day are single digits in this panel


def _rounded(value: float) -> float | None:
    """Round for a stable cache key; a NaN becomes JSON null.

    Sending 0.0 for a missing feature would tell the model "this measured
    zero", which is a different and wrong claim.
    """
    return None if not np.isfinite(value) else round(float(value), 6)


def _numeric_states(panel: AlignedPanel, channel: str, horizon: int) -> pd.DataFrame:
    """One state per asset-day from that channel's own information set."""
    state_map, names = build_state_frame(panel, channel)
    rows = []
    for t, date in enumerate(panel.dates):
        day = date.strftime("%Y-%m-%d")
        for n, asset in enumerate(panel.assets):
            features = {name: _rounded(state_map[name][t, n]) for name in names}
            rows.append(
                (
                    asset,
                    day,
                    {
                        "asset": asset,
                        "asset_type": panel.asset_types[n],
                        "date": day,
                        "prediction_horizon_trading_days": horizon,
                        "features": features,
                    },
                    # Nothing to judge if every feature is missing (warm-up).
                    all(value is None for value in features.values()),
                    0.0,
                )
            )
    return pd.DataFrame(rows, columns=["asset", "date", "state", "skip", "news_count"])


def _news_states(panel: AlignedPanel, horizon: int) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [panel.dates, list(panel.assets)], names=["date", "asset"]
    )
    frame = (
        panel.news.set_index(["date", "asset"])
        .reindex(index)
        .reset_index()
        .assign(
            headlines=lambda f: f["headlines"].fillna(""),
            news_count=lambda f: f["news_count"].fillna(0).astype(int),
        )
    )
    separator = config_separator()
    rows = []
    for asset, date, headlines, count in zip(
        frame["asset"],
        frame["date"],
        frame["headlines"],
        frame["news_count"],
        strict=True,
    ):
        day = date.strftime("%Y-%m-%d")
        items = [
            part.strip() for part in str(headlines).split(separator) if part.strip()
        ]
        rows.append(
            (
                asset,
                day,
                {
                    "asset": asset,
                    "asset_type": panel.asset_types[panel.assets.index(asset)],
                    "date": day,
                    "prediction_horizon_trading_days": horizon,
                    "headlines": items,
                    "news_count": int(count),
                },
                not items,
                float(count),
            )
        )
    return pd.DataFrame(rows, columns=["asset", "date", "state", "skip", "news_count"])


def config_separator() -> str:
    """The separator ``yoda.data`` joined same-day headlines with."""
    from yoda.data.settings import NewsConfig as DataNewsConfig

    return DataNewsConfig().separator


def build_jev_features(
    panel: AlignedPanel, config: Config, channel: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Materialise ``(T, N, F)`` OpenJev features for one channel."""
    if channel not in CHANNELS:
        raise ValueError(f"Unknown OpenJev channel: {channel!r}")
    spec = CHANNELS[channel]
    settings = config.research.jev
    columns = spec["columns"]
    horizon = panel.horizon

    states = (
        _news_states(panel, horizon)
        if channel == "news"
        else _numeric_states(panel, channel, horizon)
    )

    model = "synthetic" if settings.synthetic else resolve_model(settings)
    states["key"] = [
        cache_key(asset, date, channel, state, model, settings.prompt_version, horizon)
        for asset, date, state in zip(
            states["asset"], states["date"], states["state"], strict=True
        )
    ]

    cache_path: Path = config.path(settings.cache) / f"{channel}.parquet"
    cache = JevCache(cache_path)
    todo = states.loc[~states["key"].map(cache.has)].drop_duplicates("key")
    skipped, pending = todo.loc[todo["skip"]], todo.loc[~todo["skip"]]

    for key, asset, date in zip(
        skipped["key"], skipped["asset"], skipped["date"], strict=True
    ):
        cache.add(key, asset, date, neutral(channel))

    if len(pending) and settings.synthetic:
        _fill_synthetic(cache, pending, spec, channel, settings)
    elif len(pending):
        _fill_from_model(cache, pending, spec, channel, settings, model)
    cache.flush()

    joined = cache.table(columns).reindex(states["key"]).reset_index(drop=True)
    for column in columns:
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)
    if "news_count_normalized" in columns:
        joined["news_count_normalized"] = (
            states["news_count"].to_numpy() / NEWS_COUNT_SCALE
        )
    values = joined[list(columns)].to_numpy(dtype=np.float64)
    cube = values.reshape(len(panel.dates), panel.n_assets, len(columns))

    status = cache.table(columns).reindex(states["key"])["inference_status"]
    counts = (
        status.value_counts().to_dict() if "inference_status" in cache.frame else {}
    )
    logger.info(
        "jev_features_built channel=%s horizon=%d shape=%s model=%s status=%s",
        channel,
        horizon,
        cube.shape,
        model,
        counts,
    )
    bad = sum(
        count
        for name, count in counts.items()
        if name in {InferenceStatus.RETRY_EXHAUSTED, InferenceStatus.INVALID_RESPONSE}
    )
    if bad:
        logger.warning(
            "jev_failed_rows channel=%s rows=%d - these carry zeros and are "
            "flagged in inference_status; re-run to fill them",
            channel,
            bad,
        )
    return cube, tuple(columns)


def _fill_from_model(cache, pending, spec, channel, settings, model) -> None:
    logger.info(
        "jev_start channel=%s rows=%d model=%s prompt=%s",
        channel,
        len(pending),
        model,
        settings.prompt_version,
    )
    with JevClient(settings) as client:

        def ask(row: tuple) -> tuple[str, str, str, dict]:
            key, asset, date, state = row
            try:
                response = client.ask(state, spec["questions"])
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                logger.warning(
                    "jev_call_failed asset=%s date=%s error=%s", asset, date, error
                )
                return (
                    key,
                    asset,
                    date,
                    _failed(channel, InferenceStatus.RETRY_EXHAUSTED),
                )
            try:
                record = spec["read"](response)
            except InvalidResponse as error:
                logger.warning(
                    "jev_invalid asset=%s date=%s error=%s", asset, date, error
                )
                return (
                    key,
                    asset,
                    date,
                    _failed(channel, InferenceStatus.INVALID_RESPONSE),
                )
            record["inference_status"] = InferenceStatus.OK
            return key, asset, date, record

        work = list(
            pending[["key", "asset", "date", "state"]].itertuples(
                index=False, name=None
            )
        )
        with ThreadPoolExecutor(max_workers=settings.max_concurrency) as pool:
            for result in tqdm(
                pool.map(ask, work), total=len(work), desc=f"jev-{channel}"
            ):
                cache.add(*result)


def _failed(channel: str, status: str) -> dict:
    record = dict.fromkeys(CHANNELS[channel]["columns"], 0.0)
    record["inference_status"] = status
    return record


def _fill_synthetic(cache, pending, spec, channel, settings) -> None:
    """Deterministic fake answers for CI and dry runs. Never real inference."""
    logger.warning(
        "jev_SYNTHETIC channel=%s rows=%d - fabricated answers, NOT inference; "
        "results from this run are meaningless",
        channel,
        len(pending),
    )
    seed = abs(hash((channel, settings.prompt_version))) % 2**32
    rng = np.random.default_rng(seed)
    distribution = spec["distribution"]
    for key, asset, date in zip(
        pending["key"], pending["asset"], pending["date"], strict=True
    ):
        draw = rng.dirichlet(np.ones(len(distribution)))
        record = dict.fromkeys(spec["columns"], 0.0)
        record.update(dict(zip(distribution, draw, strict=True)))
        record[spec["signal"]] = float(draw[-1] - draw[0])
        record["inference_status"] = InferenceStatus.OK
        if "has_news" in record:
            record["has_news"] = 1.0
        cache.add(key, asset, date, record)


def load_jev_table(config: Config, channel: str) -> pd.DataFrame:
    """The raw cached records, for prediction-quality scoring."""
    path: Path = config.path(config.research.jev.cache) / f"{channel}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No OpenJev cache for {channel!r}: {path}")
    return pd.read_parquet(path)
