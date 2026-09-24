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

import json
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
from yoda.specialists.jev.questions import CHANNELS, InferenceStatus, neutral
from yoda.specialists.jev.states import build_state_frame
from yoda.specialists.prompts import PROMPTS, VIEWS

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


def _news_states(
    panel: AlignedPanel, horizon: int, lookback: int = 0, history_max: int = 40
) -> pd.DataFrame:
    """One state per asset-day: today's headlines plus the recent record.

    The numeric channels carry their history inside the indicators - a 252-day
    volatility percentile, a 60-day drawdown - so the model can tell where the
    present sits in the recent past. News had no such summary: every headline
    arrived context-free, and the third downgrade in a week read exactly like
    the first. ``lookback`` trading days of prior headlines fix that.

    What does *not* change is when the model is asked. A day with no headlines
    is still skipped even when the window behind it is full, so adding history
    enriches the calls already being made rather than multiplying them.
    """
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

    def split(text: object) -> list[str]:
        return [part.strip() for part in str(text).split(separator) if part.strip()]

    # Per-asset chronology, so the window for day t is a slice and not a scan.
    order = {asset: n for n, asset in enumerate(panel.assets)}
    days = [date.strftime("%Y-%m-%d") for date in panel.dates]
    per_asset: dict[str, list[list[str]]] = {
        asset: [[] for _ in days] for asset in panel.assets
    }
    position = {day: t for t, day in enumerate(days)}
    for asset, date, headlines in zip(
        frame["asset"], frame["date"], frame["headlines"], strict=True
    ):
        per_asset[asset][position[date.strftime("%Y-%m-%d")]] = split(headlines)

    def history(asset: str, t: int) -> list[dict]:
        """Prior days that actually carried news, most recent first."""
        if lookback <= 0:
            return []
        recent: list[dict] = []
        budget = history_max
        for offset in range(1, min(lookback, t) + 1):
            items = per_asset[asset][t - offset]
            if not items or budget <= 0:
                continue
            recent.append(
                {
                    "date": days[t - offset],
                    "trading_days_ago": offset,
                    "headlines": items[:budget],
                }
            )
            budget -= len(items[:budget])
        return recent

    rows = []
    for asset, date, headlines, count in zip(
        frame["asset"],
        frame["date"],
        frame["headlines"],
        frame["news_count"],
        strict=True,
    ):
        day = date.strftime("%Y-%m-%d")
        items = split(headlines)
        t = position[day]
        past = history(asset, t)
        rows.append(
            (
                asset,
                day,
                {
                    "asset": asset,
                    "asset_type": panel.asset_types[order[asset]],
                    "date": day,
                    "prediction_horizon_trading_days": horizon,
                    "headlines": items,
                    "news_count": int(count),
                    # Oldest-first would bury the live story; the model reads
                    # the current day, then walks backwards.
                    "recent_headlines": past,
                    "recent_headlines_days": lookback,
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
        _news_states(panel, horizon, settings.news_lookback, settings.news_history_max)
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
        prompt, schema = PROMPTS[channel], VIEWS[channel]

        def ask(row: tuple) -> tuple[str, str, str, dict]:
            key, asset, date, state = row
            try:
                view = client.view(prompt, state, schema)
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                logger.warning(
                    "jev_view_failed asset=%s date=%s error=%s", asset, date, error
                )
                status = (
                    InferenceStatus.INVALID_RESPONSE
                    if isinstance(error, (ValueError, TypeError))
                    else InferenceStatus.RETRY_EXHAUSTED
                )
                return key, asset, date, _failed(channel, status)
            return key, asset, date, _record(view)

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


def _record(view) -> dict:
    """Split the view: numbers feed the gate, prose is kept for the CIO.

    ``view_summary`` and the evidence lists never reach the numeric path - they
    are the semantic message, and the gate must not be able to key off free
    text it cannot calibrate.
    """
    numeric = {
        name: float(value)
        for name, value in view.model_dump().items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    numeric.update(view.core())
    return {
        **numeric,
        "view_summary": view.view_summary,
        "narrative": json.dumps(
            {
                "evidence": getattr(view, "key_evidence", None)
                or getattr(view, "key_signals", None)
                or getattr(view, "key_risk_drivers", None)
                or [],
                "scenarios": [s.model_dump() for s in view.risk_scenarios],
            }
        ),
        "inference_status": InferenceStatus.OK,
    }


def _failed(channel: str, status: str) -> dict:
    record = dict.fromkeys(CHANNELS[channel]["columns"], 0.0)
    record["inference_status"] = status
    return record


def _fill_synthetic(cache, pending, spec, channel, settings) -> None:
    """Deterministic fake views for CI and dry runs. Never real inference."""
    logger.warning(
        "jev_SYNTHETIC channel=%s rows=%d - fabricated views, NOT inference; "
        "results from this run are meaningless",
        channel,
        len(pending),
    )
    rng = np.random.default_rng(abs(hash((channel, settings.prompt_version))) % 2**32)
    columns = spec["columns"]
    signed = {
        "expected_return_score",
        "directional_bias",
        "momentum_score",
        "volume_confirmation",
        "volatility_impact",
    }
    for key, asset, date in zip(
        pending["key"], pending["asset"], pending["date"], strict=True
    ):
        record = {
            name: float(rng.uniform(-1, 1) if name in signed else rng.uniform(0, 1))
            for name in columns
        }
        record["view_summary"] = "synthetic"
        record["narrative"] = "{}"
        record["inference_status"] = InferenceStatus.OK
        cache.add(key, asset, date, record)


def load_jev_table(config: Config, channel: str) -> pd.DataFrame:
    """The raw cached records, for prediction-quality scoring."""
    path: Path = config.path(config.research.jev.cache) / f"{channel}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No OpenJev cache for {channel!r}: {path}")
    return pd.read_parquet(path)
