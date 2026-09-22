"""The news channel: a specialist plus the interchangeable feature backends.

``NEWS__BACKEND`` picks how ``headlines`` becomes numbers:

``llm_agent`` (default)
    The LangGraph agent in :mod:`yoda.specialists.news_agent`, snapshotted to a
    point-in-time parquet cache. Richer, but a pretrained model has seen the
    future, so this channel is on trial.
``encoder``
    Frozen FinBERT sentiment plus a sentence-transformer embedding, entirely
    local. Fully point-in-time safe, and the control in the leakage ablation.
``none``
    No news features at all - the ``no-news`` arm of the ablation.

All three produce the same ``(T, N, F)`` cube, so the gate, optimizer and
policies are untouched by the choice.
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
from yoda.specialists.base import BaseSpecialist
from yoda.specialists.news_agent.cache import FIELDS, FeatureCache, cache_key, project
from yoda.specialists.news_agent.client import LLMClient
from yoda.specialists.news_agent.graph import assess, build_graph
from yoda.specialists.news_agent.schema import NewsAssessment


class NewsSpecialist(BaseSpecialist):
    """``mu_hat_news``: the signed, confidence-weighted headline view."""

    name = "news"


def _panel_news(panel: AlignedPanel) -> pd.DataFrame:
    """Dense (date, asset) news frame in panel order, empty rows included."""
    index = pd.MultiIndex.from_product(
        [panel.dates, list(panel.assets)], names=["date", "asset"]
    )
    frame = (
        panel.news.set_index(["date", "asset"])
        .reindex(index)
        .reset_index()
        .assign(
            headlines=lambda f: f["headlines"].fillna(""),
            news_count=lambda f: f["news_count"].fillna(0).astype(float),
        )
    )
    frame["day"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


def _cube(frame: pd.DataFrame, columns: list[str], panel: AlignedPanel) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float64)
    return values.reshape(len(panel.dates), panel.n_assets, len(columns))


def _llm_agent_features(
    panel: AlignedPanel, config: Config
) -> tuple[np.ndarray, tuple[str, ...]]:
    settings = config.research.news
    llm = settings.llm
    frame = _panel_news(panel)
    frame["key"] = [
        cache_key(asset, day, headlines, llm.model, llm.prompt_version)
        for asset, day, headlines in zip(
            frame["asset"], frame["day"], frame["headlines"], strict=True
        )
    ]
    cache = FeatureCache(config.path(settings.cache))

    todo = frame.loc[~frame["key"].map(cache.has)].drop_duplicates("key")
    if len(todo):
        client = LLMClient(llm)
        graph = build_graph(client)

        def run(row: tuple) -> tuple[str, str, str, NewsAssessment, list[float]]:
            key, asset, day, headlines = row
            verdict = assess(graph, asset, day, headlines)
            text = verdict.rationale or headlines
            embedding = client.embed(text) if headlines.strip() else []
            return key, asset, day, verdict, embedding

        rows = list(
            todo[["key", "asset", "day", "headlines"]].itertuples(
                index=False, name=None
            )
        )
        logger.info("news_agent_start rows=%d model=%s", len(rows), llm.model)
        with ThreadPoolExecutor(max_workers=llm.max_concurrency) as pool:
            for result in tqdm(pool.map(run, rows), total=len(rows), desc="news-agent"):
                cache.add(*result)
        cache.flush()

    table = cache.table().drop_duplicates("key", keep="last").set_index("key")
    joined = table.reindex(frame["key"]).reset_index(drop=True)
    for column in FIELDS:
        joined[column] = joined[column].fillna(0.0)
    embedded = project(
        joined.get("embedding", pd.Series([None] * len(joined))), llm.embed_dim
    )

    merged = pd.concat(
        [
            joined[list(FIELDS)].reset_index(drop=True),
            pd.DataFrame(embedded, columns=[f"emb_{i}" for i in range(llm.embed_dim)]),
            frame[["news_count"]].reset_index(drop=True),
        ],
        axis=1,
    )
    names = tuple(merged.columns)
    return _cube(merged, list(names), panel), names


def _encoder_features(
    panel: AlignedPanel, config: Config
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Frozen local encoders: FinBERT sentiment + sentence-transformer embedding."""
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline as hf_pipeline
    except ImportError as error:  # pragma: no cover - optional extra
        raise ImportError(
            "NEWS__BACKEND=encoder needs the 'encoder' extra: uv sync --extra encoder"
        ) from error

    settings = config.research.news
    frame = _panel_news(panel)
    cache_path: Path = config.path(settings.cache).with_suffix(".encoder.parquet")
    if cache_path.exists():
        stored = pd.read_parquet(cache_path)
    else:
        text = frame["headlines"].tolist()
        filled = [value for value in text if value.strip()]
        classifier = hf_pipeline(
            "sentiment-analysis",
            model=settings.encoder_sentiment_model,
            truncation=True,
        )
        embedder = SentenceTransformer(settings.encoder_embed_model)
        scores = dict(
            zip(
                filled,
                (
                    (1.0 if out["label"].lower().startswith("pos") else -1.0)
                    * float(out["score"])
                    if out["label"].lower() != "neutral"
                    else 0.0
                    for out in classifier(filled, batch_size=32)
                ),
                strict=True,
            )
        )
        vectors = dict(zip(filled, embedder.encode(filled, batch_size=32), strict=True))
        stored = pd.DataFrame(
            {
                "sentiment": [scores.get(value, 0.0) for value in text],
                "embedding": [
                    vectors.get(
                        value, np.zeros(embedder.get_sentence_embedding_dimension())
                    )
                    for value in text
                ],
            }
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stored.to_parquet(cache_path, index=False, compression="zstd")
        logger.info("news_encoder_cached rows=%d path=%s", len(stored), cache_path)

    width = settings.llm.embed_dim
    merged = pd.concat(
        [
            stored[["sentiment"]].reset_index(drop=True),
            pd.DataFrame(
                project(stored["embedding"], width),
                columns=[f"emb_{i}" for i in range(width)],
            ),
            frame[["news_count"]].reset_index(drop=True),
        ],
        axis=1,
    )
    names = tuple(merged.columns)
    return _cube(merged, list(names), panel), names


def build_news_features(
    panel: AlignedPanel, config: Config
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Materialise the news feature cube for the configured backend."""
    backend = config.research.news.backend
    if backend == "none":
        raise ValueError("NEWS__BACKEND=none has no news features by definition")
    build = _llm_agent_features if backend == "llm_agent" else _encoder_features
    cube, names = build(panel, config)
    logger.info(
        "news_features_built backend=%s shape=%s features=%d",
        backend,
        cube.shape,
        len(names),
    )
    return cube, names
