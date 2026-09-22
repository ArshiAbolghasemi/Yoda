"""Point-in-time feature cache for the news agent.

Every asset-day record is computed once and snapshotted to parquet, keyed by
``(asset, date, sha(headlines), model, prompt_version)``. The backtest reads the
frozen table and never calls a model inside the evaluation loop, so folds and
ablations are deterministic and cost nothing to repeat.

Raw embedding vectors are stored, not the projection: changing ``embed_dim``
should re-project locally, never re-issue thousands of embedding calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from yoda.common.logger import logger
from yoda.specialists.news_agent.schema import NewsAssessment

FIELDS: tuple[str, ...] = (
    "sentiment",
    "confidence",
    "tail_risk_flag",
    "vol_flag",
    "event_count",
)


def cache_key(asset: str, date: str, headlines: str, model: str, prompt: str) -> str:
    digest = sha256("|".join([asset, date, headlines, model, prompt]).encode())
    return digest.hexdigest()


@dataclass
class FeatureCache:
    path: Path
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if self.path.exists():
            self.frame = pd.read_parquet(self.path)
            logger.info("news_cache_loaded rows=%d path=%s", len(self.frame), self.path)
        self._index = set(self.frame["key"]) if len(self.frame) else set()
        self._pending: list[dict] = []

    def has(self, key: str) -> bool:
        return key in self._index

    def add(
        self,
        key: str,
        asset: str,
        date: str,
        assessment: NewsAssessment,
        embedding: list[float],
    ) -> None:
        self._pending.append(
            {
                "key": key,
                "asset": asset,
                "date": date,
                **assessment.features(),
                "rationale": assessment.rationale,
                "embedding": np.asarray(embedding, dtype=np.float32),
            }
        )
        self._index.add(key)

    def flush(self) -> None:
        if not self._pending:
            return
        self.frame = pd.concat(
            [self.frame, pd.DataFrame(self._pending)], ignore_index=True
        ).drop_duplicates("key", keep="last")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(self.path, index=False, compression="zstd")
        logger.info(
            "news_cache_written new=%d total=%d path=%s",
            len(self._pending),
            len(self.frame),
            self.path,
        )
        self._pending.clear()

    def table(self) -> pd.DataFrame:
        """Cached records, ready to join onto the panel by key."""
        if not len(self.frame):
            return pd.DataFrame(columns=["key", *FIELDS, "rationale", "embedding"])
        return self.frame


def project(embeddings: pd.Series, width: int, seed: int = 0) -> np.ndarray:
    """Seeded Gaussian random projection of raw embeddings to ``width`` columns.

    Data-independent by construction, so it adds no train/test leakage and needs
    no fitting; rows without an embedding project to zeros.
    """
    lengths = {
        len(value) for value in embeddings if value is not None and len(value) > 0
    }
    if not lengths:
        return np.zeros((len(embeddings), width), dtype=np.float64)
    dimension = max(lengths)
    matrix = np.random.default_rng(seed).normal(
        scale=1 / np.sqrt(width), size=(dimension, width)
    )
    stacked = np.zeros((len(embeddings), dimension), dtype=np.float64)
    for row, value in enumerate(embeddings):
        if value is not None and len(value) == dimension:
            stacked[row] = value
    return stacked @ matrix
