"""Frozen point-in-time store for OpenJev answers.

Every asset-day answer is computed once and snapshotted to parquet, keyed by
``(asset, date, specialist, input_hash, model_version, prompt_version,
prediction_horizon)``. Walk-forward folds, ablations and repeated backtests read
the frozen table; the backtest never calls the model.

The **complete probability record** is stored alongside an explicit
``inference_status`` - not a thresholded label. Brier score and ECE need the
distribution, and a failed row has to stay distinguishable from a confident
neutral one.

Writes flush incrementally: a channel is tens of thousands of asset-days, and
losing that near the end would be painful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import pandas as pd

from yoda.common.logger import logger

IDENTITY: tuple[str, ...] = ("key", "asset", "date")


def state_hash(state: object) -> str:
    """Stable hash of the exact state the model was shown."""
    return sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[
        :24
    ]


def cache_key(
    asset: str,
    date: str,
    specialist: str,
    state: object,
    model: str,
    prompt: str,
    horizon: int,
) -> str:
    """Everything that could change an answer goes into the key.

    ``specialist`` and ``horizon`` are part of it because the same asset-day at
    a 1-day and a 20-day horizon are different questions, and a model or prompt
    rotation must invalidate cleanly rather than mixing two populations into one
    feature table.
    """
    digest = sha256(
        "|".join(
            [asset, date, specialist, state_hash(state), model, prompt, str(horizon)]
        ).encode()
    )
    return digest.hexdigest()


@dataclass
class JevCache:
    """One parquet per channel, addressed by cache key."""

    path: Path
    flush_every: int = 2000
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if self.path.exists():
            self.frame = pd.read_parquet(self.path)
            logger.info("jev_cache_loaded rows=%d path=%s", len(self.frame), self.path)
        self._index = set(self.frame["key"]) if len(self.frame) else set()
        self._pending: list[dict] = []

    def has(self, key: str) -> bool:
        return key in self._index

    def add(self, key: str, asset: str, date: str, record: dict) -> None:
        self._pending.append({"key": key, "asset": asset, "date": date, **record})
        self._index.add(key)
        if len(self._pending) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self.frame = pd.concat(
            [self.frame, pd.DataFrame(self._pending)], ignore_index=True
        ).drop_duplicates("key", keep="last")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(self.path, index=False, compression="zstd")
        logger.info(
            "jev_cache_written new=%d total=%d path=%s",
            len(self._pending),
            len(self.frame),
            self.path,
        )
        self._pending.clear()

    def table(self, columns: tuple[str, ...]) -> pd.DataFrame:
        """Cached answers indexed by key, with every expected column present."""
        if not len(self.frame):
            empty = pd.DataFrame(columns=[*IDENTITY, *columns, "inference_status"])
            return empty.set_index("key")
        frame = self.frame.drop_duplicates("key", keep="last").set_index("key")
        for column in columns:
            if column not in frame:
                frame[column] = 0.0
        if "inference_status" not in frame:
            frame["inference_status"] = "ok"
        return frame
