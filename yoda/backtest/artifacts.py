"""Durable run artifacts.

A backtest writes three files and stops; scoring is a separate offline pass over
them, which is what lets any interval be re-scored without re-running anything.

    data/processed/runs/<run_id>/
        weights.parquet   long (date, asset, weight) - the realised book
        ledger.parquet    per-step returns, turnover and tail statistics
        run_meta.json     version, gate, news backend, splits, seed, config hash
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd

from yoda.common.logger import logger

WEIGHT_COLUMNS: tuple[str, ...] = ("date", "asset", "weight")
LEDGER_COLUMNS: tuple[str, ...] = (
    "date",
    "port_return",
    "turnover",
    "realized_cvar",
    "realized_var",
    "cash",
    "gross",
)
# Written when available rather than required, so an arm that leaves a column
# unset still produces a valid ledger.
LEDGER_OPTIONAL: tuple[str, ...] = (
    "portfolio_value",
    "nu",
    "scenario_vol",
    "rp_lam",
    "rp_budget",
    "rp_turnover_penalty",
    "rp_alpha",
)


@dataclass(frozen=True)
class Run:
    run_id: str
    path: Path
    weights: pd.DataFrame
    ledger: pd.DataFrame
    meta: dict

    @property
    def label(self) -> str:
        return str(self.meta.get("label", self.run_id))


def config_hash(config) -> str:
    return sha256(repr(config).encode()).hexdigest()[:16]


def _validate(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} is empty - the backtest produced no rows")


def write_run(
    root: Path, run_id: str, weights: pd.DataFrame, ledger: pd.DataFrame, meta: dict
) -> Path:
    _validate(weights, WEIGHT_COLUMNS, "weights.parquet")
    _validate(ledger, LEDGER_COLUMNS, "ledger.parquet")
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(path / "weights.parquet", index=False)
    ledger.to_parquet(path / "ledger.parquet", index=False)
    (path / "run_meta.json").write_text(
        json.dumps({**meta, "run_id": run_id}, indent=2)
    )
    logger.info(
        "run_written id=%s steps=%d books=%d path=%s",
        run_id,
        len(ledger),
        weights["date"].nunique(),
        path,
    )
    return path


def load_run(root: Path, run_id: str) -> Run:
    path = root / run_id
    if not path.exists():
        raise FileNotFoundError(f"No such run: {path}")
    return Run(
        run_id=run_id,
        path=path,
        weights=pd.read_parquet(path / "weights.parquet"),
        ledger=pd.read_parquet(path / "ledger.parquet"),
        meta=json.loads((path / "run_meta.json").read_text()),
    )


def list_runs(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        child.name for child in root.iterdir() if (child / "run_meta.json").exists()
    )
