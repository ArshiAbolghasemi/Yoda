"""Walk-forward execution.

The engine fits on a fold's training rows, trades the fold's test rows, applies
transaction costs, and writes artifacts. It computes no performance metrics:
that is :mod:`yoda.evaluation`'s job, offline, from the files written here.

No lookahead anywhere. At row ``t`` the stack sees features up to and including
``t``, the policy chooses ``RiskParams``, the optimizer chooses the book, and the
book earns ``returns[t + 1]``. Between rebalances the book drifts with the
market rather than being silently rebalanced for free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from yoda.backtest.artifacts import config_hash, write_run
from yoda.common.alignment import AlignedPanel
from yoda.common.logger import logger
from yoda.common.types import DROCVaROptimizer, Gate, RiskParamPolicy
from yoda.config.settings import Config
from yoda.pipeline.stack import AllocationStack, build_stack
from yoda.tailvoi.baselines import EqualWeightGate

GateFactory = Callable[[AllocationStack, np.ndarray], Gate]
PolicyFactory = Callable[[AllocationStack, np.ndarray, np.ndarray], RiskParamPolicy]


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def generate_folds(panel: AlignedPanel, config: Config) -> list[Fold]:
    """Expanding-window folds. ``SPLIT__REFIT_DAYS=0`` means one fold."""
    split = config.research.split
    train = panel.positions(split.train_start, split.train_end)
    val = panel.positions(split.val_start, split.val_end)
    test = panel.positions(split.test_start, split.test_end)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        if len(rows) == 0:
            raise ValueError(f"Split '{name}' selects no rows of the panel")
    stride = split.refit_days
    if stride <= 0:
        return [Fold(0, train, val, test)]

    folds, start = [], 0
    while start < len(test):
        # Expanding history: everything already traded joins the training set.
        history = np.concatenate([train, val, test[:start]])
        folds.append(Fold(len(folds), history, val, test[start : start + stride]))
        start += stride
    return folds


def _drift(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    grown = weights * (1.0 + returns)
    total = grown.sum()
    return grown / total if total > 0 else weights


def run_backtest(
    panel: AlignedPanel,
    config: Config,
    run_id: str,
    *,
    pipeline: str,
    fit_gate: GateFactory | None = None,
    make_policy: PolicyFactory,
    features: dict[str, np.ndarray] | None = None,
    optimizer: DROCVaROptimizer | None = None,
    label: str = "",
) -> Path:
    """Run every fold and persist the weights/ledger artifacts."""
    research = config.research
    backtest = research.backtest
    alpha = research.optimizer.alpha
    rows_out: list[dict] = []
    books: list[dict] = []
    gate_name = "equal_weight"

    for fold in generate_folds(panel, config):
        stack = build_stack(
            panel, config, fold.train, EqualWeightGate(), features, optimizer
        )
        if fit_gate is not None:
            stack.gate = fit_gate(stack, fold.train)
        gate_name = getattr(stack.gate, "name", type(stack.gate).__name__)
        policy = make_policy(stack, fold.train, fold.val)

        weights = stack.equal_weight_book()
        stats: dict = {}
        gate_weights: dict[str, float] = {}
        last = len(panel.dates) - 1
        for step, position in enumerate(fold.test):
            if position >= last:
                break
            rebalance = step % backtest.rebalance_days == 0
            previous = weights
            if rebalance:
                state, scen, gate = stack.state(position, previous)
                params = policy.act(state)
                weights = stack.allocate(state, scen, params)
                stats = state.tail_stats
                gate_weights = gate.g
            turnover = float(np.abs(weights - previous).sum())
            cost = turnover * backtest.cost_bps / 1e4
            realized = panel.returns[position + 1][stack.universe]
            port_return = float(weights @ realized) - cost

            rows_out.append(
                {
                    "date": panel.dates[position + 1],
                    "fold": fold.index,
                    "port_return": port_return,
                    "turnover": turnover,
                    "cost": cost,
                    "realized_cvar": float(stats.get("cvar", np.nan)),
                    "realized_var": float(stats.get("var", np.nan)),
                    "cash": 0.0,
                    "gross": 1.0,
                    **{f"gate_g_{name}": value for name, value in gate_weights.items()},
                }
            )
            books.extend(
                {
                    "date": panel.dates[position],
                    "asset": asset,
                    "weight": float(weight),
                }
                for asset, weight in zip(
                    np.asarray(panel.assets)[stack.universe], weights, strict=True
                )
            )
            weights = _drift(weights, realized)

        logger.info(
            "fold_done index=%d test_rows=%d train_rows=%d",
            fold.index,
            len(fold.test),
            len(fold.train),
        )

    ledger = pd.DataFrame(rows_out)
    meta = {
        "pipeline": pipeline,
        "label": label or run_id,
        "gate": gate_name,
        "news_backend": research.news.backend,
        "policy": getattr(policy, "name", type(policy).__name__),
        "alpha": alpha,
        "splits": research.split.ranges,
        "rebalance_days": backtest.rebalance_days,
        "cost_bps": backtest.cost_bps,
        "seed": backtest.seed,
        "config_hash": config_hash(research),
        "assets": int(stack.n_investable),
    }
    return write_run(
        config.path(backtest.runs), run_id, pd.DataFrame(books), ledger, meta
    )
