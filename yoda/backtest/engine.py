"""Walk-forward execution.

The engine fits on a fold's training rows, trades the fold's test rows, applies
transaction costs, and writes artifacts. It computes no performance metrics:
that is :mod:`yoda.evaluation`'s job, offline, from the files written here.

No lookahead anywhere. At row ``t`` the CIO sees features up to and including
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
from yoda.cio import CIOAgent, build_cio
from yoda.common.alignment import AlignedPanel
from yoda.common.logger import logger
from yoda.common.types import DROCVaROptimizer, Gate, RiskParamPolicy
from yoda.config.settings import Config
from yoda.stack import resolve_features  # noqa: F401 - re-exported for callers
from yoda.tailvoi.tailvoi_gate import TailVoIGate

GateFactory = Callable[[CIOAgent, np.ndarray], Gate]
PolicyFactory = Callable[[CIOAgent, np.ndarray, np.ndarray], RiskParamPolicy]


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
        cio = build_cio(
            panel,
            config,
            fold.train,
            # Unfitted: it degrades to an even gate, which is what the
            # counterfactual generator needs while it builds the targets.
            TailVoIGate(research.tailvoi),
            features,
            optimizer,
        )
        if fit_gate is not None:
            cio.gate = fit_gate(cio, fold.train)
        gate_name = getattr(cio.gate, "name", type(cio.gate).__name__)
        # The policy is trained against the CIO, then seated inside it.
        cio.install_policy(make_policy(cio, fold.train, fold.val))

        weights = cio.equal_weight_book()
        stats: dict = {}
        gate_weights: dict[str, float] = {}
        delta_hat: dict[str, float] = {}
        params = cio.policy.act(cio.state(int(fold.test[0]), weights)[0])
        value = 1.0
        last = len(panel.dates) - 1
        for step, position in enumerate(fold.test):
            if position >= last:
                break
            rebalance = step % backtest.rebalance_days == 0
            previous = weights
            if rebalance:
                # One call, one decision site: the CIO gates, prices the
                # dependence, picks the risk stance and solves.
                decision = cio.decide(position, previous)
                weights, params = decision.weights, decision.risk
                stats = decision.state.tail_stats
                gate_weights = decision.gate.g
                delta_hat = decision.gate.delta_hat
            turnover = float(np.abs(weights - previous).sum())
            cost = turnover * backtest.cost_bps / 1e4
            realized = panel.returns[position + 1][cio.universe]
            port_return = float(weights @ realized) - cost
            value *= 1.0 + port_return

            rows_out.append(
                {
                    "date": panel.dates[position + 1],
                    "fold": fold.index,
                    "rebalanced": bool(rebalance),
                    "port_return": port_return,
                    "portfolio_value": value,
                    "turnover": turnover,
                    "cost": cost,
                    "realized_cvar": float(stats.get("cvar", np.nan)),
                    "realized_var": float(stats.get("var", np.nan)),
                    "nu": float(stats.get("nu", np.nan)),
                    "scenario_vol": float(stats.get("vol", np.nan)),
                    # The risk parameters the policy chose - static rule or SAC.
                    "rp_lam": params.lam,
                    "rp_budget": params.budget,
                    "rp_turnover_penalty": params.turnover_penalty,
                    "rp_alpha": params.alpha,
                    "cash": 0.0,
                    "gross": 1.0,
                    **{f"gate_g_{name}": share for name, share in gate_weights.items()},
                    **{f"tail_voi_{name}": share for name, share in delta_hat.items()},
                }
            )
            books.extend(
                {
                    "date": panel.dates[position],
                    "asset": asset,
                    "weight": float(weight),
                    "weight_prev": float(before),
                    "weight_change": float(weight - before),
                }
                for asset, weight, before in zip(
                    np.asarray(panel.assets)[cio.universe],
                    weights,
                    previous,
                    strict=True,
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
        "sources": list(cio.sources),
        "backends": dict.fromkeys(cio.sources, "openjev"),
        "synthetic": research.jev.synthetic,
        "news_backend": research.news.backend,
        "target_horizon": research.panel.target_horizon,
        "policy": getattr(cio.policy, "name", type(cio.policy).__name__),
        "alpha": alpha,
        "splits": research.split.ranges,
        "rebalance_days": backtest.rebalance_days,
        "cost_bps": backtest.cost_bps,
        "seed": backtest.seed,
        "config_hash": config_hash(research),
        "assets": int(cio.n_investable),
    }
    return write_run(
        config.path(backtest.runs), run_id, pd.DataFrame(books), ledger, meta
    )
