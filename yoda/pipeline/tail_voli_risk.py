"""The full stack with a **static** risk-parameter policy.

    panel -> specialists -> Tail-VoI gate -> t-copula -> DRO-CVaR -> portfolio

Stages 1-4 (align, fit specialists, fit the copula, generate counterfactual
targets and train the gate) happen per fold inside the backtest engine, so the
walk-forward discipline is enforced in one place. Stage 5 trades the fold's test
rows with :class:`~yoda.policy.static.StaticRiskPolicy`.

The question it answers: *does gating on Tail-VoI beat the baseline gates and the
classical allocators?*
"""

from __future__ import annotations

import numpy as np

from yoda.backtest.engine import run_backtest
from yoda.common.alignment import AlignedPanel, build_panel
from yoda.common.logger import logger
from yoda.common.types import DROCVaROptimizer
from yoda.config.settings import Config
from yoda.evaluation.report import Evaluation, evaluate
from yoda.pipeline.stack import AllocationStack, fit_gate
from yoda.policy.static import StaticRiskPolicy

PIPELINE = "tail_voli_risk"


def run_tail_voli_risk(
    config: Config,
    *,
    run_id: str = "tail_voli_risk",
    gate: str = "tailvoi",
    panel: AlignedPanel | None = None,
    features: dict[str, np.ndarray] | None = None,
    optimizer: DROCVaROptimizer | None = None,
    label: str = "",
    make_plots: bool | None = None,
) -> Evaluation:
    """Run the static pipeline end to end and score it."""
    panel = panel or build_panel(config)
    alpha = config.research.optimizer.alpha

    def make_policy(stack: AllocationStack, train: np.ndarray, val: np.ndarray):
        # The seam: a fixed, config-driven rule. The RL pipeline swaps this call.
        return StaticRiskPolicy(config.research.static_policy, alpha=alpha)

    logger.info("tail_voli_risk_start run=%s gate=%s", run_id, gate)
    run_backtest(
        panel,
        config,
        run_id,
        pipeline=PIPELINE,
        fit_gate=fit_gate(gate, config),
        make_policy=make_policy,
        features=features,
        optimizer=optimizer,
        label=label or f"static/{gate}",
    )
    return evaluate(config, run_id, panel=panel, make_plots=make_plots)
