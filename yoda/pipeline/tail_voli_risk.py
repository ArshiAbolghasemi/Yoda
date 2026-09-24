"""The full stack with a **static** risk-parameter policy.

    panel -> specialists -> Tail-VoI gate -> t-copula -> DRO-CVaR -> portfolio

Stages 1-4 (align, fit specialists, fit the copula, generate counterfactual
targets and train the gate) happen per fold inside the backtest engine, so the
walk-forward discipline is enforced in one place. Stage 5 trades the fold's test
rows with :class:`~yoda.policy.static.StaticRiskPolicy`.

The question it answers: *does the Tail-VoI gate, driving a static risk policy,
control tail risk?* The RL pipeline is the same stack with SAC in the policy
socket.
"""

from __future__ import annotations

import numpy as np

from yoda.backtest.engine import run_backtest
from yoda.common.alignment import AlignedPanel, build_panel
from yoda.common.logger import logger
from yoda.common.types import DROCVaROptimizer
from yoda.config.settings import Config
from yoda.evaluation.report import Evaluation, evaluate
from yoda.policy.static import StaticRiskPolicy
from yoda.stack import AllocationStack, fit_gate

PIPELINE = "tail_voli_risk"


def run_tail_voli_risk(
    config: Config,
    *,
    gate: str | None = None,
    run_id: str = "tail_voli_risk",
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
        # The seam. A fixed config-driven rule, unless the CIO is installed -
        # it owns the whole decision, so it fills this socket too rather than a
        # second policy contradicting it. The RL pipeline swaps the same call.
        return StaticRiskPolicy(config.research.static_policy, alpha=alpha)

    logger.info("tail_voli_risk_start run=%s", run_id)
    run_backtest(
        panel,
        config,
        run_id,
        pipeline=PIPELINE,
        fit_gate=fit_gate(config, gate),
        make_policy=make_policy,
        features=features,
        optimizer=optimizer,
        label=label or f"static/{gate or config.research.tailvoi.gate}",
    )
    return evaluate(config, run_id, panel=panel, make_plots=make_plots)
