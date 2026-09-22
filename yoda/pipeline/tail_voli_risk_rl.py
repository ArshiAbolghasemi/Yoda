"""The same stack with an **RL** risk-parameter controller.

Stages 1-4 are the static pipeline's, reused verbatim through the same engine and
``fit_gate`` factory - nothing is forked. Then the specialists, copula and gate
are frozen, SAC is trained against
:class:`~yoda.env.allocation_env.AllocationEnv`, and the resulting
:class:`~yoda.policy.rl.RLRiskPolicy` is asked for ``RiskParams`` at exactly the
point where ``tail_voli_risk`` asks its static rule.

Compare this file with ``tail_voli_risk.py``: the diff is ``make_policy``. That is the
whole seam.

The question it answers: *does regime-adaptive risk budgeting on top of the gate
add value over the static risk policy?*
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
from yoda.rl.sac import train_sac

PIPELINE = "tail_voli_risk_rl"


def run_tail_voli_risk_rl(
    config: Config,
    *,
    run_id: str = "tail_voli_risk_rl",
    gate: str = "tailvoi",
    panel: AlignedPanel | None = None,
    features: dict[str, np.ndarray] | None = None,
    optimizer: DROCVaROptimizer | None = None,
    label: str = "",
    make_plots: bool | None = None,
) -> Evaluation:
    """Run the RL pipeline end to end and score it."""
    panel = panel or build_panel(config)
    research = config.research

    def make_policy(stack: AllocationStack, train: np.ndarray, val: np.ndarray):
        # The seam: train SAC on train+val with the stack frozen, then wrap it.
        rows = np.concatenate([train, val])
        return train_sac(
            stack,
            rows,
            research.rl,
            rebalance_days=research.backtest.rebalance_days,
            checkpoint=config.path(f"{research.rl.checkpoint}_{run_id}"),
        )

    logger.info("tail_voli_risk_rl_start run=%s gate=%s", run_id, gate)
    run_backtest(
        panel,
        config,
        run_id,
        pipeline=PIPELINE,
        fit_gate=fit_gate(gate, config),
        make_policy=make_policy,
        features=features,
        optimizer=optimizer,
        label=label or f"rl/{gate}",
    )
    return evaluate(config, run_id, panel=panel, make_plots=make_plots)
