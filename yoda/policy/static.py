"""``tail_voli_risk``'s policy: fixed risk parameters, optionally vol-targeted."""

from __future__ import annotations

import numpy as np

from yoda.common.types import MarketState, RiskParamPolicy, RiskParams
from yoda.config.research import StaticPolicyConfig


class StaticRiskPolicy(RiskParamPolicy):
    """Config-driven ``RiskParams``; no learning, no state dependence by default.

    With ``vol_target > 0`` the budget is rescaled toward that realized-CVaR
    target using the current scenario tail - the classical vol-target rule, and
    the fair static comparison for the RL controller.
    """

    name = "static"

    def __init__(self, config: StaticPolicyConfig | None = None, alpha: float = 0.05):
        self.config = config or StaticPolicyConfig()
        self.alpha = alpha

    def act(self, state: MarketState) -> RiskParams:
        budget = self.config.budget
        if self.config.vol_target > 0:
            current = float(state.tail_stats.get("cvar", budget)) or budget
            budget = float(
                np.clip(
                    budget * self.config.vol_target / current, 0.2 * budget, 5 * budget
                )
            )
        return RiskParams(
            lam=self.config.lam,
            budget=budget,
            turnover_penalty=self.config.turnover_penalty,
            alpha=self.alpha,
        )
