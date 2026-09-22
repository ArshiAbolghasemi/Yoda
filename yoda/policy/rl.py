"""The RL pipeline's policy: a trained SAC agent choosing the risk parameters.

The agent's action space is ``(lam, B, turnover_penalty)`` and nothing else. It
never emits portfolio weights - the same DRO-CVaR solver produces those - and it
never touches ``alpha``, which stays a fixed hyperparameter of the risk measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yoda.common.logger import logger
from yoda.common.types import MarketState, RiskParamPolicy, RiskParams
from yoda.config.research import RLConfig

ACTION_DIM = 3


def action_to_params(action: np.ndarray, config: RLConfig, alpha: float) -> RiskParams:
    """Map a ``[-1, 1]^3`` action onto the configured risk-parameter ranges."""
    unit = (np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1) + 1) / 2
    lam_low, lam_high = config.lam_bounds
    budget_low, budget_high = config.budget_bounds
    turnover_low, turnover_high = config.turnover_bounds
    return RiskParams(
        lam=float(lam_low + unit[0] * (lam_high - lam_low)),
        budget=float(budget_low + unit[1] * (budget_high - budget_low)),
        turnover_penalty=float(turnover_low + unit[2] * (turnover_high - turnover_low)),
        alpha=alpha,
    )


class RLRiskPolicy(RiskParamPolicy):
    """Wraps a trained SAC checkpoint behind the risk-parameter seam."""

    name = "rl"

    def __init__(self, model, config: RLConfig, alpha: float = 0.05, sources=None):
        self.model = model
        self.config = config
        self.alpha = alpha
        self.sources = tuple(sources) if sources else None

    @classmethod
    def load(
        cls, path: str | Path, config: RLConfig, alpha: float = 0.05, sources=None
    ):
        from stable_baselines3 import SAC

        model = SAC.load(str(path))
        logger.info("rl_policy_loaded path=%s", path)
        return cls(model, config, alpha, sources)

    def act(self, state: MarketState) -> RiskParams:
        observation = (
            state.observation(self.sources) if self.sources else state.observation()
        )
        action, _ = self.model.predict(observation, deterministic=True)
        return action_to_params(action, self.config, self.alpha)
