"""Gym environment for the risk-parameter controller (RL pipeline only).

The specialists, copula and gate are **frozen**. The agent sees the gated market
state and chooses ``(lam, B, turnover_penalty)``; the same DRO-CVaR solver turns
that into a book. The agent never emits weights, and ``alpha`` is not in the
action space.

Reward::

    r_t = R_p,t - lam_cvar * CVaR_alpha,t - lam_dd * DD_t - c * ||w_t - w_t-1||_1

``lam_cvar`` here is a **fixed reward-shaping weight** from ``RL__REWARD_LAMBDA_CVAR``.
It is a different quantity from the ``lam`` the policy chooses for the optimizer
objective, and the two must never be conflated - one shapes learning, the other
is the action.

Synthetic mode replaces the realised return with a draw from the conditional
copula while keeping the real state sequence: it enlarges the effective sample
over tail paths the 2018-2026 history only contains once. The states are still
real - only the outcomes are simulated - and that limitation is deliberate.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from yoda.common.logger import logger
from yoda.config.research import RLConfig
from yoda.policy.rl import ACTION_DIM, action_to_params


class AllocationEnv(gym.Env):
    """One episode = one walk through the training rows at the rebalance stride."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        stack,
        rows: np.ndarray,
        config: RLConfig,
        *,
        rebalance_days: int = 5,
        synthetic: bool = False,
        max_steps: int = 256,
    ):
        super().__init__()
        self.stack = stack
        self.rows = np.asarray(rows)
        self.config = config
        self.rebalance_days = max(rebalance_days, 1)
        self.synthetic = synthetic
        self.max_steps = max_steps
        self.alpha = stack.config.research.optimizer.alpha
        self.sources = tuple(stack.sources)

        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        probe = self._observe(int(self.rows[0]), stack.equal_weight_book())[0]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, probe.shape, dtype=np.float32
        )
        logger.info(
            "env_ready rows=%d obs_dim=%d synthetic=%s",
            len(self.rows),
            probe.size,
            synthetic,
        )

    # ---- gym API ---------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        span = self.max_steps * self.rebalance_days
        last = max(len(self.rows) - span - 1, 1)
        self.cursor = int(self.np_random.integers(0, last))
        self.weights = self.stack.equal_weight_book()
        self.equity = 1.0
        self.peak = 1.0
        self.steps = 0
        observation, self.state, self.scenarios = self._observe(
            self._position(), self.weights
        )
        return observation.astype(np.float32), {}

    def step(self, action: np.ndarray):
        params = action_to_params(action, self.config, self.alpha)
        previous = self.weights
        self.weights = self.stack.allocate(self.state, self.scenarios, params)
        turnover = float(np.abs(self.weights - previous).sum())

        realised = self._realised_return(self._position())
        self.equity *= 1.0 + realised
        self.peak = max(self.peak, self.equity)
        drawdown = self.peak / self.equity - 1.0
        cvar = float(self.state.tail_stats.get("cvar", 0.0))
        reward = (
            realised
            - self.config.reward_lambda_cvar * cvar
            - self.config.reward_lambda_dd * drawdown
            - self.config.reward_turnover_cost * turnover
        )

        self.steps += 1
        self.cursor += self.rebalance_days
        terminated = self.cursor >= len(self.rows) - 1
        truncated = self.steps >= self.max_steps
        if terminated or truncated:
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                float(reward),
                terminated,
                truncated,
                {"turnover": turnover, "cvar": cvar},
            )
        observation, self.state, self.scenarios = self._observe(
            self._position(), self.weights
        )
        return (
            observation.astype(np.float32),
            float(reward),
            False,
            False,
            {"turnover": turnover, "cvar": cvar},
        )

    # ---- internals -------------------------------------------------------

    def _position(self) -> int:
        return int(self.rows[min(self.cursor, len(self.rows) - 1)])

    def _observe(self, position: int, weights: np.ndarray):
        state, scenarios, _ = self.stack.state(
            position, weights, n_scenarios=self.config.env_scenarios
        )
        return state.observation(self.sources), state, scenarios

    def _realised_return(self, position: int) -> float:
        """Portfolio return over the holding period, real or copula-simulated."""
        if self.synthetic:
            draws = self.np_random.integers(
                0, len(self.scenarios.scenarios), size=self.rebalance_days
            )
            window = self.scenarios.scenarios[draws]
            return float(np.prod(1.0 + window @ self.weights) - 1.0)
        stop = min(position + 1 + self.rebalance_days, len(self.stack.panel.dates))
        window = self.stack.panel.returns[position + 1 : stop][:, self.stack.universe]
        if window.size == 0:
            return 0.0
        return float(np.prod(1.0 + window @ self.weights) - 1.0)
