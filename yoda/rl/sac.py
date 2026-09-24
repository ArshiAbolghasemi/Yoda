"""SAC training for the risk-parameter controller (RL pipeline only).

Off-policy SAC over a continuous 3-dim action, trained against
:class:`~yoda.rl.allocation_env.AllocationEnv`. Stable-Baselines3 supplies the
algorithm - there is nothing research-novel in the SAC implementation itself and
no reason to hand-roll one.

Optional copula-generated episodes run as a second training pass, which is how
the agent sees more than the single major crash the real sample contains.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from yoda.common.logger import logger
from yoda.config.research import RLConfig
from yoda.policy.rl import RLRiskPolicy
from yoda.rl.allocation_env import AllocationEnv


def train_sac(
    stack,
    rows: np.ndarray,
    config: RLConfig,
    *,
    rebalance_days: int = 5,
    checkpoint: Path | None = None,
) -> RLRiskPolicy:
    """Train on real episodes, optionally top up on synthetic ones, and wrap."""
    env = AllocationEnv(stack, rows, config, rebalance_days=rebalance_days)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        seed=config.seed,
        verbose=0,
    )
    logger.info("sac_train_start steps=%d rows=%d", config.total_timesteps, len(rows))
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)

    if config.synthetic_episodes > 0:
        synthetic = AllocationEnv(
            stack, rows, config, rebalance_days=rebalance_days, synthetic=True
        )
        model.set_env(synthetic)
        extra = config.synthetic_episodes * synthetic.max_steps
        logger.info("sac_synthetic_pass steps=%d", extra)
        model.learn(total_timesteps=extra, reset_num_timesteps=False)
        model.set_env(env)

    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint))
        logger.info("sac_saved path=%s", checkpoint)

    return RLRiskPolicy(
        model,
        config,
        alpha=stack.config.research.optimizer.alpha,
        sources=stack.sources,
    )
