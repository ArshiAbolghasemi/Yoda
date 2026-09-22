"""Exact counterfactual Tail-VoI targets.

For each information source ``i`` the whole downstream stack is re-solved with
that source removed, and the difference in the resulting portfolio's tail risk
is the target::

    Delta_i = rho(I_-i) - rho(I)

A positive ``Delta_i`` means the book you would have held *without* source ``i``
was worse in the tail - the source earned its place. Each source is ablated at
the stage it actually enters: dropping ``technical`` or ``news`` removes a
contributor to ``fused_mu``, while dropping ``volatility`` de-conditions the
copula and therefore changes the scenario set, not the mean.

This is leave-one-out counterfactual risk attribution over *information sources*
- the close relative of Shapley-style expected-shortfall attribution, which is
where it should be positioned in the paper. The formula itself is not the
contribution; what the gate does with it is.

It is expensive (``1 + n_sources`` full solves per state), so it runs offline
over training rows only and its output is the gate's supervised training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yoda.common.logger import logger
from yoda.common.types import RiskParams
from yoda.config.research import StaticPolicyConfig, TailVoIConfig
from yoda.tailvoi.base import summarise


@dataclass(frozen=True)
class CounterfactualTargets:
    summaries: np.ndarray  # (M, D) gate inputs
    deltas: np.ndarray  # (M, K) Delta_i per source
    sources: tuple[str, ...]
    positions: np.ndarray  # (M,) panel rows the states came from

    def frame_summary(self) -> dict[str, float]:
        return {
            f"delta_{name}": float(self.deltas[:, index].mean())
            for index, name in enumerate(self.sources)
        }


def _realized_cvar(returns: np.ndarray, w: np.ndarray, alpha: float) -> float:
    if returns.size == 0:
        return 0.0
    losses = -(returns @ w)
    var = float(np.quantile(losses, 1 - alpha))
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else var


def sample_positions(rows: np.ndarray, n_states: int, tail_margin: int) -> np.ndarray:
    """Evenly spaced training rows that still have a full forward window."""
    usable = rows[: max(len(rows) - tail_margin, 1)]
    if len(usable) <= n_states:
        return usable
    return usable[np.linspace(0, len(usable) - 1, n_states).astype(int)]


def counterfactual_deltas(
    stack,
    rows: np.ndarray,
    config: TailVoIConfig,
    policy: StaticPolicyConfig,
) -> CounterfactualTargets:
    """Generate ``(state summary, Delta)`` pairs over the training window.

    ``stack`` is any object exposing the :class:`~yoda.pipeline.stack.AllocationStack`
    surface: ``gate_at``, ``scenarios``, ``allocate``-style solving, ``universe``
    and ``forward_returns``. It is duck-typed on purpose so this module stays
    independent of the orchestration layer.
    """
    sources = tuple(stack.sources)
    alpha = stack.config.research.optimizer.alpha
    rp = RiskParams(
        lam=policy.lam,
        budget=policy.budget,
        turnover_penalty=policy.turnover_penalty,
        alpha=alpha,
    )
    margin = config.rho_window if config.rho == "realized" else 1
    positions = sample_positions(rows, config.n_states, margin)
    reference_book = stack.equal_weight_book()
    n = config.counterfactual_scenarios

    summaries, deltas = [], []
    for position in positions:
        spec, gate_full = stack.gate_at(position, stack.equal_weights)
        scen_full = stack.scenarios(spec, gate_full, n)
        w_full = stack.optimizer.solve(
            gate_full.fused_mu, scen_full, rp, reference_book
        )
        forward = stack.forward_returns(position, config.rho_window)

        def rho(w: np.ndarray, forward=forward, scen_full=scen_full) -> float:
            if config.rho == "realized":
                return _realized_cvar(forward, w, alpha)
            # "reference": score every counterfactual book under one common
            # full-information measure, which removes the scenario-set noise.
            return float(
                np.mean(
                    np.sort(-(scen_full.scenarios @ w))[
                        -max(int(alpha * len(scen_full.scenarios)), 1) :
                    ]
                )
            )

        baseline = rho(w_full)
        row = []
        for dropped in sources:
            remaining = [name for name in sources if name != dropped]
            weights = {name: 1.0 / len(remaining) for name in remaining}
            weights[dropped] = 0.0
            _, gate_without = stack.gate_at(position, weights)
            scen_without = stack.scenarios(spec, gate_without, n)
            w_without = stack.optimizer.solve(
                gate_without.fused_mu, scen_without, rp, reference_book
            )
            row.append(rho(w_without) - baseline)
        summaries.append(summarise(spec, sources))
        deltas.append(row)

    targets = CounterfactualTargets(
        summaries=np.asarray(summaries, dtype=np.float64),
        deltas=np.asarray(deltas, dtype=np.float64),
        sources=sources,
        positions=np.asarray(positions),
    )
    logger.info(
        "counterfactual_targets states=%d rho=%s mean=%s",
        len(positions),
        config.rho,
        targets.frame_summary(),
    )
    return targets
