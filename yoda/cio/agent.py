"""The CIO: the decision-maker that turns specialist views into a portfolio.

It *contains* the two judgements and the solver, and exposes one thing —
weights:

```
                  CIO
    ┌───────────────────────────────┐
    │  Tail-VoI gate   → g          │
    │  risk policy     → (λ, B, c)  │   static or SAC, by flag
    │  DRO-CVaR        → w          │
    └───────────────────────────────┘
                  │
          portfolio weights
```

There is exactly one gate (Tail-VoI) and one allocator (DRO-CVaR). The only
thing that varies inside the CIO is the risk policy, chosen by the pipeline
flag: the static rule, or the trained SAC controller.

Keeping the three steps inside one object is what makes the seam legible from
outside: callers ask for weights, and the CIO is where the decision actually
happens. It still records what it decided — the gate weights, the predicted
Tail-VoI per source and the risk parameters — because the ledger needs them and
because a decision you cannot inspect is not one you can trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yoda.common.logger import logger
from yoda.common.types import (
    DROCVaROptimizer,
    GateOutput,
    MarketState,
    RiskParamPolicy,
    RiskParams,
    SpecialistOutput,
    TailScenarios,
)
from yoda.tailvoi.tailvoi_gate import TailVoIGate


@dataclass(frozen=True)
class CIODecision:
    """What the CIO decided, and why. Everything here is persisted."""

    weights: np.ndarray
    gate: GateOutput
    risk: RiskParams


class CIOAgent:
    """Tail-VoI gate + risk policy + DRO-CVaR, behind a single call."""

    name = "cio"

    def __init__(
        self,
        gate: TailVoIGate,
        policy: RiskParamPolicy,
        optimizer: DROCVaROptimizer,
    ):
        self.gate = gate
        self.policy = policy
        self.optimizer = optimizer

    def decide(
        self,
        spec: SpecialistOutput,
        scen: TailScenarios,
        state: MarketState,
        w_prev: np.ndarray,
    ) -> CIODecision:
        """Gate the views, choose the risk stance, then build the book."""
        risk = self.policy.act(state)
        weights = self.optimizer.solve(state.fused_mu, scen, risk, w_prev)
        return CIODecision(weights=weights, gate=state_gate(state, spec), risk=risk)

    def describe(self) -> str:
        def label(component) -> str:
            return getattr(component, "name", type(component).__name__)

        return (
            f"gate={label(self.gate)} policy={label(self.policy)} "
            f"optimizer={label(self.optimizer)}"
        )


def state_gate(state: MarketState, spec: SpecialistOutput) -> GateOutput:
    """Recover the gate output that produced ``state``.

    The stack gates once when it builds the state; re-gating here would pay for
    the same decision twice and risk the two disagreeing.
    """
    return GateOutput(
        g=dict(state.g),
        fused_mu=state.fused_mu,
        fused_z=np.atleast_2d(state.fused_z),
        delta_hat=getattr(state, "delta_hat", {}) or {},
    )


def build_cio(stack, policy: RiskParamPolicy) -> CIOAgent:
    """Compose the CIO from a fitted stack and the chosen risk policy."""
    agent = CIOAgent(gate=stack.gate, policy=policy, optimizer=stack.optimizer)
    logger.info("cio_built %s", agent.describe())
    return agent
