"""The CIO: the one decision-maker, and everything it decides with.

Every specialist reports here. The CIO holds the world model it needs to read
those reports and the machinery to act on them, and exposes one thing —
weights:

```
    technical ─┐
    volatility ┼─► CIO ─────────────────────────────┐
    news ──────┘   │  Tail-VoI gate     → g         │
                   │  Student-t copula  → scenarios │
                   │  risk policy       → (λ, B, c) │  static or SAC, by flag
                   │  DRO-CVaR          → w         │
                   └────────────────────────────────┘
                                  │
                          portfolio weights
```

There is exactly one gate seam and one allocator. The only thing that varies is
the risk policy, chosen by the pipeline flag: the static rule, or the trained
SAC controller.

**Why one object.** Gating the views, pricing the dependence, choosing how much
risk to carry and building the book are a single decision. Split across
collaborators they drift: the same decision gets made twice in two places, and
the copies disagree. Behind one call there is one decision site.

**Construction is in phases**, because the gate is fitted from counterfactuals
that run through the CIO itself:

    cio = build_cio(...)               # specialists + copula, fitted on train
    cio.gate = fit_gate(config)(cio, train_rows)
    cio.install_policy(policy)         # static rule, or SAC trained against it

That is not circularity - the counterfactual generator forces gate weights
rather than asking the gate - but it does mean the CIO is usable before its
gate is fitted, and deliberately so.

**Why ``state`` is public.** SAC trains *against* the risk policy, so the RL
environment needs the observation the policy would act on, without a policy
having acted. ``state`` is that surface; ``decide`` is ``state`` plus the two
judgements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from yoda.common.alignment import AlignedPanel
from yoda.common.logger import logger
from yoda.common.types import (
    DROCVaROptimizer,
    Gate,
    GateOutput,
    MarketState,
    RiskParamPolicy,
    RiskParams,
    Specialist,
    SpecialistOutput,
    TailScenarios,
)
from yoda.config.settings import Config
from yoda.copula.student_t import StudentTCopula, tail_stats
from yoda.optimizer.dro_cvar import DROCVaR
from yoda.specialists.news import NewsSpecialist
from yoda.specialists.technical import TechnicalSpecialist
from yoda.specialists.volatility import VolatilitySpecialist
from yoda.stack import resolve_features
from yoda.tailvoi.base import conditioning_strength, fuse

SPECIALIST_TYPES: dict[str, type[Specialist]] = {
    "technical": TechnicalSpecialist,
    "volatility": VolatilitySpecialist,
    "news": NewsSpecialist,
}


@dataclass(frozen=True)
class CIODecision:
    """What the CIO decided, and why. Everything here is persisted."""

    weights: np.ndarray
    gate: GateOutput
    risk: RiskParams
    state: MarketState


@dataclass
class CIOAgent:
    """Gate, copula, risk policy and DRO-CVaR behind a single call."""

    name = "cio"

    panel: AlignedPanel
    config: Config
    specialists: dict[str, Specialist]
    copula: StudentTCopula
    gate: Gate
    optimizer: DROCVaROptimizer
    universe: np.ndarray  # (N,) bool: the fold's investable assets
    features: dict[str, np.ndarray]  # source -> (T, N, F), news cube included
    policy: RiskParamPolicy | None = None
    equal_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sources = tuple(self.specialists)
        share = 1.0 / len(self.sources)
        self.equal_weights = dict.fromkeys(self.sources, share)
        self.n_investable = int(self.universe.sum())

    # ---- the decision ----------------------------------------------------

    def install_policy(self, policy: RiskParamPolicy) -> CIOAgent:
        """Seat the risk controller. The one seam between the two pipelines."""
        self.policy = policy
        logger.info("cio_ready %s", self.describe())
        return self

    def decide(self, position: int, w_prev: np.ndarray) -> CIODecision:
        """Gate the views, choose the risk stance, then build the book."""
        if self.policy is None:
            raise RuntimeError("CIO has no risk policy; call install_policy first")
        state, scen, gate = self.state(position, w_prev)
        risk = self.policy.act(state)
        weights = self.allocate(state, scen, risk)
        return CIODecision(weights=weights, gate=gate, risk=risk, state=state)

    def describe(self) -> str:
        def label(component) -> str:
            return getattr(component, "name", type(component).__name__)

        return (
            f"gate={label(self.gate)} policy={label(self.policy)} "
            f"optimizer={label(self.optimizer)} sources={','.join(self.sources)}"
        )

    # ---- per-date evaluation --------------------------------------------

    def specialist_output(self, position: int) -> SpecialistOutput:
        """Encode one date into per-source ``z_i`` and ``mu_hat_i``."""
        z, mu_hat = {}, {}
        for name, specialist in self.specialists.items():
            X = self.features[name][position][self.universe]
            z[name] = specialist.encode(X)
            mu_hat[name] = specialist.predict(X)
        return SpecialistOutput(z=z, mu_hat=mu_hat)

    def scenarios(
        self, spec: SpecialistOutput, gate: GateOutput, n: int | None = None
    ) -> TailScenarios:
        """Condition the copula on the gated volatility view and sample."""
        n = n or self.config.research.copula.n_scenarios
        strength = conditioning_strength(gate)
        unconditional = self.copula.scale
        if strength <= 0 or "volatility" not in spec.mu_hat:
            return self.copula.conditional(self._regime_state(None)).sample(n)
        sigma = np.asarray(spec.mu_hat["volatility"], dtype=np.float64).reshape(-1)
        # E|r| from the specialist; rescale to a standard deviation.
        sigma = sigma * np.sqrt(np.pi / 2)
        regime = (1 - strength) * unconditional + strength * sigma
        return self.copula.conditional(self._regime_state(regime)).sample(n)

    def _regime_state(self, regime: np.ndarray | None) -> MarketState:
        zeros = np.zeros(self.n_investable)
        return MarketState(
            fused_mu=zeros,
            fused_z=zeros,
            g={},
            tail_stats={},
            w_prev=zeros,
            regime=regime,
        )

    def gate_at(
        self, position: int, weights: dict[str, float] | None = None
    ) -> tuple[SpecialistOutput, GateOutput]:
        """Gate one date, optionally forcing the weights (counterfactual use)."""
        spec = self.specialist_output(position)
        gate = fuse(spec, weights) if weights is not None else self.gate.gate(spec)
        return spec, gate

    def state(
        self,
        position: int,
        w_prev: np.ndarray,
        weights: dict[str, float] | None = None,
        n_scenarios: int | None = None,
    ) -> tuple[MarketState, TailScenarios, GateOutput]:
        """The full observation a policy acts on, plus the scenario set."""
        spec, gate = self.gate_at(position, weights)
        scen = self.scenarios(spec, gate, n_scenarios)
        stats = tail_stats(w_prev, scen, self.config.research.optimizer.alpha)
        state = MarketState(
            fused_mu=gate.fused_mu,
            fused_z=gate.fused_z.mean(axis=0),
            g=gate.g,
            tail_stats=stats,
            w_prev=w_prev,
            regime=None,
        )
        return state, scen, gate

    def allocate(
        self, state: MarketState, scen: TailScenarios, rp: RiskParams
    ) -> np.ndarray:
        return self.optimizer.solve(state.fused_mu, scen, rp, state.w_prev)

    # ---- helpers used by the counterfactual generator ---------------------

    def equal_weight_book(self) -> np.ndarray:
        return np.full(self.n_investable, 1.0 / self.n_investable)

    def forward_returns(self, position: int, window: int) -> np.ndarray:
        """Realised returns over ``(position, position + window]``, universe only."""
        stop = min(position + 1 + window, len(self.panel.dates))
        return self.panel.returns[position + 1 : stop][:, self.universe]

    def expand(self, weights: np.ndarray) -> np.ndarray:
        """Universe-space weights back into full asset space."""
        full = np.zeros(self.panel.n_assets)
        full[self.universe] = weights
        return full


def build_cio(
    panel: AlignedPanel,
    config: Config,
    train_rows: np.ndarray,
    gate: Gate,
    features: dict[str, np.ndarray] | None = None,
    optimizer: DROCVaROptimizer | None = None,
) -> CIOAgent:
    """Fit specialists and the copula on ``train_rows`` and assemble the CIO.

    The gate arrives unfitted and the policy is absent: both are installed
    afterwards, because fitting the gate needs counterfactuals that run through
    this object, and training SAC needs an environment built on it.
    """
    research = config.research
    features = resolve_features(panel, config, features)

    universe = panel.universe(train_rows)
    if universe.sum() < 2:
        raise ValueError("Fewer than two investable assets in the training window")

    specialists: dict[str, Specialist] = {}
    targets = panel.targets[train_rows][:, universe]
    for name in research.tailvoi.sources:
        if name not in features:
            continue
        specialist = SPECIALIST_TYPES[name](
            z_dim=research.specialists.z_dim,
            ridge_alpha=research.specialists.ridge_alpha,
            seed=research.specialists.seed,
        )
        specialist.fit(features[name][train_rows][:, universe], targets)
        specialists[name] = specialist

    copula = StudentTCopula(research.copula)
    copula.fit(panel.returns[train_rows][:, universe])

    cio = CIOAgent(
        panel=panel,
        config=config,
        specialists=specialists,
        copula=copula,
        gate=gate,
        optimizer=optimizer or DROCVaR(research.optimizer),
        universe=universe,
        features=features,
    )
    logger.info(
        "cio_built sources=%s assets=%d train_rows=%d gate=%s",
        ",".join(cio.sources),
        cio.n_investable,
        len(train_rows),
        getattr(gate, "name", type(gate).__name__),
    )
    return cio
