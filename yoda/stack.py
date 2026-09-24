"""The shared stack: everything below the ``RiskParamPolicy`` seam.

    specialists -> gate -> copula conditioning -> scenarios -> DRO-CVaR

Both pipelines build the *same* ``AllocationStack``; they differ only
in which policy is asked for the ``RiskParams`` that the last step consumes.
A direct-weight policy, when it arrives, replaces the last step and touches
nothing here.

Everything is fitted on a fold's training rows only. ``build_stack`` takes the
row indices explicitly so the caller - never this module - owns the walk-forward
discipline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from yoda.common.alignment import AlignedPanel
from yoda.common.logger import logger
from yoda.common.types import (
    DROCVaROptimizer,
    Gate,
    GateOutput,
    MarketState,
    RiskParams,
    Specialist,
    SpecialistOutput,
    TailScenarios,
)
from yoda.config.settings import Config
from yoda.copula.student_t import StudentTCopula, tail_stats
from yoda.optimizer.dro_cvar import DROCVaR
from yoda.specialists.jev import build_jev_features
from yoda.specialists.news import NewsSpecialist
from yoda.specialists.technical import TechnicalSpecialist
from yoda.specialists.volatility import VolatilitySpecialist
from yoda.tailvoi.base import conditioning_strength, fuse, summarise
from yoda.tailvoi.baselines import AccuracyGate, AttentionGate, EqualWeightGate
from yoda.tailvoi.counterfactual import counterfactual_deltas, sample_positions
from yoda.tailvoi.tailvoi_gate import TailVoIGate

SPECIALIST_TYPES: dict[str, type[Specialist]] = {
    "technical": TechnicalSpecialist,
    "volatility": VolatilitySpecialist,
    "news": NewsSpecialist,
}


@dataclass
class AllocationStack:
    """A fitted stack, evaluable at any row of the panel."""

    panel: AlignedPanel
    config: Config
    specialists: dict[str, Specialist]
    copula: StudentTCopula
    gate: Gate
    optimizer: DROCVaROptimizer
    universe: np.ndarray  # (N,) bool: the fold's investable assets
    features: dict[str, np.ndarray]  # source -> (T, N, F), news cube included
    equal_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sources = tuple(self.specialists)
        share = 1.0 / len(self.sources)
        self.equal_weights = dict.fromkeys(self.sources, share)
        self.n_investable = int(self.universe.sum())

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


def resolve_features(
    panel: AlignedPanel, config: Config, features: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Materialise the feature cube for each configured channel.

    Every channel's features are OpenJev's calibrated probabilities. The panel's
    indicator cubes are still built, but they are the *state* each decision task
    is shown, not the specialist's input.

    Pre-built cubes passed in ``features`` are never rebuilt: the experiment
    harness uses that to share one expensive Jev build across many arms.
    """
    research = config.research
    supplied = features or {}
    resolved: dict[str, np.ndarray] = {}
    for channel in research.tailvoi.sources:
        if channel == "news" and research.news.backend == "none":
            continue
        if channel in supplied:
            resolved[channel] = supplied[channel]
            continue
        if channel != "news" and research.specialists.backend(channel) == "numeric":
            # The conventional arm: the raw indicator cube the panel already
            # carries, which is also what the OpenJev state is built from.
            resolved[channel] = panel.features[channel]
            continue
        resolved[channel], _ = build_jev_features(panel, config, channel)
    return resolved


def build_stack(
    panel: AlignedPanel,
    config: Config,
    train_rows: np.ndarray,
    gate: Gate,
    features: dict[str, np.ndarray] | None = None,
    optimizer: DROCVaROptimizer | None = None,
) -> AllocationStack:
    """Fit specialists and the copula on ``train_rows`` and assemble the stack."""
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
            model=getattr(research.specialists, f"{name}_model", None),
        )
        specialist.fit(features[name][train_rows][:, universe], targets)
        specialists[name] = specialist

    copula = StudentTCopula(research.copula)
    copula.fit(panel.returns[train_rows][:, universe])

    allocator = optimizer or DROCVaR(research.optimizer)

    stack = AllocationStack(
        panel=panel,
        config=config,
        specialists=specialists,
        copula=copula,
        gate=gate,
        optimizer=allocator,
        universe=universe,
        features=features,
    )
    logger.info(
        "stack_built sources=%s assets=%d train_rows=%d gate=%s",
        ",".join(stack.sources),
        stack.n_investable,
        len(train_rows),
        getattr(gate, "name", type(gate).__name__),
    )
    return stack


def fit_gate(
    config: Config, kind: str | None = None
) -> Callable[[AllocationStack, np.ndarray], Gate]:
    """Build the gate the CIO will contain: ``(stack, train_rows) -> Gate``.

    ``kind`` defaults to ``TAILVOI__GATE``. Tail-VoI is the proposal; the other
    three are the controls it has to beat, and each closes a specific escape
    route - "does gating do anything at all", "is this just accuracy weighting",
    "is this just any fitted gate beating a fixed one".

    Tail-VoI is fitted here rather than up front because it needs the
    counterfactual generator to have run, which needs a stack: the engine hands
    over one already fitted on the fold's training rows.
    """
    research = config.research
    kind = kind or research.tailvoi.gate

    def build(stack: AllocationStack, rows: np.ndarray) -> Gate:
        if kind == "equal_weight":
            return EqualWeightGate()

        if kind == "accuracy":
            gate = AccuracyGate(temperature=research.tailvoi.temperature)
            views = {
                name: specialist.predict(stack.features[name][rows][:, stack.universe])
                for name, specialist in stack.specialists.items()
            }
            gate.fit(views, stack.panel.targets[rows][:, stack.universe])
            return gate

        if kind == "attention":
            positions = sample_positions(rows, research.tailvoi.n_states, 1)
            outputs = [stack.specialist_output(position) for position in positions]
            sources = tuple(stack.sources)
            gate = AttentionGate(seed=research.tailvoi.seed)
            gate.fit(
                np.stack([summarise(spec, sources) for spec in outputs]),
                np.stack(
                    [np.stack([spec.mu_hat[n] for n in sources]) for spec in outputs]
                ),
                stack.panel.targets[positions][:, stack.universe],
                sources,
            )
            return gate

        if kind == "tailvoi":
            targets = counterfactual_deltas(
                stack, rows, research.tailvoi, research.static_policy
            )
            gate = TailVoIGate(research.tailvoi)
            gate.fit(targets)
            return gate

        raise ValueError(f"Unknown gate: {kind!r}")

    return build
