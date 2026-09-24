"""The structured view every agent returns, and the common representation.

All three agents expose the **same six core variables**, which is what makes
Tail-VoI computable over them at all — the gate has to compare channels, and it
can only do that if they speak one language:

    (mu, m, c, u, tau, r)

    mu   direction            expected_return_score / directional_bias   [-1, 1]
    m    magnitude            expected_magnitude / volatility_score      [ 0, 1]
    c    confidence           confidence                                 [ 0, 1]
    u    epistemic uncertainty epistemic_uncertainty                     [ 0, 1]
    tau  downside tail risk   downside_tail_risk                         [ 0, 1]
    r    regime shift         regime_shift_probability                   [ 0, 1]

Each agent then adds its own channel-specific fields on top. The numeric fields
become ``z_i`` and feed the gate, the copula and the optimizer; the prose
(``view_summary``, ``key_evidence``) is kept for the CIO and never reaches the
numeric path.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CORE: tuple[str, ...] = (
    "expected_return_score",
    "expected_magnitude",
    "confidence",
    "epistemic_uncertainty",
    "downside_tail_risk",
    "regime_shift_probability",
)
DIRECTIONS = Literal[
    "strong_bearish", "bearish", "neutral", "bullish", "strong_bullish"
]
REGIMES = Literal["low", "normal", "elevated", "high", "extreme"]


class RiskScenario(BaseModel):
    scenario: str = ""
    direction: Literal["downside", "upside"] | None = None
    severity: float = Field(0.0, ge=0.0, le=1.0)
    probability: float = Field(0.0, ge=0.0, le=1.0)


class AgentView(BaseModel):
    """The fields every agent shares. Subclasses add channel-specific ones."""

    agent: str = ""
    asset: str = ""
    horizon_days: int = 0

    expected_return_score: float = Field(0.0, ge=-1.0, le=1.0)
    expected_magnitude: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    epistemic_uncertainty: float = Field(1.0, ge=0.0, le=1.0)
    downside_tail_risk: float = Field(0.0, ge=0.0, le=1.0)
    regime_shift_probability: float = Field(0.0, ge=0.0, le=1.0)

    risk_scenarios: list[RiskScenario] = Field(default_factory=list)
    view_summary: str = ""

    def core(self) -> dict[str, float]:
        """(mu, m, c, u, tau, r) - the representation the gate compares."""
        return {name: float(getattr(self, name)) for name in CORE}


class NewsView(AgentView):
    agent: str = "news"
    direction: DIRECTIONS = "neutral"
    upside_tail_potential: float = Field(0.0, ge=0.0, le=1.0)
    volatility_impact: float = Field(0.0, ge=-1.0, le=1.0)
    event_types: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)


class TechnicalView(AgentView):
    agent: str = "technical"
    direction: DIRECTIONS = "neutral"
    trend_strength: float = Field(0.0, ge=0.0, le=1.0)
    momentum_score: float = Field(0.0, ge=-1.0, le=1.0)
    volume_confirmation: float = Field(0.0, ge=-1.0, le=1.0)
    upside_tail_potential: float = Field(0.0, ge=0.0, le=1.0)
    breakout_probability: float = Field(0.0, ge=0.0, le=1.0)
    breakdown_probability: float = Field(0.0, ge=0.0, le=1.0)
    signal_agreement: float = Field(0.0, ge=0.0, le=1.0)
    key_signals: list[str] = Field(default_factory=list)


class VolatilityView(AgentView):
    """Its directional field is ``directional_bias``; magnitude is volatility."""

    agent: str = "volatility"
    volatility_regime: REGIMES = "normal"
    volatility_score: float = Field(0.0, ge=0.0, le=1.0)
    volatility_expansion_probability: float = Field(0.0, ge=0.0, le=1.0)
    directional_bias: float = Field(0.0, ge=-1.0, le=1.0)
    tail_event_probability: float = Field(0.0, ge=0.0, le=1.0)
    tail_severity: float = Field(0.0, ge=0.0, le=1.0)
    negative_skew_risk: float = Field(0.0, ge=0.0, le=1.0)
    heavy_tail_score: float = Field(0.0, ge=0.0, le=1.0)
    jump_risk: float = Field(0.0, ge=0.0, le=1.0)
    liquidity_stress: float = Field(0.0, ge=0.0, le=1.0)
    systemic_risk_component: float = Field(0.0, ge=0.0, le=1.0)
    idiosyncratic_risk_component: float = Field(0.0, ge=0.0, le=1.0)
    key_risk_drivers: list[str] = Field(default_factory=list)

    def core(self) -> dict[str, float]:
        """This agent forecasts risk, so direction and magnitude map differently.

        ``directional_bias`` is its (deliberately secondary) view, and
        ``volatility_score`` is its magnitude - the expected size of the move,
        not the expected direction of it.
        """
        return {
            "expected_return_score": float(self.directional_bias),
            "expected_magnitude": float(self.volatility_score),
            "confidence": float(self.confidence),
            "epistemic_uncertainty": float(self.epistemic_uncertainty),
            "downside_tail_risk": float(self.downside_tail_risk),
            "regime_shift_probability": float(self.regime_shift_probability),
        }


VIEWS: dict[str, type[AgentView]] = {
    "news": NewsView,
    "technical": TechnicalView,
    "volatility": VolatilityView,
}
