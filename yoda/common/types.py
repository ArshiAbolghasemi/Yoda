"""Shared contracts every research module composes through.

These are the seams: implementations live in their own packages and only ever
depend on the abstractions declared here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

SOURCES: tuple[str, ...] = ("technical", "volatility", "news")


# ---- value objects -------------------------------------------------------


@dataclass(frozen=True)
class RiskParams:
    """The action space of the risk-parameter policy. Drives the optimizer."""

    lam: float  # risk aversion: weight on the DRO-CVaR term
    budget: float  # B: CVaR upper bound (risk budget)
    turnover_penalty: float  # c: L1 turnover cost weight
    alpha: float = 0.05  # FIXED CVaR tail level - NOT an action

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1): {self.alpha}")
        if self.lam < 0 or self.turnover_penalty < 0:
            raise ValueError("lam and turnover_penalty must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return {
            "lam": self.lam,
            "budget": self.budget,
            "turnover_penalty": self.turnover_penalty,
            "alpha": self.alpha,
        }


@dataclass
class SpecialistOutput:
    z: dict[str, np.ndarray]  # {"technical": ..., "volatility": ..., "news": ...}
    mu_hat: dict[str, np.ndarray]  # per-source expected-return view (per asset)


@dataclass
class TailScenarios:
    scenarios: np.ndarray  # (n_scenarios, n_assets) simulated returns
    R: np.ndarray  # conditional correlation matrix
    nu: float  # Student-t degrees of freedom


@dataclass
class GateOutput:
    g: dict[str, float]  # gating weight per information source
    fused_mu: np.ndarray  # gate-weighted expected returns (per asset)
    fused_z: np.ndarray  # gate-weighted representation (for RL state)
    delta_hat: dict[str, float] = field(default_factory=dict)  # predicted Tail-VoI


@dataclass
class MarketState:
    """Input to RiskParamPolicy; also the RL observation."""

    fused_mu: np.ndarray
    fused_z: np.ndarray
    g: dict[str, float]
    tail_stats: dict  # {"cvar": ..., "var": ..., "nu": ..., "es": ...}
    w_prev: np.ndarray
    regime: np.ndarray | None = None

    def observation(self, sources: tuple[str, ...] = SOURCES) -> np.ndarray:
        """Flatten to the RL observation vector (stable ordering, finite values)."""
        parts = [
            np.asarray(self.fused_mu, dtype=np.float64).ravel(),
            np.asarray(self.fused_z, dtype=np.float64).ravel(),
            np.array([self.g.get(name, 0.0) for name in sources], dtype=np.float64),
            np.array(
                [float(self.tail_stats.get(key, 0.0)) for key in TAIL_STAT_KEYS],
                dtype=np.float64,
            ),
            np.asarray(self.w_prev, dtype=np.float64).ravel(),
        ]
        if self.regime is not None:
            parts.append(np.asarray(self.regime, dtype=np.float64).ravel())
        return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)


TAIL_STAT_KEYS: tuple[str, ...] = ("cvar", "var", "es", "nu", "vol")


# ---- abstract interfaces -------------------------------------------------


class Specialist(ABC):
    """Supervised encoder + view producer for one information source."""

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:  # -> z_i
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # -> mu_hat_i / sigma_hat_i
        ...


class TailModel(ABC):
    @abstractmethod
    def fit(self, returns: np.ndarray) -> None: ...

    @abstractmethod
    def conditional(self, state: MarketState) -> TailModel: ...

    @abstractmethod
    def sample(self, n: int) -> TailScenarios: ...

    @abstractmethod
    def tail_stats(self, w: np.ndarray, scen: TailScenarios, alpha: float) -> dict: ...


class Gate(ABC):
    name: str

    @abstractmethod
    def gate(self, spec: SpecialistOutput) -> GateOutput: ...


class DROCVaROptimizer(ABC):
    @abstractmethod
    def solve(
        self,
        mu: np.ndarray,
        scen: TailScenarios,
        rp: RiskParams,
        w_prev: np.ndarray,
    ) -> np.ndarray:
        """Return long-only weights on the simplex (w>=0, sum w = 1)."""


class RiskParamPolicy(ABC):
    """THE SEAM.

    ``StaticRiskPolicy`` => ``tail_voli_risk``.
    ``RLRiskPolicy``     => ``tail_voli_risk_rl``.
    """

    name: str

    @abstractmethod
    def act(self, state: MarketState) -> RiskParams: ...
