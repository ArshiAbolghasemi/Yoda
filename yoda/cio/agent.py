"""The CIO agent: one decision-maker for the whole allocation.

Where the main architecture decomposes the decision into three measurable
pieces — Tail-VoI decides *whom to trust*, the risk policy decides *how much
risk to carry*, and DRO-CVaR *builds the book* — this agent does all three
itself:

* :class:`~yoda.common.types.Gate` — sets ``g`` over the information sources.
* :class:`~yoda.common.types.RiskParamPolicy` — sets ``(lam, B, c)``.
* :class:`~yoda.common.types.DROCVaROptimizer` — emits the portfolio weights.

It uses **no OpenJev and no model inference at all**. Everything it decides is
a deterministic function of what the specialists already reported, so it is
reproducible, free to run, and needs nothing served.

Why it exists
-------------
It is the monolithic-manager control. A single agent reading the analysts and
producing a portfolio is the common design in multi-agent trading systems, and
the question this repository asks is whether *decomposing* that decision buys
anything. Running both on identical splits is the only way to find out.

What it costs
-------------
Two properties the main stack guarantees are **not** available here, and both
matter:

1. **The CVaR budget is not enforced.** ``B`` becomes advisory. DRO-CVaR
   imposes ``CVaR_alpha(w) <= B`` as a hard constraint; this agent solves a
   mean-variance program instead, so a run can breach its own budget. The
   realised CVaR is still recorded in the ledger, so the breach is visible
   after the fact rather than prevented.
2. **Attribution is gone.** Tail-VoI exists to measure what each source is
   worth in the tail. When one agent consumes all three channels and emits
   weights, there is no longer a quantity being measured — which is precisely
   what makes this a control rather than a proposal.

The simplex, the per-asset cap and the turnover penalty *are* still enforced,
because those come from the convex program it solves.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from yoda.common.logger import logger
from yoda.common.types import (
    DROCVaROptimizer,
    Gate,
    GateOutput,
    MarketState,
    RiskParamPolicy,
    RiskParams,
    SpecialistOutput,
    TailScenarios,
)
from yoda.config.research import CIOConfig, OptimizerConfig, StaticPolicyConfig
from yoda.config.settings import Config
from yoda.tailvoi.base import fuse, softmax

EPSILON = 1e-12


def _dispersion(view: np.ndarray) -> float:
    """Cross-sectional spread of a source's view on one date.

    A channel that differentiates between assets today is saying something; a
    channel reporting nearly the same number for everything is not, whatever
    its level. Spread is comparable across a directional view and a risk view,
    which a raw mean is not.
    """
    values = np.asarray(view, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(finite.std()) if finite.size else 0.0


class CIOAgent(Gate, RiskParamPolicy, DROCVaROptimizer):
    """Reads the desk, sets the trust and the risk stance, and builds the book."""

    name = "cio"

    def __init__(
        self,
        config: Config,
        sources: tuple[str, ...],
        cio: CIOConfig | None = None,
        static: StaticPolicyConfig | None = None,
        optimizer: OptimizerConfig | None = None,
    ):
        research = config.research
        self.config = config
        self.cio = cio or research.cio
        self.static = static or research.static_policy
        self.optimizer = optimizer or research.optimizer
        self.alpha = self.optimizer.alpha
        self.sources = tuple(sources)
        # Per-source dispersion normaliser, fitted on the training window so the
        # channels are comparable before they are compared.
        self.centre: dict[str, float] = dict.fromkeys(self.sources, 0.0)
        self.scale: dict[str, float] = dict.fromkeys(self.sources, 1.0)
        self.fitted = False
        self._stance = 0.5

    # ---- fitting ---------------------------------------------------------

    def fit(self, stack, rows: np.ndarray) -> None:
        """Learn each channel's typical dispersion from the training window.

        Without this the softmax would compare a volatility view measured in
        daily standard deviations against a directional view measured in basis
        points, and the larger unit would win every day regardless of content.
        """
        sample = rows[:: max(len(rows) // self.cio.fit_states, 1)][
            : self.cio.fit_states
        ]
        observed: dict[str, list[float]] = {name: [] for name in self.sources}
        for position in sample:
            spec = stack.specialist_output(int(position))
            for name in self.sources:
                if name in spec.mu_hat:
                    observed[name].append(_dispersion(spec.mu_hat[name]))
        for name, values in observed.items():
            if values:
                self.centre[name] = float(np.mean(values))
                self.scale[name] = float(np.std(values)) or 1.0
        self.fitted = True
        logger.info(
            "cio_fitted states=%d sources=%s centre=%s",
            len(sample),
            ",".join(self.sources),
            {k: round(v, 6) for k, v in self.centre.items()},
        )

    # ---- Gate ------------------------------------------------------------

    def gate(self, spec: SpecialistOutput) -> GateOutput:
        """Trust each channel in proportion to how much it is saying today."""
        sources = tuple(name for name in self.sources if name in spec.mu_hat)
        if not sources:
            raise ValueError("CIO has no information sources to gate")
        scores = np.array(
            [
                (_dispersion(spec.mu_hat[name]) - self.centre.get(name, 0.0))
                / max(self.scale.get(name, 1.0), EPSILON)
                for name in sources
            ]
        )
        weights = softmax(np.nan_to_num(scores), self.cio.temperature)
        return fuse(spec, dict(zip(sources, weights, strict=True)))

    # ---- RiskParamPolicy --------------------------------------------------

    def act(self, state: MarketState) -> RiskParams:
        """A volatility-targeting stance: widen when calm, tighten when not."""
        current = float(state.tail_stats.get("cvar", 0.0))
        target = self.cio.vol_target
        if current > 0 and target > 0:
            ratio = float(np.clip(target / current, 0.2, 5.0))
        else:
            ratio = 1.0
        self._stance = ratio
        return RiskParams(
            # More headroom when the tail is quiet, less when it is not.
            lam=float(np.clip(self.static.lam / ratio, 0.01, 100.0)),
            budget=float(self.static.budget * ratio),
            turnover_penalty=self.static.turnover_penalty,
            alpha=self.alpha,
        )

    # ---- DROCVaROptimizer -------------------------------------------------

    def solve(
        self,
        mu: np.ndarray,
        scen: TailScenarios,
        rp: RiskParams,
        w_prev: np.ndarray,
    ) -> np.ndarray:
        """Long-only mean-variance on the scenario covariance.

        Deliberately **not** the DRO-CVaR program: this agent is the control
        for what the decomposed stack buys, so it must not borrow the stack's
        risk machinery. ``rp.budget`` is therefore advisory here - the simplex,
        the cap and the turnover penalty bind, the CVaR bound does not.
        """
        view = np.nan_to_num(np.asarray(mu, dtype=np.float64).reshape(-1))
        previous = np.asarray(w_prev, dtype=np.float64).reshape(-1)
        assets = view.size
        covariance = np.atleast_2d(
            np.cov(np.asarray(scen.scenarios, dtype=np.float64), rowvar=False)
        )
        w = cp.Variable(assets, nonneg=True)
        cap = max(self.optimizer.max_weight, 1.5 / assets)
        objective = (
            view @ w
            - (rp.lam / 2) * cp.quad_form(w, cp.psd_wrap(covariance))
            - rp.turnover_penalty * cp.norm1(w - previous)
        )
        problem = cp.Problem(cp.Maximize(objective), [cp.sum(w) == 1, w <= cap])
        try:
            problem.solve(solver=self.optimizer.solver)
        except cp.error.SolverError:
            problem.status = "solver_error"
        if w.value is None:
            logger.warning("cio_solve_fallback status=%s", problem.status)
            total = previous.sum()
            return previous / total if total > 0 else np.full(assets, 1.0 / assets)
        weights = np.clip(np.asarray(w.value, dtype=np.float64), 0.0, None)
        total = weights.sum()
        return weights / total if total > 0 else np.full(assets, 1.0 / assets)
