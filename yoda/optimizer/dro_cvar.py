"""Distributionally robust CVaR allocation - the layer both A and C drive.

    max_w   mu'w  -  lam * DRO-CVaR_alpha(w)  -  c * ||w - w_prev||_1
    s.t.    w >= 0,  sum w = 1,  w <= max_weight,  [CVaR_alpha(w) <= B]

CVaR is the Rockafellar-Uryasev convex form over the scenario set, so the whole
problem is a single convex program solved with cvxpy.

Two robustifications are configurable:

``wasserstein`` (default)
    Worst-case CVaR over a Wasserstein-2 ball of radius ``radius`` around the
    empirical scenario distribution. For a linear loss this has the closed form
    ``CVaR_emp(w) + (radius / alpha) * ||w||_2`` - the robustness shows up as a
    concentration penalty, which is exactly the behaviour you want out of it.
``moment``
    Worst-case CVaR over all distributions matching the scenario mean and
    covariance: ``-mu_s'w + kappa * sqrt((1-alpha)/alpha) * ||Sigma^(1/2) w||_2``.

The solver is deliberately blind to where ``rp`` came from: a static rule and an
RL action reach it through the identical call.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from yoda.common.logger import logger
from yoda.common.types import DROCVaROptimizer as DROCVaROptimizerABC
from yoda.common.types import RiskParams, TailScenarios
from yoda.config.research import OptimizerConfig


def _risk_expression(
    w: cp.Variable, scen: TailScenarios, rp: RiskParams, config: OptimizerConfig
) -> tuple[cp.Expression, list]:
    """The (robust) CVaR expression plus any auxiliary constraints."""
    scenarios = np.asarray(scen.scenarios, dtype=np.float64)
    if config.dro == "moment":
        mean = scenarios.mean(axis=0)
        covariance = np.cov(scenarios, rowvar=False)
        values, vectors = np.linalg.eigh(np.atleast_2d(covariance))
        root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0))) @ vectors.T
        kappa = np.sqrt((1 - rp.alpha) / rp.alpha)
        return -mean @ w + config.radius * kappa * cp.norm(root @ w, 2), []

    rows = len(scenarios)
    zeta = cp.Variable()
    excess = cp.Variable(rows, nonneg=True)
    cvar = zeta + cp.sum(excess) / (rp.alpha * rows)
    constraints = [excess >= -(scenarios @ w) - zeta]
    if config.dro == "wasserstein":
        cvar = cvar + (config.radius / rp.alpha) * cp.norm(w, 2)
    return cvar, constraints


class DROCVaR(DROCVaROptimizerABC):
    """cvxpy implementation of the shared allocation layer."""

    def __init__(self, config: OptimizerConfig | None = None):
        self.config = config or OptimizerConfig()
        self.failures = 0

    def solve(
        self,
        mu: np.ndarray,
        scen: TailScenarios,
        rp: RiskParams,
        w_prev: np.ndarray,
    ) -> np.ndarray:
        mu = np.nan_to_num(np.asarray(mu, dtype=np.float64).reshape(-1))
        w_prev = np.asarray(w_prev, dtype=np.float64).reshape(-1)
        assets = mu.size
        if scen.scenarios.shape[1] != assets or w_prev.size != assets:
            raise ValueError(
                "mu, scenarios and w_prev must agree on the asset count: "
                f"{assets}, {scen.scenarios.shape[1]}, {w_prev.size}"
            )

        w = cp.Variable(assets, nonneg=True)
        cvar, constraints = _risk_expression(w, scen, rp, self.config)
        cap = max(self.config.max_weight, 1.5 / assets)
        constraints += [cp.sum(w) == 1, w <= cap]

        objective = mu @ w - rp.lam * cvar - rp.turnover_penalty * cp.norm1(w - w_prev)
        if self.config.budget_mode == "constraint":
            constraints.append(cvar <= rp.budget)
        else:
            objective = objective - self.config.budget_penalty * cp.pos(
                cvar - rp.budget
            )

        problem = cp.Problem(cp.Maximize(objective), constraints)
        try:
            problem.solve(solver=self.config.solver)
        except cp.error.SolverError:
            problem.status = "solver_error"
        if w.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
            return self._fallback(problem.status, assets, scen, rp, mu, w_prev)
        weights = np.clip(np.asarray(w.value, dtype=np.float64), 0.0, None)
        total = weights.sum()
        return weights / total if total > 0 else self._equal(assets)

    def _fallback(
        self,
        status: str,
        assets: int,
        scen: TailScenarios,
        rp: RiskParams,
        mu: np.ndarray,
        w_prev: np.ndarray,
    ) -> np.ndarray:
        """A hard budget can be infeasible; relax it once, then hold the book."""
        self.failures += 1
        logger.warning(
            "optimizer_fallback status=%s failures=%d", status, self.failures
        )
        if self.config.budget_mode == "constraint":
            relaxed = DROCVaR(
                OptimizerConfig(**{**self.config.__dict__, "budget_mode": "penalty"})
            )
            return relaxed.solve(mu, scen, rp, w_prev)
        total = w_prev.sum()
        return w_prev / total if total > 0 else self._equal(assets)

    @staticmethod
    def _equal(assets: int) -> np.ndarray:
        return np.full(assets, 1.0 / assets)
