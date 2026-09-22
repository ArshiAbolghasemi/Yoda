"""Classical allocators, behind the same interface, for the system baselines.

They take the identical ``solve`` call, so the backtest harness cannot tell them
apart from the DRO-CVaR solver - which is the point: every baseline in the paper
runs through one code path.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from yoda.common.types import DROCVaROptimizer, RiskParams, TailScenarios


class EqualWeight(DROCVaROptimizer):
    """1/N over the investable universe."""

    def solve(self, mu, scen, rp, w_prev):
        assets = np.asarray(mu).reshape(-1).size
        return np.full(assets, 1.0 / assets)


class MeanVariance(DROCVaROptimizer):
    """Long-only Markowitz on the scenario covariance; ``lam`` is risk aversion."""

    def __init__(self, max_weight: float = 0.25, solver: str = "CLARABEL"):
        self.max_weight, self.solver = max_weight, solver

    def solve(
        self,
        mu: np.ndarray,
        scen: TailScenarios,
        rp: RiskParams,
        w_prev: np.ndarray,
    ) -> np.ndarray:
        mu = np.nan_to_num(np.asarray(mu, dtype=np.float64).reshape(-1))
        covariance = np.cov(scen.scenarios, rowvar=False)
        assets = mu.size
        w = cp.Variable(assets, nonneg=True)
        cap = max(self.max_weight, 1.5 / assets)
        problem = cp.Problem(
            cp.Maximize(mu @ w - rp.lam * cp.quad_form(w, cp.psd_wrap(covariance))),
            [cp.sum(w) == 1, w <= cap],
        )
        problem.solve(solver=self.solver)
        if w.value is None:
            return np.full(assets, 1.0 / assets)
        weights = np.clip(np.asarray(w.value), 0.0, None)
        return weights / weights.sum()


class RiskParity(DROCVaROptimizer):
    """Long-only equal-risk-contribution weights via the convex log-barrier form."""

    def __init__(self, solver: str = "CLARABEL"):
        self.solver = solver

    def solve(
        self,
        mu: np.ndarray,
        scen: TailScenarios,
        rp: RiskParams,
        w_prev: np.ndarray,
    ) -> np.ndarray:
        covariance = np.cov(scen.scenarios, rowvar=False)
        assets = covariance.shape[0]
        y = cp.Variable(assets, nonneg=True)
        problem = cp.Problem(
            cp.Minimize(
                0.5 * cp.quad_form(y, cp.psd_wrap(covariance)) - cp.sum(cp.log(y))
            )
        )
        problem.solve(solver=self.solver)
        if y.value is None:
            return np.full(assets, 1.0 / assets)
        weights = np.clip(np.asarray(y.value), 1e-12, None)
        return weights / weights.sum()
