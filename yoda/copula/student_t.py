"""Conditional Student-t tail model over the aligned return panel.

Marginals and dependence are separated: returns are devolatilised with an EWMA
scale, the standardised residuals keep their empirical (fat, skewed) marginal
shape, and a Student-t copula carries the joint tail dependence that a Gaussian
copula would throw away.

*Dimensionality.* With 36 assets and ~1250 training rows the sample is comfortably
longer than it is wide, so no factor reduction or asset grouping is applied. The
correlation matrix is estimated from Kendall's tau - robust to the fat marginals
- and ridge-shrunk toward the identity, which is what keeps it well conditioned
and positive definite. If the universe grows past a few hundred names this is the
first thing that has to change.

*Conditioning.* ``MarketState.regime`` carries the volatility specialist's
per-asset ``sigma_hat``. ``conditional`` swaps that in for the unconditional EWMA
scale; passing a state with ``regime=None`` deliberately returns the
unconditional model, which is how the counterfactual generator ablates the
volatility channel at the stage it actually enters.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from yoda.common.logger import logger
from yoda.common.types import MarketState, TailModel, TailScenarios
from yoda.config.research import CopulaConfig


def _kendall_correlation(residuals: np.ndarray, shrinkage: float) -> np.ndarray:
    """Kendall's tau -> correlation, ridge-shrunk toward the identity."""
    tau = pd.DataFrame(residuals).corr(method="kendall").to_numpy(dtype=np.float64)
    correlation = np.sin(np.pi * tau / 2.0)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    correlation = (correlation + correlation.T) / 2.0
    n = correlation.shape[0]
    correlation = (1 - shrinkage) * correlation + shrinkage * np.eye(n)
    # Project onto the PSD cone; Kendall-implied matrices are not PSD in general.
    values, vectors = np.linalg.eigh(correlation)
    correlation = vectors @ np.diag(np.maximum(values, 1e-8)) @ vectors.T
    scale = np.sqrt(np.diag(correlation))
    return correlation / np.outer(scale, scale)


def _copula_loglik(u: np.ndarray, correlation: np.ndarray, nu: float) -> float:
    """Log-likelihood of a Student-t copula at pseudo-observations ``u``."""
    n = correlation.shape[0]
    x = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=nu)
    sign, logdet = np.linalg.slogdet(correlation)
    if sign <= 0:
        return -np.inf
    inverse = np.linalg.inv(correlation)
    quadratic = np.einsum("ij,jk,ik->i", x, inverse, x)
    rows = len(x)
    constant = (
        gammaln((nu + n) / 2)
        + (n - 1) * gammaln(nu / 2)
        - n * gammaln((nu + 1) / 2)
        - 0.5 * logdet
    )
    joint = -(nu + n) / 2 * np.log1p(quadratic / nu)
    marginal = ((nu + 1) / 2 * np.log1p(x**2 / nu)).sum(axis=1)
    return float(rows * constant + (joint + marginal).sum())


class StudentTCopula(TailModel):
    """Empirical marginals + Student-t dependence, with a conditional vol scale."""

    def __init__(self, config: CopulaConfig | None = None):
        self.config = config or CopulaConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.residuals: np.ndarray | None = None  # (T, N) standardised
        self.scale: np.ndarray | None = None  # (N,) unconditional vol
        self.R: np.ndarray | None = None
        self.nu: float = float(self.config.nu_grid[-1])

    # ---- fitting ---------------------------------------------------------

    def fit(self, returns: np.ndarray) -> None:
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim != 2 or len(returns) < 50:
            raise ValueError(
                f"Copula needs a (T, N) matrix with T >= 50: {returns.shape}"
            )
        if self.config.marginals != "empirical":
            raise NotImplementedError(
                "Only empirical marginals are implemented; GARCH-t marginals would "
                "slot in here behind the same interface."
            )
        weights = self._ewma_weights(len(returns))
        mean = weights @ returns
        scale = np.sqrt(np.maximum(weights @ (returns - mean) ** 2, 1e-12))
        residuals = (returns - mean) / scale

        recent = residuals[-self.config.corr_window :]
        blend = self.config.corr_blend
        correlation = (1 - blend) * _kendall_correlation(
            residuals, self.config.shrinkage
        ) + blend * _kendall_correlation(recent, self.config.shrinkage)

        u = (stats.rankdata(residuals, axis=0)) / (len(residuals) + 1)
        likelihoods = {
            nu: _copula_loglik(u, correlation, nu) for nu in self.config.nu_grid
        }
        self.nu = float(max(likelihoods, key=likelihoods.get))
        self.residuals, self.scale, self.R = residuals, scale, correlation
        logger.info(
            "copula_fitted rows=%d assets=%d nu=%.1f mean_corr=%.3f",
            len(returns),
            returns.shape[1],
            self.nu,
            float(correlation[np.triu_indices_from(correlation, 1)].mean()),
        )

    def _ewma_weights(self, rows: int) -> np.ndarray:
        decay = 0.5 ** (1 / self.config.vol_halflife)
        weights = decay ** np.arange(rows - 1, -1, -1)
        return weights / weights.sum()

    # ---- conditioning and sampling ---------------------------------------

    def _check(self) -> None:
        if self.residuals is None or self.R is None or self.scale is None:
            raise RuntimeError("Copula is not fitted")

    def conditional(self, state: MarketState) -> StudentTCopula:
        """Re-scale the marginals by the state's volatility view."""
        self._check()
        conditioned = StudentTCopula(self.config)
        conditioned.residuals, conditioned.R, conditioned.nu = (
            self.residuals,
            self.R,
            self.nu,
        )
        conditioned.rng = self.rng
        if state.regime is None:
            conditioned.scale = self.scale
            return conditioned
        view = np.asarray(state.regime, dtype=np.float64).reshape(-1)
        if view.shape != self.scale.shape:
            raise ValueError(
                f"Regime vector must be per-asset: {view.shape} != {self.scale.shape}"
            )
        conditioned.scale = np.clip(view, 1e-6, 10 * self.scale)
        return conditioned

    def sample(self, n: int) -> TailScenarios:
        """Draw ``n`` joint return scenarios through the t-copula."""
        self._check()
        assets = self.R.shape[0]
        normal = self.rng.multivariate_normal(np.zeros(assets), self.R, size=n)
        chi = self.rng.chisquare(self.nu, size=n)
        t = normal * np.sqrt(self.nu / chi)[:, None]
        u = np.clip(stats.t.cdf(t, df=self.nu), 1e-6, 1 - 1e-6)
        # Inverse empirical marginal, per asset: the fat tails come from the data.
        shocks = np.stack(
            [
                np.quantile(self.residuals[:, asset], u[:, asset])
                for asset in range(assets)
            ],
            axis=1,
        )
        return TailScenarios(scenarios=shocks * self.scale, R=self.R, nu=self.nu)

    # ---- risk measurement ------------------------------------------------

    def tail_stats(self, w: np.ndarray, scen: TailScenarios, alpha: float) -> dict:
        return tail_stats(w, scen, alpha)


def tail_stats(w: np.ndarray, scen: TailScenarios, alpha: float) -> dict:
    """VaR/CVaR of the loss distribution, plus the shape of the scenario set.

    ``var`` and ``cvar`` are stated in loss units (positive is a loss); ``es`` is
    the mean *return* in the worst ``alpha`` tail (negative), which is what the
    reward and the reports quote.
    """
    portfolio = np.asarray(scen.scenarios, dtype=np.float64) @ np.asarray(w).reshape(-1)
    losses = -portfolio
    var = float(np.quantile(losses, 1 - alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return {
        "cvar": cvar,
        "var": var,
        "es": -cvar,
        "nu": float(scen.nu),
        "vol": float(portfolio.std(ddof=0)),
        "mean": float(portfolio.mean()),
    }


def stress_scenarios(
    model: StudentTCopula, n: int, severity: float = 2.0
) -> TailScenarios:
    """Synthetic stress paths: the fitted copula with an inflated vol scale.

    The real sample contains essentially one major crash, so tail evaluation
    leans on these as well as on the COVID sub-period.
    """
    stressed = replace(model.config)
    inflated = StudentTCopula(stressed)
    inflated.residuals, inflated.R, inflated.nu = model.residuals, model.R, model.nu
    inflated.scale = model.scale * severity
    inflated.rng = model.rng
    return inflated.sample(n)
