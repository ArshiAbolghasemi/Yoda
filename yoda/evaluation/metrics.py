"""Performance and risk metrics, computed offline from run artifacts.

The nine HedgeAgents-style metrics - TR, ARR, Sharpe, Calmar, Sortino, MaxDD,
Volatility, Entropy and Effective Number of Bets - plus realized CVaR/ES at the
run's alpha and average turnover.

``ponytail:`` ENB uses the marginal-risk-contribution decomposition rather than
Meucci's minimum-torsion basis. It is the standard cheap form and agrees with
the torsion version on ranking; swap in minimum-torsion if a referee asks for
the exact construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_ORDER: tuple[str, ...] = (
    "TR",
    "ARR",
    "Sharpe",
    "Calmar",
    "Sortino",
    "MaxDD",
    "Volatility",
    "ENT",
    "ENB",
    "CVaR",
    "ES",
    "Turnover",
)


def drawdown(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(-drawdown(returns).min()) if len(returns) else 0.0


def entropy(weights: np.ndarray) -> float:
    """Shannon entropy of the average book; ``ln(N)`` for an even portfolio."""
    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 0]
    return float(-(w * np.log(w)).sum()) if w.size else 0.0


def effective_bets(weights: np.ndarray, covariance: np.ndarray) -> float:
    """``exp`` of the entropy of the marginal risk contributions."""
    w = np.asarray(weights, dtype=np.float64)
    variance = float(w @ covariance @ w)
    if variance <= 0:
        return float("nan")
    contributions = w * (covariance @ w) / variance
    contributions = contributions[contributions > 0]
    if not contributions.size:
        return float("nan")
    return float(np.exp(-(contributions * np.log(contributions)).sum()))


def tail(returns: pd.Series, alpha: float) -> tuple[float, float]:
    """(CVaR, ES): the mean loss and the mean return in the worst ``alpha`` tail."""
    if returns.empty:
        return float("nan"), float("nan")
    cut = max(int(np.ceil(alpha * len(returns))), 1)
    worst = np.sort(returns.to_numpy())[:cut]
    return float(-worst.mean()), float(worst.mean())


def performance(
    returns: pd.Series,
    *,
    alpha: float = 0.05,
    periods_per_year: int = 252,
    turnover: pd.Series | None = None,
    weights: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
) -> dict[str, float]:
    """The full metric row for one interval."""
    returns = returns.dropna()
    if returns.empty:
        return dict.fromkeys(METRIC_ORDER, float("nan"))

    total = float((1.0 + returns).prod() - 1.0)
    years = len(returns) / periods_per_year
    annual = (
        (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1 else np.nan
    )
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    downside = returns[returns < 0]
    downside_vol = (
        float(downside.std(ddof=1) * np.sqrt(periods_per_year))
        if len(downside) > 1
        else np.nan
    )
    mdd = max_drawdown(returns)
    cvar, es = tail(returns, alpha)

    return {
        "TR": total,
        "ARR": float(annual),
        "Sharpe": float(annual / volatility) if volatility > 0 else np.nan,
        "Calmar": float(annual / mdd) if mdd > 0 else np.nan,
        "Sortino": float(annual / downside_vol)
        if downside_vol and downside_vol > 0
        else np.nan,
        "MaxDD": mdd,
        "Volatility": volatility,
        "ENT": entropy(weights) if weights is not None else float("nan"),
        "ENB": effective_bets(weights, covariance)
        if weights is not None and covariance is not None
        else float("nan"),
        "CVaR": cvar,
        "ES": es,
        "Turnover": float(turnover.mean())
        if turnover is not None and len(turnover)
        else float("nan"),
    }


def rolling_sharpe(
    returns: pd.Series, window: int, periods_per_year: int = 252
) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (mean / std.replace(0, np.nan)) * np.sqrt(periods_per_year)
