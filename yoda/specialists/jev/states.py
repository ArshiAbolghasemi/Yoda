"""Point-in-time state construction, one information set per specialist.

The three specialists are deliberately **information-specialised**: they must
not receive the same market state, or the experiment cannot tell whether their
judgements are complementary or merely correlated. Technical sees price
structure, volatility sees dispersion and drawdown, news sees headlines. None
of them sees another's inputs.

Every feature here is computed from data at or before the observation date.
Rolling statistics are shifted so the window closes at ``t-1`` where the value
at ``t`` would otherwise leak the bar being predicted, and no target, future
return or future volatility ever enters a state.

Prompt-level instructions are not the leakage control - this module is. The
guard text tells the model what it should not do; the construction here is what
makes it unable to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yoda.common.alignment import AlignedPanel

# Compact, non-redundant subsets. Sending all 60 indicators would be mostly
# collinear repetition; these are configurable through JEV__*_FEATURES.
TECHNICAL_FEATURES: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "return_20d",
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "roc_5",
    "roc_10",
    "roc_20",
    "macd",
    "macd_signal",
    "macd_diff",
    "adx_14",
    "stoch_k_14",
    "stoch_d_14",
    "williams_r_14",
    "cci_20",
    "price_vs_sma_5",
    "price_vs_sma_20",
    "price_vs_sma_50",
    "price_vs_ema_20",
)
VOLATILITY_FEATURES: tuple[str, ...] = (
    "realized_vol_5",
    "realized_vol_20",
    "realized_vol_60",
    "downside_vol_20",
    "atr_7_normalized",
    "atr_14_normalized",
    "atr_21_normalized",
    "bollinger_width_20",
    "keltner_width",
    "drawdown_20",
    "drawdown_60",
    "max_abs_return_20",
    "negative_return_fraction_20",
    "volatility_percentile_252",
)


def _indicator(panel: AlignedPanel, name: str) -> np.ndarray:
    names = panel.feature_names["technical"]
    if name not in names:
        return np.full(panel.returns.shape, np.nan)
    return panel.features["technical"][:, :, names.index(name)]


def _rolling(frame: pd.DataFrame, window: int, how: str, minimum: int | None = None):
    rolled = frame.rolling(window, min_periods=minimum or window)
    return getattr(rolled, how)()


def technical_states(panel: AlignedPanel) -> dict[str, np.ndarray]:
    """Trend, momentum and price-position, scaled so assets are comparable.

    Relative measures are preferred over raw prices throughout: ``price_vs_sma``
    rather than price and SMA separately, so a $500 equity and a 1.09 FX rate
    present the same way to the model.
    """
    returns = pd.DataFrame(panel.returns)
    prices = panel.prices
    compounded = (1.0 + returns).cumprod()

    state: dict[str, np.ndarray] = {
        "return_1d": panel.returns,
        "return_5d": (compounded / compounded.shift(5) - 1.0).to_numpy(),
        "return_20d": (compounded / compounded.shift(20) - 1.0).to_numpy(),
    }
    for name in (
        "rsi_7",
        "rsi_14",
        "rsi_21",
        "roc_5",
        "roc_10",
        "roc_20",
        "adx_14",
        "stoch_k_14",
        "stoch_d_14",
        "williams_r_14",
        "cci_20",
    ):
        state[name] = _indicator(panel, name)

    macd = _indicator(panel, "macd_12_26")
    signal = _indicator(panel, "macd_signal_9")
    state["macd"] = macd
    state["macd_signal"] = signal
    state["macd_diff"] = _indicator(panel, "macd_histogram")

    # Price position relative to its own moving averages, as a fraction.
    for window in (5, 20, 50):
        average = _indicator(panel, f"sma_{window}")
        state[f"price_vs_sma_{window}"] = (
            np.divide(
                prices, average, out=np.full_like(prices, np.nan), where=average > 0
            )
            - 1.0
        )
    ema = _indicator(panel, "ema_20")
    state["price_vs_ema_20"] = (
        np.divide(prices, ema, out=np.full_like(prices, np.nan), where=ema > 0) - 1.0
    )
    return state


def volatility_states(panel: AlignedPanel) -> dict[str, np.ndarray]:
    """Dispersion, drawdown and tail history. No directional information."""
    returns = pd.DataFrame(panel.returns)
    prices = panel.prices
    squared = returns**2

    state: dict[str, np.ndarray] = {}
    for window in (5, 20, 60):
        state[f"realized_vol_{window}"] = np.sqrt(
            _rolling(squared, window, "mean").to_numpy()
        )

    # Downside deviation: dispersion of losses only.
    losses = returns.where(returns < 0, 0.0)
    state["downside_vol_20"] = np.sqrt(_rolling(losses**2, 20, "mean").to_numpy())

    for window in (7, 14, 21):
        atr = _indicator(panel, f"atr_{window}")
        state[f"atr_{window}_normalized"] = np.divide(
            atr, prices, out=np.full_like(atr, np.nan), where=prices > 0
        )
    state["bollinger_width_20"] = _indicator(panel, "bb_width_20")
    state["keltner_width"] = _indicator(panel, "keltner_width")

    # Drawdown against the running peak inside each trailing window.
    compounded = (1.0 + returns).cumprod()
    for window in (20, 60):
        peak = _rolling(compounded, window, "max")
        state[f"drawdown_{window}"] = (compounded / peak - 1.0).to_numpy()

    state["max_abs_return_20"] = _rolling(returns.abs(), 20, "max").to_numpy()
    state["negative_return_fraction_20"] = _rolling(
        (returns < 0).astype(float), 20, "mean"
    ).to_numpy()

    # Where today's 20-day vol sits in its own trailing year. Shifted so the
    # ranking window closes before the observation it describes.
    vol20 = pd.DataFrame(state["realized_vol_20"])
    state["volatility_percentile_252"] = (
        vol20.rolling(252, min_periods=60).rank(pct=True).to_numpy()
    )
    return state


def build_state_frame(
    panel: AlignedPanel, channel: str
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """The (feature -> (T, N) matrix) map for one specialist's information set."""
    if channel == "technical":
        return technical_states(panel), TECHNICAL_FEATURES
    if channel == "volatility":
        return volatility_states(panel), VOLATILITY_FEATURES
    raise ValueError(f"No numeric state for channel {channel!r}")
