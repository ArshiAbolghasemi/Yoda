"""The canonical set of 60 causal technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

INDICATOR_COLUMNS = [
    *[f"sma_{n}" for n in (5, 10, 20, 50, 100, 200)],
    *[f"ema_{n}" for n in (5, 10, 20, 50, 100, 200)],
    *[f"wma_{n}" for n in (10, 20, 50)],
    *[f"roc_{n}" for n in (5, 10, 20)],
    *[f"rsi_{n}" for n in (7, 14, 21)],
    "stoch_k_14",
    "stoch_d_14",
    "williams_r_14",
    "williams_r_28",
    "cci_14",
    "cci_20",
    "atr_7",
    "atr_14",
    "atr_21",
    "adx_14",
    "adx_pos_14",
    "adx_neg_14",
    "bb_middle_20",
    "bb_upper_20",
    "bb_lower_20",
    "bb_width_20",
    "bb_percent_20",
    "macd_12_26",
    "macd_signal_9",
    "macd_histogram",
    "obv",
    "cmf_20",
    "mfi_14",
    "force_index_13",
    "ease_of_movement_14",
    "vwap_14",
    "volume_price_trend",
    "negative_volume_index",
    "psar_up",
    "psar_down",
    "psar_up_indicator",
    "psar_down_indicator",
    "keltner_middle",
    "keltner_upper",
    "keltner_lower",
    "keltner_width",
    "keltner_percent",
    "donchian_upper_20",
    "donchian_lower_20",
]
VOLUME_INDICATORS = [
    "obv",
    "cmf_20",
    "mfi_14",
    "force_index_13",
    "ease_of_movement_14",
    "vwap_14",
    "volume_price_trend",
    "negative_volume_index",
]
assert len(INDICATOR_COLUMNS) == 60


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = (
        delta.clip(lower=0)
        .ewm(alpha=1 / window, adjust=False, min_periods=window)
        .mean()
    )
    loss = (
        (-delta.clip(upper=0))
        .ewm(alpha=1 / window, adjust=False, min_periods=window)
        .mean()
    )
    return 100 - 100 / (1 + _divide(gain, loss))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)


def _parabolic_sar(
    high: pd.Series, low: pd.Series, step: float = 0.02, maximum: float = 0.2
) -> tuple[pd.Series, pd.Series]:
    """Return causal up/down PSAR series using the standard Wilder recurrence."""
    size = len(high)
    up, down = np.full(size, np.nan), np.full(size, np.nan)
    if size < 2:
        return pd.Series(up, index=high.index), pd.Series(down, index=high.index)
    rising = bool(high.iloc[1] + low.iloc[1] >= high.iloc[0] + low.iloc[0])
    sar = low.iloc[0] if rising else high.iloc[0]
    extreme = high.iloc[0] if rising else low.iloc[0]
    acceleration = step
    for index in range(1, size):
        candidate = sar + acceleration * (extreme - sar)
        if rising:
            candidate = min(candidate, low.iloc[index - 1], low.iloc[max(0, index - 2)])
            if low.iloc[index] < candidate:
                rising, candidate, extreme, acceleration = (
                    False,
                    extreme,
                    low.iloc[index],
                    step,
                )
            elif high.iloc[index] > extreme:
                extreme, acceleration = (
                    high.iloc[index],
                    min(acceleration + step, maximum),
                )
        else:
            candidate = max(
                candidate, high.iloc[index - 1], high.iloc[max(0, index - 2)]
            )
            if high.iloc[index] > candidate:
                rising, candidate, extreme, acceleration = (
                    True,
                    extreme,
                    high.iloc[index],
                    step,
                )
            elif low.iloc[index] < extreme:
                extreme, acceleration = (
                    low.iloc[index],
                    min(acceleration + step, maximum),
                )
        sar = candidate
        (up if rising else down)[index] = sar
    return pd.Series(up, index=high.index), pd.Series(down, index=high.index)


def _indicators(group: pd.DataFrame) -> pd.DataFrame:
    result = group.copy()
    high, low, close, volume = (
        result[name].astype(float) for name in ("high", "low", "close", "volume")
    )
    for window in (5, 10, 20, 50, 100, 200):
        result[f"sma_{window}"] = close.rolling(window).mean()
        result[f"ema_{window}"] = close.ewm(
            span=window, adjust=False, min_periods=window
        ).mean()
    for window in (10, 20, 50):
        weights = np.arange(1, window + 1)
        result[f"wma_{window}"] = close.rolling(window).apply(
            lambda values, weights=weights: np.dot(values, weights) / weights.sum(),
            raw=True,
        )
    for window in (5, 10, 20):
        result[f"roc_{window}"] = close.pct_change(window, fill_method=None) * 100
    for window in (7, 14, 21):
        result[f"rsi_{window}"] = _rsi(close, window)

    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    result["stoch_k_14"] = _divide(close - low14, high14 - low14) * 100
    result["stoch_d_14"] = result["stoch_k_14"].rolling(3).mean()
    for window in (14, 28):
        rolling_low, rolling_high = (
            low.rolling(window).min(),
            high.rolling(window).max(),
        )
        result[f"williams_r_{window}"] = (
            _divide(rolling_high - close, rolling_high - rolling_low) * -100
        )
    typical = (high + low + close) / 3
    for window in (14, 20):
        mean = typical.rolling(window).mean()
        deviation = typical.rolling(window).apply(
            lambda values: np.abs(values - values.mean()).mean(), raw=True
        )
        result[f"cci_{window}"] = _divide(typical - mean, 0.015 * deviation)

    true_range = _true_range(high, low, close)
    for window in (7, 14, 21):
        result[f"atr_{window}"] = true_range.ewm(
            alpha=1 / window, adjust=False, min_periods=window
        ).mean()
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr14 = result["atr_14"]
    result["adx_pos_14"] = (
        _divide(plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(), atr14)
        * 100
    )
    result["adx_neg_14"] = (
        _divide(minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(), atr14)
        * 100
    )
    dx = (
        _divide(
            (result["adx_pos_14"] - result["adx_neg_14"]).abs(),
            result["adx_pos_14"] + result["adx_neg_14"],
        )
        * 100
    )
    result["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    middle, std = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    result["bb_middle_20"], result["bb_upper_20"], result["bb_lower_20"] = (
        middle,
        middle + 2 * std,
        middle - 2 * std,
    )
    result["bb_width_20"] = _divide(
        result["bb_upper_20"] - result["bb_lower_20"], middle
    )
    result["bb_percent_20"] = _divide(
        close - result["bb_lower_20"], result["bb_upper_20"] - result["bb_lower_20"]
    )
    macd = (
        close.ewm(span=12, adjust=False, min_periods=26).mean()
        - close.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    result["macd_12_26"] = macd
    result["macd_signal_9"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_histogram"] = macd - result["macd_signal_9"]

    result["obv"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    multiplier = _divide((close - low) - (high - close), high - low)
    result["cmf_20"] = _divide(
        (multiplier * volume).rolling(20).sum(), volume.rolling(20).sum()
    )
    raw_money = typical * volume
    positive = raw_money.where(typical.diff() > 0, 0).rolling(14).sum()
    negative = raw_money.where(typical.diff() < 0, 0).rolling(14).sum()
    result["mfi_14"] = 100 - 100 / (1 + _divide(positive, negative))
    result["force_index_13"] = (
        (close.diff() * volume).ewm(span=13, adjust=False, min_periods=13).mean()
    )
    distance = ((high + low) / 2).diff()
    result["ease_of_movement_14"] = (
        _divide(distance * (high - low), volume).rolling(14).mean()
    )
    result["vwap_14"] = _divide(
        (typical * volume).rolling(14).sum(), volume.rolling(14).sum()
    )
    result["volume_price_trend"] = (
        (close.pct_change(fill_method=None) * volume).fillna(0).cumsum()
    )
    nvi = np.full(len(result), np.nan)
    if len(result):
        nvi[0] = 1000
        returns, volumes = (
            close.pct_change(fill_method=None).to_numpy(),
            volume.to_numpy(),
        )
        for index in range(1, len(result)):
            nvi[index] = (
                nvi[index - 1] * (1 + returns[index])
                if volumes[index] < volumes[index - 1]
                else nvi[index - 1]
            )
    result["negative_volume_index"] = nvi

    result["psar_up"], result["psar_down"] = _parabolic_sar(high, low)
    result["psar_up_indicator"] = result["psar_up"].notna().astype("int8")
    result["psar_down_indicator"] = result["psar_down"].notna().astype("int8")
    keltner_middle = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["keltner_middle"] = keltner_middle
    result["keltner_upper"], result["keltner_lower"] = (
        keltner_middle + 2 * atr14,
        keltner_middle - 2 * atr14,
    )
    result["keltner_width"] = _divide(
        result["keltner_upper"] - result["keltner_lower"], keltner_middle
    )
    result["keltner_percent"] = _divide(
        close - result["keltner_lower"],
        result["keltner_upper"] - result["keltner_lower"],
    )
    result["donchian_upper_20"], result["donchian_lower_20"] = (
        high.rolling(20).max(),
        low.rolling(20).min(),
    )
    if result["asset_type"].iloc[0] == "fx" or volume.notna().sum() == 0:
        result[VOLUME_INDICATORS] = np.nan
    return result


def add_indicators(frame: pd.DataFrame, *, fill_infinite: bool = True) -> pd.DataFrame:
    """Add exactly ``INDICATOR_COLUMNS`` using current and previous rows only."""
    result = pd.concat(
        (_indicators(group) for _, group in frame.groupby("asset", sort=False)),
        ignore_index=True,
    )
    if fill_infinite:
        result[INDICATOR_COLUMNS] = result[INDICATOR_COLUMNS].replace(
            [np.inf, -np.inf], np.nan
        )
    return result
