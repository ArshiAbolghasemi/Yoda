"""Final dataset integrity checks."""

import numpy as np
import pandas as pd

from yoda.data.features import INDICATOR_COLUMNS, VOLUME_INDICATORS


def validate_dataset(frame: pd.DataFrame, horizons: tuple[int, ...]) -> None:
    required = {
        "date",
        "asset",
        "asset_type",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "headlines",
        "news_count",
        *INDICATOR_COLUMNS,
        *(f"future_return_{horizon}d" for horizon in horizons),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if len(INDICATOR_COLUMNS) != 60 or len(set(INDICATOR_COLUMNS)) != 60:
        raise ValueError("Indicator schema must contain exactly 60 unique columns")
    if frame.duplicated(["date", "asset"]).any():
        raise ValueError("Dataset contains duplicate asset-date rows")
    if frame[["open", "high", "low", "close", "adjusted_close"]].isna().any().any():
        raise ValueError("Dataset contains missing OHLC or adjusted close values")
    fx = frame["asset_type"].eq("fx")
    if frame.loc[fx, VOLUME_INDICATORS].notna().any().any():
        raise ValueError(
            "FX volume indicators must be null when Yahoo volume is unavailable"
        )
    for horizon in horizons:
        expected = (
            frame.groupby("asset", sort=False)["adjusted_close"]
            .shift(-horizon)
            .div(frame["adjusted_close"])
            - 1
        )
        if not np.allclose(
            frame[f"future_return_{horizon}d"], expected, equal_nan=True
        ):
            raise ValueError(f"Invalid future-return target for horizon {horizon}")
