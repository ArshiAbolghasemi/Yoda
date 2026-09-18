"""Market-data normalization."""

import numpy as np
import pandas as pd

from yoda.common.logger import logger

MARKET_COLUMNS = [
    "date",
    "asset",
    "asset_type",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]


def preprocess_market(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo columns and make FX volume absence explicit."""
    renamed = frame.rename(
        columns=lambda value: str(value).strip().lower().replace(" ", "_")
    )
    if "adj_close" in renamed and "adjusted_close" not in renamed:
        renamed = renamed.rename(columns={"adj_close": "adjusted_close"})
    missing = set(MARKET_COLUMNS) - set(renamed.columns)
    if missing:
        raise ValueError(f"Missing market columns: {sorted(missing)}")
    result = renamed[MARKET_COLUMNS].copy()
    result["date"] = (
        pd.to_datetime(result["date"], utc=True).dt.tz_localize(None).dt.normalize()
    )
    numeric = ["open", "high", "low", "close", "adjusted_close", "volume"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result.loc[(result["asset_type"] == "fx") & (result["volume"] == 0), "volume"] = (
        np.nan
    )
    result = result.drop_duplicates(["date", "asset"], keep="last").sort_values(
        ["asset", "date"]
    )
    logger.info(
        "market_preprocessed rows=%d assets=%d", len(result), result["asset"].nunique()
    )
    return result.reset_index(drop=True)
