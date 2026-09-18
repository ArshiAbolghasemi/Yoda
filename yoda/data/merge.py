"""Feature, news, and target assembly."""

import pandas as pd

from yoda.common.logger import logger


def add_targets(frame: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Compute close-to-future-close returns within each asset."""
    result = frame.copy()
    grouped = result.groupby("asset", sort=False)["adjusted_close"]
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"Target horizon must be positive: {horizon}")
        result[f"future_return_{horizon}d"] = (
            grouped.shift(-horizon).div(result["adjusted_close"]) - 1
        )
    return result


def build_dataset(
    market_features: pd.DataFrame, daily_news: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    result = market_features.merge(
        daily_news, on=["date", "asset"], how="left", validate="one_to_one"
    )
    result["headlines"] = result["headlines"].fillna("")
    result["news_count"] = result["news_count"].fillna(0).astype("int32")
    result = (
        add_targets(result, horizons)
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )
    logger.info(
        "dataset_built rows=%d assets=%d columns=%d",
        len(result),
        result["asset"].nunique(),
        len(result.columns),
    )
    return result
