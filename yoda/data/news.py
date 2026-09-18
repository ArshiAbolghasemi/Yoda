"""Headline-only news normalization and asset attribution."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from yoda.common.logger import logger
from yoda.data.settings import NewsConfig


def _headline(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _direct_assets(symbols: Iterable[str], assets: set[str]) -> set[str]:
    aliases = {asset.replace("-", "").replace("=X", ""): asset for asset in assets}
    return {
        aliases[symbol.replace("/", "").replace("-", "")]
        for symbol in symbols
        if symbol.replace("/", "").replace("-", "") in aliases
    }


def process_news(
    items: list[dict], assets: dict[str, str], config: NewsConfig
) -> pd.DataFrame:
    """Keep normalized headlines and assign only assets known at publication time."""
    rows: list[dict] = []
    for item in items:
        headline = _headline(item.get("headline"))
        if not headline or not item.get("created_at"):
            continue
        timestamp = pd.Timestamp(item["created_at"])
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        local = timestamp.tz_convert(config.market_timezone)
        availability = local.normalize().tz_localize(None)
        if local.hour >= config.market_close_hour:
            availability += pd.Timedelta(days=1)
        attributed = _direct_assets(item.get("symbols") or (), set(assets))
        lowered = headline.casefold()
        for asset, keywords in config.fx_keywords.items():
            if asset in assets and any(
                re.search(rf"(?<!\w){re.escape(word.casefold())}(?!\w)", lowered)
                for word in keywords
            ):
                attributed.add(asset)
        for asset in attributed:
            rows.append(
                {
                    "availability_date": availability,
                    "asset": asset,
                    "headline": headline,
                }
            )
    result = pd.DataFrame(
        rows, columns=["availability_date", "asset", "headline"]
    ).drop_duplicates()
    logger.info(
        "news_preprocessed rows=%d assets=%d",
        len(result),
        result["asset"].nunique() if len(result) else 0,
    )
    return result


def align_and_aggregate_news(
    news: pd.DataFrame, market: pd.DataFrame, separator: str
) -> pd.DataFrame:
    """Move weekend/after-close news to the first following trading day, then aggregate."""
    if news.empty:
        return pd.DataFrame(columns=["date", "asset", "headlines", "news_count"])
    aligned: list[pd.DataFrame] = []
    calendars = {
        asset: group["date"].drop_duplicates().sort_values()
        for asset, group in market.groupby("asset")
    }
    for asset, group in news.groupby("asset"):
        calendar = calendars.get(asset)
        if calendar is None or calendar.empty:
            continue
        positions = calendar.searchsorted(group["availability_date"], side="left")
        valid = positions < len(calendar)
        selected = group.loc[valid].copy()
        selected["date"] = calendar.iloc[positions[valid]].to_numpy()
        aligned.append(selected)
    if not aligned:
        return pd.DataFrame(columns=["date", "asset", "headlines", "news_count"])
    result = pd.concat(aligned, ignore_index=True)
    return result.groupby(["date", "asset"], as_index=False).agg(
        headlines=("headline", lambda values: separator.join(dict.fromkeys(values))),
        news_count=("headline", "nunique"),
    )
