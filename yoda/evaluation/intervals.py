"""Slice a finished run by split, by named market regime, or by rolling window.

This is what makes the cross-regime protocol cheap: the backtest ran once and
left ``weights.parquet`` / ``ledger.parquet`` behind, so scoring COVID, the 2022
bear market and the test split is three slices of the same file rather than
three backtests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from yoda.config.research import EvalConfig, SplitConfig


@dataclass(frozen=True)
class Interval:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str  # split | regime | full

    def mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.start) & (dates <= self.end)


def split_intervals(split: SplitConfig) -> list[Interval]:
    return [
        Interval(name, pd.Timestamp(start), pd.Timestamp(end), "split")
        for name, (start, end) in split.ranges.items()
    ]


def named_intervals(config: EvalConfig) -> list[Interval]:
    """``EVAL__INTERVALS`` - the bull/bear/turbulent sub-periods."""
    return [
        Interval(
            str(item.get("name", f"interval_{index}")),
            pd.Timestamp(item["start"]),
            pd.Timestamp(item["end"]),
            "regime",
        )
        for index, item in enumerate(config.intervals)
    ]


def resolve(
    ledger: pd.DataFrame, split: SplitConfig, config: EvalConfig
) -> list[Interval]:
    """Every interval worth scoring, clipped to what the run actually covers."""
    dates = pd.to_datetime(ledger["date"])
    first, last = dates.min(), dates.max()
    candidates = [
        Interval("full", first, last, "full"),
        *split_intervals(split),
        *named_intervals(config),
    ]
    return [item for item in candidates if item.mask(dates).any()]


def slice_ledger(ledger: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    dates = pd.to_datetime(ledger["date"])
    return ledger.loc[interval.mask(dates)].reset_index(drop=True)


def slice_weights(weights: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    dates = pd.to_datetime(weights["date"])
    return weights.loc[interval.mask(dates)].reset_index(drop=True)


def weight_matrix(weights: pd.DataFrame) -> pd.DataFrame:
    """Long (date, asset, weight) -> wide date x asset, missing assets at zero."""
    return (
        weights.pivot_table(
            index="date", columns="asset", values="weight", aggfunc="last"
        )
        .fillna(0.0)
        .sort_index()
    )
