"""One common trading calendar for a mixed-calendar panel.

BTC trades 24/7, FX 24/5, equities on the NYSE session. Every cross-asset step
(copula, optimizer, RL) needs a single date index, so the panel is reindexed
onto one calendar *before* returns are computed: an off-calendar move (a BTC
weekend) is folded into the next trading day's return rather than dropped, and
short holiday gaps are carried forward.

Assets do not all start together - DOW listed in 2019 - so the panel also
carries an ``available`` mask. Everything downstream trades the per-fold
universe that mask implies; nothing is back-filled into an asset's pre-listing
history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from yoda.common.logger import logger
from yoda.config.settings import Config
from yoda.data.features import INDICATOR_COLUMNS

VOLATILITY_COLUMNS: tuple[str, ...] = (
    "atr_7",
    "atr_14",
    "atr_21",
    "bb_width_20",
    "keltner_width",
)
SCALED_BY_PRICE: frozenset[str] = frozenset({"atr_7", "atr_14", "atr_21"})


@dataclass(frozen=True)
class AlignedPanel:
    """The aligned panel every downstream module reads."""

    dates: pd.DatetimeIndex
    assets: tuple[str, ...]
    asset_types: tuple[str, ...]
    returns: np.ndarray  # (T, N) simple returns; 0 where unavailable
    prices: np.ndarray  # (T, N) adjusted close, forward-filled over short gaps
    available: np.ndarray  # (T, N) bool: listed, priced and past warm-up
    features: dict[str, np.ndarray]  # source -> (T, N, F)
    feature_names: dict[str, tuple[str, ...]]
    targets: np.ndarray  # (T, N) forward return over the configured horizon
    news: pd.DataFrame  # long: date, asset, headlines, news_count
    horizon: int

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def positions(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> np.ndarray:
        """Integer positions of the dates inside ``[start, end]``."""
        mask = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return np.flatnonzero(mask)

    def universe(self, rows: np.ndarray, min_fraction: float = 0.95) -> np.ndarray:
        """Assets available on at least ``min_fraction`` of ``rows`` (bool, (N,))."""
        if len(rows) == 0:
            return np.zeros(self.n_assets, dtype=bool)
        return self.available[rows].mean(axis=0) >= min_fraction

    def slice(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> AlignedPanel:
        """A panel restricted to a date range (targets keep their own horizon)."""
        rows = self.positions(start, end)
        return self.take(rows)

    def take(self, rows: np.ndarray) -> AlignedPanel:
        dates = self.dates[rows]
        return AlignedPanel(
            dates=dates,
            assets=self.assets,
            asset_types=self.asset_types,
            returns=self.returns[rows],
            prices=self.prices[rows],
            available=self.available[rows],
            features={name: cube[rows] for name, cube in self.features.items()},
            feature_names=self.feature_names,
            targets=self.targets[rows],
            news=self.news[self.news["date"].isin(dates)].reset_index(drop=True),
            horizon=self.horizon,
        )

    def news_at(self, position: int) -> pd.DataFrame:
        """Headline rows for one date, reindexed to the panel's asset order."""
        day = self.news[self.news["date"] == self.dates[position]]
        return (
            day.set_index("asset")
            .reindex(list(self.assets))
            .assign(
                headlines=lambda frame: frame["headlines"].fillna(""),
                news_count=lambda frame: frame["news_count"].fillna(0).astype(int),
            )
            .rename_axis("asset")
            .reset_index()
        )


def _calendar(frame: pd.DataFrame, policy: str) -> pd.DatetimeIndex:
    per_asset = frame.groupby("asset", sort=False)["date"].apply(set)
    if policy == "union":
        dates = set().union(*per_asset)
    elif policy == "intersection":
        dates = set.intersection(*per_asset)
    else:
        # "equity": the NYSE session = every date any listed equity traded.
        # Intersecting instead would truncate the panel to the youngest listing.
        equity = frame.loc[frame["asset_type"].eq("equity")]
        if equity.empty:
            raise ValueError("Calendar policy 'equity' needs equity rows in the panel")
        dates = set().union(*equity.groupby("asset", sort=False)["date"].apply(set))
    return pd.DatetimeIndex(sorted(dates))


def _cube(
    wide: pd.DataFrame,
    columns: list[str],
    dates: pd.DatetimeIndex,
    assets: list[str],
) -> np.ndarray:
    """Reindex an unstacked (field, asset) frame into a (T, N, F) cube."""
    target = pd.MultiIndex.from_product([columns, assets], names=["field", "asset"])
    values = wide.reindex(index=dates, columns=target).to_numpy(dtype=np.float64)
    return values.reshape(len(dates), len(columns), len(assets)).transpose(0, 2, 1)


def build_panel(config: Config) -> AlignedPanel:
    """Load the dataset and align every asset onto one trading calendar."""
    settings = config.research.panel
    path: Path = config.dataset_path
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])

    dates = _calendar(frame, settings.calendar)
    assets = sorted(frame["asset"].unique())
    asset_types = tuple(
        frame.drop_duplicates("asset").set_index("asset")["asset_type"].reindex(assets)
    )

    fields = ["adjusted_close", *INDICATOR_COLUMNS]
    wide = frame.set_index(["date", "asset"])[fields].unstack("asset")

    prices = (
        wide["adjusted_close"]
        .reindex(index=dates, columns=assets)
        .ffill(limit=settings.max_forward_fill)
        .to_numpy(dtype=np.float64)
    )
    raw_returns = np.vstack(
        [np.full((1, len(assets)), np.nan), prices[1:] / prices[:-1] - 1.0]
    )

    # Available once priced and past the indicator warm-up for that asset.
    listed = np.isfinite(prices) & np.isfinite(raw_returns)
    age = listed.cumsum(axis=0)
    available = listed & (age > settings.min_history)
    returns = np.where(np.isfinite(raw_returns), raw_returns, 0.0)

    technical = _cube(wide, list(INDICATOR_COLUMNS), dates, assets)
    volatility = _cube(wide, list(VOLATILITY_COLUMNS), dates, assets)
    scaled = [name in SCALED_BY_PRICE for name in VOLATILITY_COLUMNS]
    # ATR is in price units; divide by price so assets are comparable.
    volatility[:, :, scaled] /= prices[:, :, None]

    realized = [
        pd.DataFrame(np.where(listed, raw_returns, np.nan))
        .rolling(window, min_periods=window)
        .std(ddof=0)
        .to_numpy()
        for window in settings.realized_vol_windows
    ]
    volatility = np.concatenate([volatility, np.stack(realized, axis=-1)], axis=-1)
    volatility_names = (
        *VOLATILITY_COLUMNS,
        *(f"realized_vol_{window}" for window in settings.realized_vol_windows),
    )

    horizon = settings.target_horizon
    targets = np.full_like(prices, np.nan)
    targets[:-horizon] = prices[horizon:] / prices[:-horizon] - 1.0
    targets = np.where(available, targets, np.nan)

    news = (
        frame.loc[
            frame["date"].isin(dates), ["date", "asset", "headlines", "news_count"]
        ]
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )

    head = available.any(axis=1).argmax()  # first date with any tradable asset
    keep = slice(int(head), None)
    panel = AlignedPanel(
        dates=dates[keep],
        assets=tuple(assets),
        asset_types=asset_types,
        returns=returns[keep],
        prices=prices[keep],
        available=available[keep],
        features={"technical": technical[keep], "volatility": volatility[keep]},
        feature_names={
            "technical": tuple(INDICATOR_COLUMNS),
            "volatility": volatility_names,
        },
        targets=targets[keep],
        news=news[news["date"].isin(dates[keep])].reset_index(drop=True),
        horizon=horizon,
    )
    logger.info(
        "panel_built rows=%d assets=%d start=%s end=%s calendar=%s available=%.3f",
        len(panel.dates),
        panel.n_assets,
        panel.dates[0].date(),
        panel.dates[-1].date(),
        settings.calendar,
        panel.available.mean(),
    )
    return panel
