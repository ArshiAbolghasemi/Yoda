"""The panel contract, checked against the real dataset."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from yoda.common.alignment import build_panel
from yoda.config.research import SplitConfig


def test_one_calendar_shared_by_every_asset(panel):
    assert panel.returns.shape == (len(panel.dates), panel.n_assets)
    assert panel.dates.is_monotonic_increasing
    assert not panel.dates.duplicated().any()
    for cube in panel.features.values():
        assert cube.shape[:2] == panel.returns.shape


def test_returns_are_finite_and_mask_the_unlisted(panel):
    assert np.isfinite(panel.returns).all()
    # DOW listed in 2019, so the mask must be false somewhere and true mostly.
    assert not panel.available.all()
    assert panel.available.mean() > 0.9
    late = panel.assets.index("DOW")
    assert not panel.available[0, late]
    assert panel.available[-1, late]


def test_targets_are_forward_returns_of_the_aligned_prices(panel):
    horizon = panel.horizon
    expected = panel.prices[horizon:] / panel.prices[:-horizon] - 1.0
    actual = panel.targets[:-horizon]
    both = np.isfinite(expected) & np.isfinite(actual)
    assert np.allclose(expected[both], actual[both])
    # The label must never be knowable at its own row: the last rows have none.
    assert np.isnan(panel.targets[-horizon:]).all()


def test_btc_weekend_moves_are_folded_not_dropped(panel, config):
    """BTC trades 24/7; on the NYSE calendar a Monday return spans the weekend."""
    raw = pd.read_parquet(config.dataset_path)
    btc = raw.loc[raw["asset"].eq("BTC-USD")].set_index("date")["adjusted_close"]
    column = panel.assets.index("BTC-USD")
    mondays = np.flatnonzero(panel.dates.dayofweek == 0)[5:10]
    for row in mondays:
        previous, current = panel.dates[row - 1], panel.dates[row]
        if previous not in btc.index or current not in btc.index:
            continue
        assert panel.returns[row, column] == pytest.approx(
            btc[current] / btc[previous] - 1.0, rel=1e-9
        )


def test_universe_excludes_assets_without_history(panel, train_rows):
    universe = panel.universe(train_rows)
    assert universe.sum() >= 30
    assert not universe[panel.assets.index("DOW")]


def test_fx_volume_indicators_stay_nan_rather_than_zero(panel):
    fx = [index for index, kind in enumerate(panel.asset_types) if kind == "fx"]
    names = panel.feature_names["technical"]
    obv = names.index("obv")
    assert np.isnan(panel.features["technical"][:, fx, obv]).all()


def test_slice_preserves_the_contract(panel, config):
    split = config.research.split
    piece = panel.slice(split.test_start, split.test_end)
    assert len(piece.dates) < len(panel.dates)
    assert piece.returns.shape[1] == panel.n_assets
    assert piece.dates[0] >= pd.Timestamp(split.test_start)


def test_union_calendar_is_at_least_as_long_as_the_intersection(config):
    intersected = build_panel(
        dataclasses.replace(
            config,
            research=dataclasses.replace(
                config.research,
                panel=dataclasses.replace(
                    config.research.panel, calendar="intersection"
                ),
            ),
        )
    )
    default = build_panel(config)
    assert len(default.dates) >= len(intersected.dates)


def test_split_boundaries_must_increase():
    with pytest.raises(ValueError, match="must increase"):
        SplitConfig(train_end="2022-12-31", val_start="2021-01-01")
