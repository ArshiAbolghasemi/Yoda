"""The news channel's headline history must look backwards and stay bounded."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from yoda.specialists.jev.features import _news_states, config_separator

DAYS = pd.to_datetime([f"2020-01-{d:02d}" for d in range(1, 11)])
LOOKBACK = 30


def _panel():
    separator = config_separator()
    # One headline per day for AAA, none for BBB: the empty asset proves that
    # a silent window stays empty rather than borrowing its neighbour's news.
    news = pd.DataFrame(
        [
            {
                "date": day,
                "asset": "AAA",
                "headlines": separator.join([f"story {n}", f"story {n} follow-up"]),
                "news_count": 2,
            }
            for n, day in enumerate(DAYS)
        ]
    )
    return SimpleNamespace(
        dates=DAYS, assets=["AAA", "BBB"], asset_types=["equity", "equity"], news=news
    )


def _states(lookback=LOOKBACK, history_max=40):
    frame = _news_states(
        _panel(), horizon=5, lookback=lookback, history_max=history_max
    )
    return {(row.asset, row.date): row.state for row in frame.itertuples()}


def test_history_is_strictly_backward_looking():
    states = _states()
    for (asset, day), state in states.items():
        for entry in state["recent_headlines"]:
            assert entry["date"] < day, f"{asset} {day} saw {entry['date']}"
            assert 1 <= entry["trading_days_ago"] <= LOOKBACK
        # Today's own headlines never appear twice.
        assert (
            all(
                entry["headlines"] != state["headlines"]
                for entry in state["recent_headlines"]
            )
            or not state["headlines"]
        )


def test_history_grows_then_is_capped():
    states = _states()
    first = states[("AAA", "2020-01-01")]
    fifth = states[("AAA", "2020-01-05")]
    assert first["recent_headlines"] == []  # nothing precedes day one
    assert len(fifth["recent_headlines"]) == 4  # every prior day carried news
    assert [e["trading_days_ago"] for e in fifth["recent_headlines"]] == [1, 2, 3, 4]


def test_history_respects_the_item_budget():
    states = _states(history_max=3)
    last = states[("AAA", "2020-01-10")]
    carried = sum(len(entry["headlines"]) for entry in last["recent_headlines"])
    assert carried <= 3
    # The budget spends on the most recent day first.
    assert last["recent_headlines"][0]["trading_days_ago"] == 1


def test_silent_asset_has_no_history_and_is_skipped():
    frame = _news_states(_panel(), horizon=5, lookback=LOOKBACK)
    silent = frame[frame["asset"] == "BBB"]
    assert silent["skip"].all()
    assert all(state["recent_headlines"] == [] for state in silent["state"])


def test_zero_lookback_restores_the_single_day_state():
    states = _states(lookback=0)
    assert all(state["recent_headlines"] == [] for state in states.values())
