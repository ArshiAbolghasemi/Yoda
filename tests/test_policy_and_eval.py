"""The policy seam, the backtest artifacts, and the offline metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yoda.backtest.artifacts import load_run, write_run
from yoda.backtest.engine import generate_folds
from yoda.common.types import MarketState, RiskParams
from yoda.config.research import RLConfig, StaticPolicyConfig
from yoda.evaluation.intervals import Interval, weight_matrix
from yoda.evaluation.metrics import effective_bets, entropy, max_drawdown, performance
from yoda.policy.rl import action_to_params
from yoda.policy.static import StaticRiskPolicy


def _state(cvar: float = 0.02) -> MarketState:
    zeros = np.zeros(4)
    return MarketState(zeros, zeros, {"technical": 1.0}, {"cvar": cvar}, zeros)


# ---- the seam ------------------------------------------------------------


def test_static_policy_is_config_driven():
    params = StaticRiskPolicy(StaticPolicyConfig(lam=2.0, budget=0.07)).act(_state())
    assert (params.lam, params.budget) == (2.0, 0.07)


def test_vol_target_tightens_the_budget_when_the_tail_widens():
    policy = StaticRiskPolicy(StaticPolicyConfig(budget=0.05, vol_target=0.02))
    assert policy.act(_state(cvar=0.04)).budget < policy.act(_state(cvar=0.01)).budget


def test_actions_stay_inside_the_configured_bounds():
    config = RLConfig()
    for action in (np.array([-2.0, -2, -2]), np.array([2.0, 2, 2]), np.zeros(3)):
        params = action_to_params(action, config, alpha=0.05)
        assert config.lam_bounds[0] <= params.lam <= config.lam_bounds[1]
        assert config.budget_bounds[0] <= params.budget <= config.budget_bounds[1]
        assert (
            config.turnover_bounds[0]
            <= params.turnover_penalty
            <= config.turnover_bounds[1]
        )


def test_alpha_is_never_an_action():
    """The agent moves lam, B and c - never its own risk-measure confidence."""
    actions = [np.array([-1.0, -1, -1]), np.zeros(3), np.array([1.0, 1, 1])]
    assert {action_to_params(a, RLConfig(), alpha=0.05).alpha for a in actions} == {
        0.05
    }


def test_risk_params_reject_nonsense():
    with pytest.raises(ValueError, match="alpha"):
        RiskParams(1.0, 0.05, 0.0, alpha=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        RiskParams(-1.0, 0.05, 0.0)


# ---- folds ---------------------------------------------------------------


def test_single_fold_by_default(panel, config):
    folds = generate_folds(panel, config)
    assert len(folds) == 1
    assert folds[0].train.max() < folds[0].test.min()  # no lookahead across the split


def test_rolling_refit_expands_the_history(panel, config):
    import dataclasses

    rolling = dataclasses.replace(
        config,
        research=dataclasses.replace(
            config.research,
            split=dataclasses.replace(config.research.split, refit_days=120),
        ),
    )
    folds = generate_folds(panel, rolling)
    assert len(folds) > 1
    assert len(folds[1].train) > len(folds[0].train)
    for fold in folds:
        assert fold.train.max() < fold.test.min()


# ---- artifacts -----------------------------------------------------------


def test_artifacts_round_trip(tmp_path):
    dates = pd.date_range("2024-01-01", periods=3)
    weights = pd.DataFrame(
        {"date": dates, "asset": ["AAPL"] * 3, "weight": [0.4, 0.5, 0.6]}
    )
    ledger = pd.DataFrame(
        {
            "date": dates,
            "port_return": [0.01, -0.02, 0.005],
            "turnover": [0.1, 0.0, 0.2],
            "realized_cvar": [0.02] * 3,
            "realized_var": [0.01] * 3,
            "cash": [0.0] * 3,
            "gross": [1.0] * 3,
        }
    )
    write_run(tmp_path, "run1", weights, ledger, {"pipeline": "tail_voli_risk"})
    run = load_run(tmp_path, "run1")
    assert run.meta["run_id"] == "run1"
    assert len(run.ledger) == 3
    assert weight_matrix(run.weights).shape == (3, 1)


def test_missing_columns_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        write_run(tmp_path, "bad", pd.DataFrame({"date": []}), pd.DataFrame(), {})


# ---- metrics -------------------------------------------------------------


def test_metrics_on_a_known_series():
    """+10% / -10% for 63 pairs: a 1% compounding loss per pair, peak at row 0."""
    returns = pd.Series([0.1, -0.1] * 63)
    row = performance(returns, alpha=0.05)
    assert row["TR"] == pytest.approx(0.99**63 - 1, rel=1e-9)
    assert row["MaxDD"] == pytest.approx(1 - 0.99**63 / 1.1, rel=1e-9)
    assert row["CVaR"] == pytest.approx(0.1, rel=1e-9)  # every tail row is -10%
    assert row["ES"] == pytest.approx(-0.1, rel=1e-9)


def test_max_drawdown_matches_the_worst_peak_to_trough():
    assert max_drawdown(pd.Series([0.5, -0.5, 0.0])) == pytest.approx(0.5)


def test_entropy_peaks_for_an_even_book():
    even = np.full(4, 0.25)
    assert entropy(even) == pytest.approx(np.log(4))
    assert entropy(np.array([0.97, 0.01, 0.01, 0.01])) < entropy(even)


def test_effective_bets_counts_independent_risk_sources():
    identity = np.eye(4)
    assert effective_bets(np.full(4, 0.25), identity) == pytest.approx(4.0)
    concentrated = np.array([1.0, 0.0, 0.0, 0.0])
    assert effective_bets(concentrated, identity) == pytest.approx(1.0)


def test_interval_masks_are_inclusive():
    dates = pd.Series(pd.date_range("2024-01-01", periods=5))
    interval = Interval(
        "x", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), "regime"
    )
    assert interval.mask(dates).sum() == 3
