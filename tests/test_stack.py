"""Specialists, copula, optimizer and gates - the layers below the seam."""

from __future__ import annotations

import numpy as np
import pytest

from yoda.common.types import MarketState, RiskParams
from yoda.copula.student_t import StudentTCopula, tail_stats
from yoda.optimizer.dro_cvar import DROCVaR
from yoda.tailvoi.base import MU_SOURCES, conditioning_strength, fuse

# ---- specialists ---------------------------------------------------------


def test_specialists_encode_and_predict_per_asset(stack, train_rows):
    position = int(train_rows[-1])
    output = stack.specialist_output(position)
    for name in stack.sources:
        assert output.z[name].shape == (
            stack.n_investable,
            stack.config.research.specialists.z_dim,
        )
        assert output.mu_hat[name].shape == (stack.n_investable,)
        assert np.isfinite(output.z[name]).all()
        assert np.isfinite(output.mu_hat[name]).all()


def test_volatility_view_is_non_negative(stack, train_rows):
    output = stack.specialist_output(int(train_rows[-1]))
    assert (output.mu_hat["volatility"] >= 0).all()


def test_missing_fx_volume_is_sent_as_null_not_a_sentinel(panel):
    """FX volume indicators are NaN by design.

    They no longer reach a specialist - the specialists read OpenJev
    probabilities - but they are still part of the *state* the decision model
    is shown, and a NaN there must travel as JSON null. Encoding it as 0.0
    would tell the model "volume was flat" when the truth is "no volume data".
    """
    from yoda.specialists.jev.features import _rounded

    names = panel.feature_names["technical"]
    obv = names.index("obv")
    fx = [i for i, kind in enumerate(panel.asset_types) if kind == "fx"]
    assert np.isnan(panel.features["technical"][:, fx, obv]).all()
    assert _rounded(float("nan")) is None
    assert _rounded(1.23456789) == 1.234568  # 6dp keeps the key stable


def test_representations_are_the_same_width_across_channels(stack, train_rows):
    """The gate sums z across channels, so every channel must be z_dim wide.

    The OpenJev cubes differ in width (13 technical, 15 volatility, 27 news),
    so this only holds because encode() pads narrow channels and z_dim is set
    wide enough that none is truncated.
    """
    output = stack.specialist_output(int(train_rows[-1]))
    widths = {name: z.shape[-1] for name, z in output.z.items()}
    assert set(widths.values()) == {stack.config.research.specialists.z_dim}


# ---- copula --------------------------------------------------------------


def test_correlation_is_symmetric_psd_with_unit_diagonal(stack):
    R = stack.copula.R
    assert np.allclose(R, R.T)
    assert np.allclose(np.diag(R), 1.0)
    assert np.linalg.eigvalsh(R).min() > 0


def test_nu_comes_from_the_configured_grid(stack, config):
    assert stack.copula.nu in config.research.copula.nu_grid


def test_sampling_reproduces_the_marginal_scale(stack):
    sample = stack.copula.sample(4000).scenarios
    assert sample.shape == (4000, stack.n_investable)
    ratio = sample.std(axis=0) / stack.copula.scale
    assert 0.5 < float(np.median(ratio)) < 2.0


def test_conditioning_rescales_only_when_a_regime_is_given(stack):
    zeros = np.zeros(stack.n_investable)
    blank = MarketState(zeros, zeros, {}, {}, zeros, regime=None)
    assert np.allclose(stack.copula.conditional(blank).scale, stack.copula.scale)

    doubled = stack.copula.scale * 2
    state = MarketState(zeros, zeros, {}, {}, zeros, regime=doubled)
    assert np.allclose(stack.copula.conditional(state).scale, doubled)


def test_conditioning_rejects_a_mis_shaped_regime(stack):
    zeros = np.zeros(stack.n_investable)
    state = MarketState(zeros, zeros, {}, {}, zeros, regime=np.ones(3))
    with pytest.raises(ValueError, match="per-asset"):
        stack.copula.conditional(state)


def test_copula_needs_enough_history():
    with pytest.raises(ValueError, match="T >= 50"):
        StudentTCopula().fit(np.random.default_rng(0).normal(size=(10, 4)))


# ---- optimizer -----------------------------------------------------------


def test_solution_is_long_only_and_on_the_simplex(stack, scenarios, equal_book, config):
    mu = np.full(stack.n_investable, 0.0005)
    w = DROCVaR(config.research.optimizer).solve(
        mu, scenarios, RiskParams(1.0, 0.05, 0.001), equal_book
    )
    assert w.shape == (stack.n_investable,)
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w.max() <= config.research.optimizer.max_weight + 1e-6


def test_risk_aversion_lowers_the_tail(stack, scenarios, equal_book, config):
    mu = stack.copula.scale * 0.1  # a mean signal worth chasing
    timid, bold = (
        DROCVaR(config.research.optimizer).solve(
            mu, scenarios, RiskParams(lam, 0.5, 0.0), equal_book
        )
        for lam in (50.0, 0.01)
    )
    assert (
        tail_stats(timid, scenarios, 0.05)["cvar"]
        < tail_stats(bold, scenarios, 0.05)["cvar"]
    )


def test_turnover_penalty_keeps_the_book_closer_to_where_it_was(
    stack, scenarios, config
):
    mu = stack.copula.scale * 0.1
    previous = np.zeros(stack.n_investable)
    previous[0] = 1.0
    free, sticky = (
        DROCVaR(config.research.optimizer).solve(
            mu, scenarios, RiskParams(1.0, 0.5, penalty), previous
        )
        for penalty in (0.0, 5.0)
    )
    assert np.abs(sticky - previous).sum() < np.abs(free - previous).sum()


def test_an_impossible_budget_falls_back_instead_of_raising(
    stack, scenarios, equal_book, config
):
    mu = np.full(stack.n_investable, 0.0005)
    w = DROCVaR(config.research.optimizer).solve(
        mu, scenarios, RiskParams(1.0, 1e-9, 0.001), equal_book
    )
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_solver_rejects_mismatched_shapes(stack, scenarios, config):
    with pytest.raises(ValueError, match="asset count"):
        DROCVaR(config.research.optimizer).solve(
            np.zeros(3), scenarios, RiskParams(1.0, 0.05, 0.0), np.zeros(3)
        )


# ---- gates ---------------------------------------------------------------


def test_gate_weights_are_a_distribution(stack, train_rows):
    spec = stack.specialist_output(int(train_rows[-1]))
    output = stack.gate.gate(spec)
    assert sum(output.g.values()) == pytest.approx(1.0)
    assert output.fused_mu.shape == (stack.n_investable,)


def test_only_directional_sources_reach_fused_mu(stack, train_rows):
    """The volatility view conditions the copula; it must never enter mu."""
    spec = stack.specialist_output(int(train_rows[-1]))
    weights = {name: (1.0 if name in MU_SOURCES else 0.0) for name in stack.sources}
    directional = fuse(spec, weights)
    both = fuse(spec, dict.fromkeys(stack.sources, 1.0))
    assert np.allclose(directional.fused_mu, both.fused_mu)


def test_dropping_volatility_deconditions_the_scenarios(stack, train_rows):
    spec = stack.specialist_output(int(train_rows[-1]))
    kept = fuse(spec, dict.fromkeys(stack.sources, 1.0 / len(stack.sources)))
    dropped = fuse(spec, {**dict.fromkeys(stack.sources, 0.5), "volatility": 0.0})
    assert conditioning_strength(kept) > 0
    assert conditioning_strength(dropped) == 0
    assert not np.allclose(
        stack.scenarios(spec, kept, 400).scenarios.std(axis=0),
        stack.scenarios(spec, dropped, 400).scenarios.std(axis=0),
    )
