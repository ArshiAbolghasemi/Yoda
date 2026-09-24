"""The CIO agent: one decision-maker, no model inference."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from yoda.cio.agent import CIOAgent
from yoda.common.types import (
    DROCVaROptimizer,
    Gate,
    MarketState,
    RiskParamPolicy,
    RiskParams,
)


@pytest.fixture
def cio(config, stack, train_rows):
    agent = CIOAgent(config, stack.sources)
    agent.fit(stack, train_rows)
    return agent


def _state(cvar: float = 0.02, assets: int = 4) -> MarketState:
    zeros = np.zeros(assets)
    return MarketState(zeros, zeros, {}, {"cvar": cvar}, zeros)


def test_fills_all_three_sockets():
    """Gate, policy and allocator - the whole decision in one object."""
    for contract in (Gate, RiskParamPolicy, DROCVaROptimizer):
        assert issubclass(CIOAgent, contract)


def test_uses_no_model_inference():
    """No OpenJev, no network: the agent must be pure and reproducible."""
    source = pathlib.Path("yoda/cio/agent.py").read_text()
    for forbidden in ("JevClient", "typesafe", "system_one", "resolve_model", "http"):
        assert forbidden not in source


def test_gate_is_a_distribution_over_the_live_sources(cio, stack, train_rows):
    spec = stack.specialist_output(int(train_rows[-1]))
    output = cio.gate(spec)
    assert set(output.g) == set(stack.sources)
    assert sum(output.g.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in output.g.values())


def test_gate_prefers_the_channel_that_is_saying_more(cio, stack, train_rows):
    """Trust follows cross-sectional dispersion, normalised per channel."""
    spec = stack.specialist_output(int(train_rows[-1]))
    sources = list(stack.sources)
    flat, loud = sources[0], sources[-1]
    spec.mu_hat[flat] = np.zeros_like(spec.mu_hat[flat])  # says nothing
    spec.mu_hat[loud] = np.linspace(-1, 1, len(spec.mu_hat[loud]))  # differentiates
    g = cio.gate(spec).g
    assert g[loud] > g[flat]


def test_is_deterministic(cio, stack, train_rows):
    spec = stack.specialist_output(int(train_rows[-1]))
    assert cio.gate(spec).g == cio.gate(spec).g


def test_stance_tightens_when_the_tail_widens(cio):
    calm, stressed = cio.act(_state(cvar=0.005)), cio.act(_state(cvar=0.10))
    assert stressed.budget < calm.budget
    assert stressed.lam > calm.lam
    assert calm.alpha == stressed.alpha  # alpha is never an action


def test_weights_are_long_only_on_the_simplex(cio, stack, scenarios, equal_book):
    n = stack.n_investable
    w = cio.solve(
        np.full(n, 0.0005), scenarios, RiskParams(5.0, 0.05, 0.001), equal_book
    )
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert w.max() <= max(cio.optimizer.max_weight, 1.5 / n) + 1e-6


def test_turnover_penalty_still_binds(cio, stack, scenarios):
    """The CVaR bound is advisory here, but the convex constraints are not."""
    n = stack.n_investable
    previous = np.zeros(n)
    previous[0] = 1.0
    mu = np.full(n, 0.0005)
    free = cio.solve(mu, scenarios, RiskParams(5.0, 0.05, 0.0), previous)
    sticky = cio.solve(mu, scenarios, RiskParams(5.0, 0.05, 5.0), previous)
    assert np.abs(sticky - previous).sum() < np.abs(free - previous).sum()


def test_installing_the_cio_replaces_the_optimizer(config, panel, train_rows):
    """gate='cio' hands the whole decision over, DRO-CVaR included."""
    from yoda.stack import build_stack, fit_gate
    from yoda.tailvoi.baselines import EqualWeightGate

    stack = build_stack(panel, config, train_rows, EqualWeightGate())
    assert not isinstance(stack.optimizer, CIOAgent)
    stack.gate = fit_gate("cio", config)(stack, train_rows)
    assert isinstance(stack.gate, CIOAgent)
    assert stack.optimizer is stack.gate
