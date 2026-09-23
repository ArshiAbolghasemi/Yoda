"""The CIO agent: fills both sockets, and never sinks a run when it cannot decide."""

from __future__ import annotations

import numpy as np
import pytest

from yoda.cio.agent import CIOAgent
from yoda.common.types import Gate, RiskParamPolicy
from yoda.tailvoi.base import MU_SOURCES


@pytest.fixture
def cio(config, stack):
    return CIOAgent(config, stack.sources)


def test_fills_both_sockets():
    assert issubclass(CIOAgent, Gate)
    assert issubclass(CIOAgent, RiskParamPolicy)


def test_gate_falls_back_to_an_even_gate_when_the_endpoint_is_down(
    cio, stack, train_rows
):
    """No OpenJev server in CI: the run must continue, not crash."""
    spec = stack.specialist_output(int(train_rows[-1]))
    output = cio.gate(spec)
    assert sum(output.g.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in output.g.values())
    assert np.isfinite(output.fused_mu).all()


def test_stance_maps_into_the_same_action_space_sac_searches(cio, config):
    bounds = config.research.rl
    for stance in (0.0, 0.5, 1.0):
        cio._latest = {"stance": stance}
        params = cio.act(_state())
        assert bounds.lam_bounds[0] <= params.lam <= bounds.lam_bounds[1]
        assert bounds.budget_bounds[0] <= params.budget <= bounds.budget_bounds[1]
        assert params.alpha == config.research.optimizer.alpha


def test_aggressive_means_less_risk_aversion_and_a_wider_budget(cio):
    cio._latest = {"stance": 1.0}
    aggressive = cio.act(_state())
    cio._latest = {"stance": 0.0}
    defensive = cio.act(_state())
    assert aggressive.lam < defensive.lam
    assert aggressive.budget > defensive.budget


def test_cio_never_emits_weights(cio):
    """The optimizer keeps the constraints; the CIO only sets parameters."""
    assert not hasattr(cio, "solve")
    cio._latest = {"stance": 0.5}
    params = cio.act(_state())
    assert set(params.as_dict()) == {"lam", "budget", "turnover_penalty", "alpha"}


def test_the_brief_carries_no_future_information(cio, stack, train_rows):
    spec = stack.specialist_output(int(train_rows[-1]))
    brief = cio._brief(spec)
    flat = repr(brief)
    for forbidden in ("future", "target", "realized_return", "next_"):
        assert forbidden not in flat
    assert set(brief["channels"]) <= set(stack.sources)


def test_only_directional_sources_reach_mu_through_the_cio_gate(cio, stack, train_rows):
    spec = stack.specialist_output(int(train_rows[-1]))
    output = cio.gate(spec)
    assert set(output.g) == set(stack.sources)
    assert any(name in MU_SOURCES for name in output.g)


def _state():
    from yoda.common.types import MarketState

    zeros = np.zeros(4)
    return MarketState(zeros, zeros, {}, {"cvar": 0.02}, zeros)
