"""The CIO: Tail-VoI gate + risk policy + DRO-CVaR, returning weights."""

from __future__ import annotations

import pytest

from yoda.cio import CIOAgent, CIODecision, build_cio
from yoda.policy.static import StaticRiskPolicy
from yoda.tailvoi.tailvoi_gate import TailVoIGate


@pytest.fixture
def cio(config, stack):
    return build_cio(stack, StaticRiskPolicy(config.research.static_policy))


def test_contains_the_gate_the_policy_and_the_allocator(cio, stack):
    assert cio.gate is stack.gate
    assert cio.optimizer is stack.optimizer
    assert isinstance(cio.policy, StaticRiskPolicy)


def test_the_gate_is_selectable(config, stack, train_rows):
    """The CIO contains whichever gate the config or flag chose."""
    from yoda.stack import fit_gate
    from yoda.tailvoi import AccuracyGate, EqualWeightGate

    for kind, expected in (
        ("tailvoi", TailVoIGate),
        ("accuracy", AccuracyGate),
        ("equal_weight", EqualWeightGate),
    ):
        stack.gate = fit_gate(config, kind)(stack, train_rows)
        agent = build_cio(stack, StaticRiskPolicy(config.research.static_policy))
        assert isinstance(agent.gate, expected)


def test_cio_still_only_returns_weights(cio):
    """Whatever gate it contains, the CIO's output is a portfolio."""
    from yoda.common.types import DROCVaROptimizer, Gate, RiskParamPolicy

    for contract in (Gate, RiskParamPolicy, DROCVaROptimizer):
        assert not isinstance(cio, contract)


def test_decide_returns_weights_on_the_simplex(cio, stack, train_rows, equal_book):
    position = int(train_rows[-1])
    state, scen, _ = stack.state(position, equal_book)
    spec = stack.specialist_output(position)
    decision = cio.decide(spec, scen, state, equal_book)

    assert isinstance(decision, CIODecision)
    w = decision.weights
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_decision_records_what_it_decided(cio, stack, train_rows, equal_book):
    """The ledger needs the gate weights and the risk parameters."""
    position = int(train_rows[-1])
    state, scen, _ = stack.state(position, equal_book)
    decision = cio.decide(stack.specialist_output(position), scen, state, equal_book)

    assert sum(decision.gate.g.values()) == pytest.approx(1.0)
    assert decision.risk.alpha == cio.policy.alpha
    assert decision.risk.lam >= 0


def test_the_policy_is_the_only_interchangeable_part(config, stack):
    """Static or SAC by flag; the gate and the allocator do not vary."""
    static = build_cio(stack, StaticRiskPolicy(config.research.static_policy))
    other = build_cio(stack, StaticRiskPolicy(config.research.static_policy))
    assert static.gate is other.gate
    assert static.optimizer is other.optimizer


def test_cio_does_not_emit_weights_itself(cio):
    """It composes the solver; it is not a solver."""
    assert not hasattr(CIOAgent, "solve")
    assert hasattr(cio.optimizer, "solve")
