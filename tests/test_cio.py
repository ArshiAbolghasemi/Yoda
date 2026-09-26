"""The CIO: every specialist reports in, one portfolio comes out.

The CIO owns the gate, the copula, the risk policy and the allocator. These
tests pin the two properties that make that safe: there is exactly one decision
site, and the decision records enough to audit itself.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from yoda.cio import CIOAgent, CIODecision
from yoda.policy.static import StaticRiskPolicy
from yoda.tailvoi.tailvoi_gate import TailVoIGate


@pytest.fixture
def cio(config, stack):
    seated = copy.copy(stack)  # the session stack is shared; do not seat on it
    return seated.install_policy(StaticRiskPolicy(config.research.static_policy))


def test_it_owns_the_gate_the_copula_the_policy_and_the_allocator(cio):
    for part in ("gate", "copula", "policy", "optimizer", "specialists"):
        assert getattr(cio, part) is not None, part
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
        agent = copy.copy(stack)
        agent.gate = fit_gate(config, kind)(agent, train_rows)
        assert isinstance(agent.gate, expected)


def test_cio_still_only_returns_weights(cio):
    """Whatever it contains, the CIO's output is a portfolio."""
    from yoda.common.types import DROCVaROptimizer, Gate, RiskParamPolicy

    for contract in (Gate, RiskParamPolicy, DROCVaROptimizer):
        assert not isinstance(cio, contract)
    assert not hasattr(CIOAgent, "solve")
    assert hasattr(cio.optimizer, "solve")


def test_decide_returns_weights_on_the_simplex(cio, train_rows, equal_book):
    decision = cio.decide(int(train_rows[-1]), equal_book)

    assert isinstance(decision, CIODecision)
    w = decision.weights
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_decision_records_what_it_decided(cio, train_rows, equal_book):
    """The ledger needs the gate weights, the Tail-VoI and the risk params."""
    decision = cio.decide(int(train_rows[-1]), equal_book)

    assert sum(decision.gate.g.values()) == pytest.approx(1.0)
    assert decision.risk.alpha == cio.policy.alpha
    assert decision.risk.lam >= 0
    assert np.isfinite(decision.state.tail_stats["cvar"])


def test_tail_voi_predictions_reach_the_decision(config, stack, train_rows, equal_book):
    """delta_hat must travel with the decision, not be rebuilt from the state.

    It used to be recovered via ``getattr(state, "delta_hat", {})`` - and
    ``MarketState`` has no such field, so the ledger recorded ``{}`` for every
    source and the Tail-VoI evidence was silently lost.
    """
    from yoda.stack import fit_gate

    agent = copy.copy(stack)
    agent.gate = fit_gate(config, "tailvoi")(agent, train_rows)
    agent.install_policy(StaticRiskPolicy(config.research.static_policy))

    decision = agent.decide(int(train_rows[-1]), equal_book)
    assert set(decision.gate.delta_hat) == set(agent.sources)
    assert all(np.isfinite(v) for v in decision.gate.delta_hat.values())


def test_decide_is_the_only_decision_site(cio, train_rows, equal_book):
    """`decide` must equal state -> act -> allocate, or the engine could drift."""
    position = int(train_rows[-1])
    decision = cio.decide(position, equal_book)

    state, scen, gate = cio.state(position, equal_book)
    risk = cio.policy.act(state)

    assert risk == decision.risk
    assert gate.g == decision.gate.g
    # Weights are not compared across the two calls: each draws its own
    # scenario set, so they differ by sampling noise. What must hold is that
    # the solve is a pure function of (state, scenarios, risk).
    assert np.array_equal(
        cio.allocate(state, scen, risk), cio.allocate(state, scen, risk)
    )


def test_a_cio_without_a_policy_refuses_to_decide(stack, train_rows, equal_book):
    """Construction is phased; deciding before the seam is filled is a bug."""
    unseated = copy.copy(stack)
    unseated.policy = None
    with pytest.raises(RuntimeError, match="no risk policy"):
        unseated.decide(int(train_rows[-1]), equal_book)
