"""End-to-end: the counterfactual generator, both versions, and the seam itself."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from yoda.common.types import RiskParamPolicy
from yoda.pipeline.tail_voli_risk import run_tail_voli_risk
from yoda.pipeline.tail_voli_risk_rl import run_tail_voli_risk_rl
from yoda.stack import fit_gate
from yoda.tailvoi.counterfactual import counterfactual_deltas


@pytest.fixture(scope="module")
def short(config):
    """A cheap configuration: a few months of test window, tiny RL budget."""
    research = config.research
    return dataclasses.replace(
        config,
        research=dataclasses.replace(
            research,
            split=dataclasses.replace(
                research.split, test_start="2026-05-01", test_end="2026-09-18"
            ),
            rl=dataclasses.replace(
                research.rl,
                total_timesteps=120,
                learning_starts=50,
                buffer_size=500,
                env_scenarios=200,
            ),
            evaluation=dataclasses.replace(research.evaluation, plots=False),
        ),
    )


@pytest.mark.parametrize("kind", ["equal_weight", "accuracy", "attention", "tailvoi"])
def test_every_gate_produces_a_valid_distribution(stack, train_rows, config, kind):
    """All four gates answer the same question through the same interface."""
    gate = fit_gate(config, kind)(stack, train_rows)
    output = gate.gate(stack.specialist_output(int(train_rows[-1])))
    assert sum(output.g.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(value >= 0 for value in output.g.values())
    assert np.isfinite(output.fused_mu).all()


def test_unknown_gate_is_rejected(stack, train_rows, config):
    with pytest.raises(ValueError, match="Unknown gate"):
        fit_gate(config, "nope")(stack, train_rows)


def test_counterfactual_targets_have_one_delta_per_source(stack, train_rows, config):
    targets = counterfactual_deltas(
        stack, train_rows, config.research.tailvoi, config.research.static_policy
    )
    assert targets.deltas.shape == (len(targets.positions), len(stack.sources))
    assert targets.summaries.shape[0] == targets.deltas.shape[0]
    assert np.isfinite(targets.deltas).all()
    assert targets.positions.max() <= train_rows.max()


def test_static_pipeline_runs_and_scores(short, panel):
    evaluation = run_tail_voli_risk(
        short, run_id="test_a", panel=panel, make_plots=False
    )
    table = evaluation.tables["full"]
    assert len(table) == 1
    assert np.isfinite(table["TR"]).all()
    assert 0 <= table["ENT"].iloc[0] <= np.log(36) + 1e-9


def test_rl_pipeline_runs_and_scores(short, panel):
    evaluation = run_tail_voli_risk_rl(
        short, run_id="test_c", panel=panel, make_plots=False
    )
    assert np.isfinite(evaluation.tables["full"]["TR"]).all()


def test_a_and_c_differ_only_at_the_policy(short, panel):
    """The seam: both versions record the same gate, optimizer and universe."""
    from yoda.backtest.artifacts import load_run

    root = short.path(short.research.backtest.runs)
    a, c = load_run(root, "test_a").meta, load_run(root, "test_c").meta
    shared = ("gate", "news_backend", "alpha", "assets", "rebalance_days", "splits")
    assert {key: a[key] for key in shared} == {key: c[key] for key in shared}
    assert (a["pipeline"], c["pipeline"]) == ("tail_voli_risk", "tail_voli_risk_rl")
    assert a["policy"] != c["policy"]


def test_policies_satisfy_the_same_interface(short, panel):
    from yoda.policy.rl import RLRiskPolicy
    from yoda.policy.static import StaticRiskPolicy

    assert issubclass(StaticRiskPolicy, RiskParamPolicy)
    assert issubclass(RLRiskPolicy, RiskParamPolicy)


def test_policy_cross_product_doubles_every_arm():
    """`--policies static rl` runs the whole matrix under both controllers."""
    from yoda.pipeline.experiments import default_arms, expand_policies

    base = [a for a in default_arms() if a.family == "gate"]
    both = expand_policies(base, ("static", "rl"))
    assert len(both) == 2 * len(base)
    assert {a.rl for a in both} == {True, False}
    assert all(a.run_id.endswith(("__static", "__rl")) for a in both)


def test_policy_family_is_not_duplicated():
    """It already varies the policy; crossing it would alias it with itself."""
    from yoda.pipeline.experiments import default_arms, expand_policies

    policy_arms = [a for a in default_arms() if a.family == "policy"]
    assert len(expand_policies(policy_arms, ("static", "rl"))) == len(policy_arms)


def test_unknown_policy_is_rejected():
    from yoda.pipeline.experiments import default_arms, expand_policies

    with pytest.raises(ValueError, match="Unknown policies"):
        expand_policies(default_arms(), ("static", "quantum"))
