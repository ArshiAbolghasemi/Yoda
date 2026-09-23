"""Shared fixtures. The panel is built once - it reads a 59MB parquet."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from yoda.common.alignment import build_panel
from yoda.config import load_config


@pytest.fixture(scope="session")
def config():
    """Config pinned to the offline path.

    The shipped defaults route every channel through OpenJev, which needs a
    served model. The suite must run on a laptop with no GPU, so the fixtures
    pin the numeric backends and drop the news channel. The Jev and CIO paths
    are covered separately by their own fallback tests.
    """
    base = load_config()
    research = base.research
    return dataclasses.replace(
        base,
        research=dataclasses.replace(
            research,
            news=dataclasses.replace(research.news, backend="none"),
            # Every channel is OpenJev now, so the suite runs it in synthetic
            # mode: deterministic fake answers, no server, no GPU. It exercises
            # the real cache -> cube -> specialist path; only the answers are
            # fabricated, and any run made this way is stamped in run_meta.
            jev=dataclasses.replace(research.jev, synthetic=True),
            tailvoi=dataclasses.replace(
                research.tailvoi,
                sources=("technical", "volatility"),
                n_states=8,
                counterfactual_scenarios=200,
            ),
            copula=dataclasses.replace(research.copula, n_scenarios=400),
        ),
    )


@pytest.fixture(scope="session")
def panel(config):
    if not config.dataset_path.exists():
        pytest.skip(f"dataset not present: {config.dataset_path} (dvc pull)")
    return build_panel(config)


@pytest.fixture(scope="session")
def train_rows(panel, config):
    split = config.research.split
    return panel.positions(split.train_start, split.train_end)


@pytest.fixture(scope="session")
def stack(panel, config, train_rows):
    from yoda.stack import build_stack
    from yoda.tailvoi.baselines import EqualWeightGate

    return build_stack(panel, config, train_rows, EqualWeightGate())


@pytest.fixture
def scenarios(stack):
    return stack.copula.sample(400)


@pytest.fixture
def equal_book(stack):
    return np.full(stack.n_investable, 1.0 / stack.n_investable)
