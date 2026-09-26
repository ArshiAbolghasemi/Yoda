"""Feature resolution and the gate factory - the pieces the CIO is built from.

The CIO itself lives in :mod:`yoda.cio.agent`. These two stay here because both
are consulted *while* a CIO is being assembled: ``resolve_features`` before it
exists, ``fit_gate`` after it exists but before its gate does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from yoda.common.alignment import AlignedPanel
from yoda.common.types import Gate
from yoda.config.settings import Config
from yoda.specialists.jev import build_jev_features
from yoda.tailvoi.base import summarise
from yoda.tailvoi.baselines import AccuracyGate, AttentionGate, EqualWeightGate
from yoda.tailvoi.counterfactual import counterfactual_deltas, sample_positions
from yoda.tailvoi.tailvoi_gate import TailVoIGate

if TYPE_CHECKING:  # importing the CIO at runtime would be a cycle
    from yoda.cio.agent import CIOAgent


def resolve_features(
    panel: AlignedPanel, config: Config, features: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Materialise the feature cube for each configured channel.

    Every channel's features are OpenJev's calibrated probabilities. The panel's
    indicator cubes are still built, but they are the *state* each decision task
    is shown, not the specialist's input.

    Pre-built cubes passed in ``features`` are never rebuilt: the experiment
    harness uses that to share one expensive Jev build across many arms.
    """
    research = config.research
    supplied = features or {}
    resolved: dict[str, np.ndarray] = {}
    for channel in research.tailvoi.sources:
        if channel == "news" and research.news.backend == "none":
            continue
        if channel in supplied:
            resolved[channel] = supplied[channel]
            continue
        resolved[channel], _ = build_jev_features(panel, config, channel)
    return resolved


def fit_gate(
    config: Config, kind: str | None = None
) -> Callable[[CIOAgent, np.ndarray], Gate]:
    """Build the gate the CIO will contain: ``(cio, train_rows) -> Gate``.

    ``kind`` defaults to ``TAILVOI__GATE``. Tail-VoI is the proposal; the other
    three are the controls it has to beat, and each closes a specific escape
    route - "does gating do anything at all", "is this just accuracy weighting",
    "is this just any fitted gate beating a fixed one".

    Tail-VoI is fitted here rather than up front because it needs the
    counterfactual generator to have run, which needs a CIO: the engine hands
    over one already fitted on the fold's training rows.
    """
    research = config.research
    kind = kind or research.tailvoi.gate

    def build(cio, rows: np.ndarray) -> Gate:
        if kind == "equal_weight":
            return EqualWeightGate()

        if kind == "accuracy":
            gate = AccuracyGate(temperature=research.tailvoi.temperature)
            views = {
                name: specialist.predict(cio.features[name][rows][:, cio.universe])
                for name, specialist in cio.specialists.items()
            }
            gate.fit(views, cio.panel.targets[rows][:, cio.universe])
            return gate

        if kind == "attention":
            positions = sample_positions(rows, research.tailvoi.n_states, 1)
            outputs = [cio.specialist_output(position) for position in positions]
            sources = tuple(cio.sources)
            gate = AttentionGate(seed=research.tailvoi.seed)
            gate.fit(
                np.stack([summarise(spec, sources) for spec in outputs]),
                np.stack(
                    [np.stack([spec.mu_hat[n] for n in sources]) for spec in outputs]
                ),
                cio.panel.targets[positions][:, cio.universe],
                sources,
            )
            return gate

        if kind == "tailvoi":
            targets = counterfactual_deltas(
                cio, rows, research.tailvoi, research.static_policy
            )
            gate = TailVoIGate(research.tailvoi)
            gate.fit(targets)
            return gate

        raise ValueError(f"Unknown gate: {kind!r}")

    return build
