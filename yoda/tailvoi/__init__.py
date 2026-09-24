"""Tail Value-of-Information gating, and the baseline gates it must beat."""

from yoda.common.types import Gate, GateOutput
from yoda.tailvoi.base import (
    CONDITIONING_SOURCES,
    MU_SOURCES,
    conditioning_strength,
    fuse,
    softmax,
    summarise,
)
from yoda.tailvoi.baselines import AccuracyGate, AttentionGate, EqualWeightGate
from yoda.tailvoi.counterfactual import (
    CounterfactualTargets,
    counterfactual_deltas,
    sample_positions,
)
from yoda.tailvoi.tailvoi_gate import TailVoIGate

__all__ = [
    "CONDITIONING_SOURCES",
    "MU_SOURCES",
    "AccuracyGate",
    "AttentionGate",
    "CounterfactualTargets",
    "EqualWeightGate",
    "Gate",
    "GateOutput",
    "TailVoIGate",
    "conditioning_strength",
    "counterfactual_deltas",
    "fuse",
    "sample_positions",
    "softmax",
    "summarise",
]
