"""The three agents as **decision** questions rather than a JSON prompt.

``jev-1.13`` is a decisions model: it refuses ``/v1/chat/completions`` outright
and answers ``/v1/systemone``, returning a calibrated probability per question.
That is the whole reason to use it. Asking it to emit a JSON blob of
self-assessed numbers - which is what a generative prompt does - throws away the
one property the evaluation depends on, because a Brier score over numbers a
model wrote about itself measures self-report, not calibration.

Two primitives carry every numeric field the views declare:

``Noul``   a yes/no question. Its answer *is* ``P(yes)`` on [0, 1], which is
           exactly the range of most view fields. Signed fields on [-1, 1] ask
           the directional form and map ``2p - 1``.
``Choice`` named alternatives with a probability each. Used for the one
           categorical field per channel.

The core variables are **derived from the Choice**, not asked separately:

    mu  expected_return_score   probability-weighted over the direction labels
    c   confidence              the answer's own confidence
    u   epistemic_uncertainty   normalised entropy of the label distribution

Asking a model "how confident are you?" returns an opinion; the entropy of its
own distribution is a measurement. Three fewer questions per call, and the ones
that matter most stop being self-reported.
"""

from __future__ import annotations

import math
from typing import Any

from yoda.specialists.prompts.schema import VIEWS

HORIZON = "over the next {horizon} trading days"


def horizon_phrase(horizon: int) -> str:
    """``{horizon}`` reads badly at 1; the model sees this text verbatim."""
    return (
        "over the next trading day"
        if horizon == 1
        else HORIZON.replace(HORIZON.format(horizon=horizon), horizon_phrase(horizon))
    )


# The categorical field each channel answers, and the numeric value each label
# carries when the distribution is collapsed to a scalar.
CATEGORICAL: dict[str, tuple[str, dict[str, float]]] = {
    "news": (
        "direction",
        {
            "strong_bearish": -1.0,
            "bearish": -0.5,
            "neutral": 0.0,
            "bullish": 0.5,
            "strong_bullish": 1.0,
        },
    ),
    "technical": (
        "direction",
        {
            "strong_bearish": -1.0,
            "bearish": -0.5,
            "neutral": 0.0,
            "bullish": 0.5,
            "strong_bullish": 1.0,
        },
    ),
    "volatility": (
        "volatility_regime",
        {"low": 0.0, "normal": 0.25, "elevated": 0.5, "high": 0.75, "extreme": 1.0},
    ),
}

LABELS: dict[str, dict[str, str]] = {
    "direction": {
        "strong_bearish": "a large fall is the most likely outcome",
        "bearish": "a fall is more likely than a rise",
        "neutral": "no directional edge either way",
        "bullish": "a rise is more likely than a fall",
        "strong_bullish": "a large rise is the most likely outcome",
    },
    "volatility_regime": {
        "low": "dispersion below this asset's normal range",
        "normal": "dispersion in line with its recent history",
        "elevated": "dispersion above normal but not disorderly",
        "high": "sustained wide swings",
        "extreme": "crisis-level dispersion",
    },
}

# Fields on [-1, 1]: the answer is P(the positive side), mapped to 2p - 1.
SIGNED = frozenset(
    {"momentum_score", "volume_confirmation", "volatility_impact", "directional_bias"}
)

# What each field asks. The state carries the evidence; these carry the task.
ASK: dict[str, str] = {
    "expected_magnitude": f"Will the move {HORIZON} be large relative to this asset's own recent typical move?",
    "downside_tail_risk": f"Is there meaningful risk of an unusually large *adverse* move {HORIZON}, beyond ordinary fluctuation?",
    "regime_shift_probability": "Does the evidence indicate a structural change rather than ordinary short-term noise?",
    "upside_tail_potential": f"Is there meaningful potential for an unusually large *favourable* move {HORIZON}?",
    "volatility_impact": "Will this information increase volatility rather than reduce it?",
    "trend_strength": "Do the indicators jointly describe a strong, coherent trend rather than a directionless market?",
    "momentum_score": "Is momentum positive rather than negative?",
    "volume_confirmation": "Does volume confirm the price move rather than contradict it?",
    "breakout_probability": f"Will price break decisively above its recent range {HORIZON}?",
    "breakdown_probability": f"Will price break decisively below its recent range {HORIZON}?",
    "signal_agreement": "Do trend, momentum and market structure agree with one another?",
    "volatility_expansion_probability": f"Will realised volatility be higher {HORIZON} than it is now?",
    "directional_bias": "Setting size aside, is the likely direction upward rather than downward?",
    "tail_event_probability": f"Will an unusually large adverse move occur {HORIZON}?",
    "tail_severity": "If such a move occurs, would the damage be severe rather than moderate?",
    "negative_skew_risk": "Is the return distribution skewed so that losses are larger than gains?",
    "heavy_tail_score": "Are extreme returns materially more likely than a Gaussian model would imply?",
    "jump_risk": "Is there risk of a discontinuous jump rather than a continuous move?",
    "liquidity_stress": "Is there evidence of deteriorating liquidity?",
    "systemic_risk_component": "Is the risk market-wide rather than specific to this asset?",
    "idiosyncratic_risk_component": "Is the risk specific to this asset rather than market-wide?",
}

# Derived from the categorical answer, never asked.
DERIVED = frozenset({"expected_return_score", "confidence", "epistemic_uncertainty"})


def _numeric_fields(channel: str) -> tuple[str, ...]:
    model = VIEWS[channel]
    return tuple(
        name
        for name, field in model.model_fields.items()
        if field.annotation is float and name not in DERIVED
    )


def build_questions(channel: str, horizon: int) -> dict[str, dict[str, Any]]:
    """Every question one call asks, as plain dicts the SDK coerces."""
    field, values = CATEGORICAL[channel]
    questions: dict[str, dict[str, Any]] = {
        field: {
            "type": "choice",
            "instructions": f"Which best describes the outlook {horizon_phrase(horizon)}?",
            "criteria": LABELS[field],
        }
    }
    skip = {field, *DERIVED}
    # The volatility channel's magnitude is its regime, so it is derived too.
    if channel == "volatility":
        skip |= {"volatility_score", "expected_magnitude"}
    for name in _numeric_fields(channel):
        if name in skip or name not in ASK:
            continue
        questions[name] = {
            "type": "noul",
            "instructions": ASK[name]
            .format(horizon=horizon)
            .replace(HORIZON.format(horizon=horizon), horizon_phrase(horizon)),
        }
    return questions


def _entropy(probabilities: dict[Any, float]) -> float:
    """Shannon entropy of the label distribution, normalised to [0, 1]."""
    values = [p for p in probabilities.values() if p > 0]
    if len(values) < 2:
        return 0.0
    total = -sum(p * math.log(p) for p in values)
    return min(1.0, total / math.log(len(probabilities)))


def decode(channel: str, answers: dict[str, Any], horizon: int) -> dict[str, float]:
    """``SystemOneResponse.answers`` -> the numeric record the cube stores."""
    field, values = CATEGORICAL[channel]
    categorical = answers[field]
    probabilities = dict(categorical.probabilities)
    # Collapse the distribution rather than taking the argmax: a 51/49 split
    # and a 99/1 split pick the same label and mean very different things.
    scalar = sum(
        values[name] * p for name, p in probabilities.items() if name in values
    )

    record: dict[str, float] = {
        # Carried, not asked: the cube's column layout includes it.
        "horizon_days": float(horizon),
        "confidence": float(categorical.confidence),
        "epistemic_uncertainty": _entropy(probabilities),
    }
    if channel == "volatility":
        record["volatility_score"] = scalar
        record["expected_magnitude"] = scalar
    else:
        record["expected_return_score"] = scalar

    for name, answer in answers.items():
        if name == field:
            continue
        value = float(answer.noul)
        record[name] = 2.0 * value - 1.0 if name in SIGNED else value

    if channel == "volatility":
        record["expected_return_score"] = record.get("directional_bias", 0.0)
    return record
