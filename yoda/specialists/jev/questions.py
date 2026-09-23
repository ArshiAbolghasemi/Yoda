"""Typed decision tasks, one bundle per specialist.

Each specialist sends **one** ``/v1/systemone`` request carrying ``model``,
``state`` and ``questions``. The three bundles are never combined: they are
separate information specialists, and merging them would make it impossible to
tell whether their judgements are complementary or merely correlated.

Only ``choice``, ``score`` and ``noul`` are used. Nothing asks for free text,
an explanation, or a trading recommendation — the model returns calibrated
distributions over declared options, which is the entire reason Brier score and
ECE are meaningful here.

Every response is validated before it is believed. A malformed answer is
rejected and recorded with an explicit ``inference_status``; it is never
silently replaced with zeros, because a zero vector is indistinguishable from a
confident neutral judgement and would quietly corrupt the gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from typesafe_sdk import Choice, Noul, Question, Score, SystemOneResponse

# Prompt-level point-in-time restriction. This is *not* the leakage control -
# yoda/specialists/jev/states.py is. This only tells the model what it must not
# reach for; the state construction makes it unable to.
PIT_GUARD = (
    "Use only the information explicitly supplied in this state. Treat the "
    "observation date as the present. Do not use, infer, or rely on events, "
    "prices, outcomes, or knowledge occurring after this date."
)

DIRECTION_LABELS: tuple[str, ...] = ("down", "flat", "up")
REGIME_LABELS: tuple[str, ...] = ("low", "normal", "high", "extreme")
EVENT_LABELS: tuple[str, ...] = (
    "earnings",
    "corporate",
    "regulatory",
    "macro",
    "market",
    "operational",
    "other",
    "none",
)
TREND_RUBRIC: tuple[str, ...] = (
    "no meaningful trend",
    "weak trend",
    "moderate trend",
    "strong trend",
    "very strong trend",
)
SEVERITY_RUBRIC: tuple[str, ...] = ("very low", "low", "normal", "high", "extreme")
SENTIMENT_RUBRIC: tuple[str, ...] = (
    "strongly negative",
    "negative",
    "neutral",
    "positive",
    "strongly positive",
)
MATERIALITY_RUBRIC: tuple[str, ...] = (
    "irrelevant",
    "low impact",
    "moderately important",
    "high impact",
    "major market-moving event",
)


class InferenceStatus:
    OK = "ok"
    NO_INPUT = "no_input"
    RETRY_EXHAUSTED = "retry_exhausted"
    INVALID_RESPONSE = "invalid_response"


class InvalidResponse(ValueError):
    """A response that failed validation; the caller records and retries."""


def _guard(instruction: str) -> str:
    return f"{instruction}\n\n{PIT_GUARD}"


# ---- question bundles ----------------------------------------------------

TECHNICAL_QUESTIONS: Mapping[str, Question] = {
    "direction": Choice(
        instructions=_guard(
            "Using only the supplied point-in-time technical market state, "
            "estimate the most likely cumulative price-return direction over "
            "the next configured trading horizon."
        ),
        criteria={
            "down": (
                "The asset is more likely to have a meaningfully negative "
                "cumulative return over the horizon."
            ),
            "flat": (
                "The asset is more likely to remain approximately unchanged "
                "over the horizon."
            ),
            "up": (
                "The asset is more likely to have a meaningfully positive "
                "cumulative return over the horizon."
            ),
        },
    ),
    "trend_strength": Score(
        instructions=_guard(
            "Rate the strength of the current directional trend using only the "
            "supplied technical state."
        ),
        criteria=list(TREND_RUBRIC),
    ),
    "reversal_risk": Noul(
        instructions=_guard(
            "Does the supplied technical state indicate a meaningful "
            "probability of a trend reversal during the prediction horizon?"
        )
    ),
}

VOLATILITY_QUESTIONS: Mapping[str, Question] = {
    "volatility_regime": Choice(
        instructions=_guard(
            "Using only the supplied point-in-time volatility and risk "
            "features, estimate the most likely realized-volatility regime "
            "over the configured future trading horizon."
        ),
        criteria={
            "low": (
                "Future realized volatility is likely to be materially below "
                "its normal historical level."
            ),
            "normal": (
                "Future realized volatility is likely to remain around its "
                "normal historical level."
            ),
            "high": (
                "Future realized volatility is likely to be materially above "
                "its normal historical level."
            ),
            "extreme": (
                "Future realized volatility is likely to enter an unusually "
                "severe or tail-risk regime."
            ),
        },
    ),
    "volatility_spike": Noul(
        instructions=_guard(
            "Is realized volatility likely to exceed the asset's configured "
            "high-volatility threshold during the prediction horizon?"
        )
    ),
    "downside_tail": Noul(
        instructions=_guard(
            "Does the supplied state indicate elevated downside-tail risk over "
            "the prediction horizon?"
        )
    ),
    "risk_severity": Score(
        instructions=_guard(
            "Rate the expected severity of short-horizon market risk for this asset."
        ),
        criteria=list(SEVERITY_RUBRIC),
    ),
}

NEWS_QUESTIONS: Mapping[str, Question] = {
    "sentiment": Score(
        instructions=_guard(
            "Using only the supplied headlines, rate their expected "
            "directional financial impact on this asset over the configured "
            "prediction horizon."
        ),
        criteria=list(SENTIMENT_RUBRIC),
    ),
    "downside_event": Noul(
        instructions=_guard(
            "Do these headlines contain information that materially increases "
            "near-term downside risk for this asset?"
        )
    ),
    "volatility_event": Noul(
        instructions=_guard(
            "Are these headlines likely to cause unusually high price "
            "volatility for this asset during the configured horizon?"
        )
    ),
    "materiality": Score(
        instructions=_guard(
            "Rate how financially material these headlines are for the asset "
            "rather than merely being general or low-impact news."
        ),
        criteria=list(MATERIALITY_RUBRIC),
    ),
    "event_type": Choice(
        instructions=_guard(
            "Which event category best characterizes the most financially "
            "material information in these headlines?"
        ),
        criteria={
            "earnings": (
                "Earnings, revenue, guidance, profitability or financial results."
            ),
            "corporate": (
                "Management, M&A, restructuring, product or major corporate action."
            ),
            "regulatory": "Government, legal, regulatory or compliance event.",
            "macro": "Macroeconomic, interest-rate, inflation or broad economic event.",
            "market": (
                "Market positioning, analyst action or broad trading-related "
                "information."
            ),
            "operational": ("Production, supply-chain, outage or operational event."),
            "other": ("Material information that does not fit another category."),
            "none": "No materially relevant financial event is present.",
        },
    ),
}


# ---- validation ----------------------------------------------------------


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _check_probability(value: float, what: str) -> float:
    if not _finite(value) or not 0.0 - 1e-6 <= value <= 1.0 + 1e-6:
        raise InvalidResponse(f"{what} is not a probability: {value!r}")
    return min(max(float(value), 0.0), 1.0)


def read_choice(
    response: SystemOneResponse, name: str, labels: tuple[str, ...]
) -> tuple[list[float], float]:
    """Full distribution over ``labels``, validated and renormalised."""
    if name not in response.choices:
        raise InvalidResponse(f"missing choice answer {name!r}")
    answer = response.choices[name]
    missing = set(labels) - set(answer.probabilities)
    if missing:
        raise InvalidResponse(f"{name}: missing options {sorted(missing)}")
    values = [
        _check_probability(answer.probabilities[label], f"{name}.{label}")
        for label in labels
    ]
    total = sum(values)
    if not 0.9 <= total <= 1.1:
        raise InvalidResponse(f"{name}: probabilities sum to {total:.4f}")
    confidence = _check_probability(answer.confidence, f"{name}.confidence")
    return [value / total for value in values], confidence


def read_score(
    response: SystemOneResponse, name: str, levels: int
) -> tuple[float, list[float], float]:
    """Expected level, the whole distribution, and confidence."""
    if name not in response.scores:
        raise InvalidResponse(f"missing score answer {name!r}")
    answer = response.scores[name]
    values = [
        _check_probability(
            answer.probabilities.get(index, answer.probabilities.get(str(index), 0.0)),
            f"{name}.{index}",
        )
        for index in range(levels)
    ]
    total = sum(values)
    if not 0.9 <= total <= 1.1:
        raise InvalidResponse(f"{name}: probabilities sum to {total:.4f}")
    values = [value / total for value in values]
    if not _finite(answer.score):
        raise InvalidResponse(f"{name}: score is not finite")
    confidence = _check_probability(answer.confidence, f"{name}.confidence")
    # Expected level from the full distribution, not just the winner.
    expected = sum(index * value for index, value in enumerate(values))
    return expected, values, confidence


def read_noul(response: SystemOneResponse, name: str) -> float:
    if name not in response.nouls:
        raise InvalidResponse(f"missing noul answer {name!r}")
    return _check_probability(response.nouls[name].noul, name)


def _unit(expected: float, levels: int) -> float:
    """Rubric level -> [0, 1]."""
    return expected / (levels - 1) if levels > 1 else 0.0


def _signed(expected: float, levels: int) -> float:
    """Rubric level -> [-1, 1]; a 5-level scale maps 0,1,2,3,4 -> -1..+1."""
    return 2.0 * _unit(expected, levels) - 1.0


# ---- readers: response -> the persisted record ---------------------------


def read_technical(response: SystemOneResponse) -> dict[str, float]:
    probabilities, confidence = read_choice(response, "direction", DIRECTION_LABELS)
    down, flat, up = probabilities
    expected, distribution, trend_confidence = read_score(
        response, "trend_strength", len(TREND_RUBRIC)
    )
    return {
        "p_down": down,
        "p_flat": flat,
        "p_up": up,
        "direction_confidence": confidence,
        "trend_strength": _unit(expected, len(TREND_RUBRIC)),
        "trend_strength_confidence": trend_confidence,
        **{f"trend_p{i}": value for i, value in enumerate(distribution)},
        "p_reversal": read_noul(response, "reversal_risk"),
        # Raw directional signal. The specialist's head fits its scale in
        # return units on training data; it is NOT a percentage return.
        "direction_signal": up - down,
    }


def read_volatility(response: SystemOneResponse) -> dict[str, float]:
    probabilities, confidence = read_choice(
        response, "volatility_regime", REGIME_LABELS
    )
    low, normal, high, extreme = probabilities
    expected, distribution, severity_confidence = read_score(
        response, "risk_severity", len(SEVERITY_RUBRIC)
    )
    return {
        "p_vol_low": low,
        "p_vol_normal": normal,
        "p_vol_high": high,
        "p_vol_extreme": extreme,
        "volatility_confidence": confidence,
        "p_vol_spike": read_noul(response, "volatility_spike"),
        "p_downside_tail": read_noul(response, "downside_tail"),
        "risk_severity": _unit(expected, len(SEVERITY_RUBRIC)),
        "risk_severity_confidence": severity_confidence,
        **{f"severity_p{i}": value for i, value in enumerate(distribution)},
        # Probability-weighted regime index, 0 calm -> 3 crisis.
        "regime_level": (normal + 2 * high + 3 * extreme) / 3.0,
    }


def read_news(response: SystemOneResponse) -> dict[str, float]:
    expected, sentiment_distribution, sentiment_confidence = read_score(
        response, "sentiment", len(SENTIMENT_RUBRIC)
    )
    material, materiality_distribution, materiality_confidence = read_score(
        response, "materiality", len(MATERIALITY_RUBRIC)
    )
    event_probabilities, _ = read_choice(response, "event_type", EVENT_LABELS)
    sentiment = _signed(expected, len(SENTIMENT_RUBRIC))
    materiality = _unit(material, len(MATERIALITY_RUBRIC))
    return {
        "expected_sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        **{f"sentiment_p{i}": v for i, v in enumerate(sentiment_distribution)},
        "p_downside_event": read_noul(response, "downside_event"),
        "p_volatility_event": read_noul(response, "volatility_event"),
        "materiality": materiality,
        "materiality_confidence": materiality_confidence,
        **{f"materiality_p{i}": v for i, v in enumerate(materiality_distribution)},
        **{
            f"event_{label}": value
            for label, value in zip(EVENT_LABELS, event_probabilities, strict=True)
        },
        # Raw news signal; the head fits the scale empirically.
        "news_signal": sentiment * materiality * sentiment_confidence,
        "has_news": 1.0,
    }


def _columns(reader_keys: tuple[str, ...]) -> tuple[str, ...]:
    return reader_keys


TECHNICAL_COLUMNS: tuple[str, ...] = (
    "p_down",
    "p_flat",
    "p_up",
    "direction_confidence",
    "trend_strength",
    "trend_strength_confidence",
    *(f"trend_p{i}" for i in range(len(TREND_RUBRIC))),
    "p_reversal",
    "direction_signal",
)
VOLATILITY_COLUMNS: tuple[str, ...] = (
    "p_vol_low",
    "p_vol_normal",
    "p_vol_high",
    "p_vol_extreme",
    "volatility_confidence",
    "p_vol_spike",
    "p_downside_tail",
    "risk_severity",
    "risk_severity_confidence",
    *(f"severity_p{i}" for i in range(len(SEVERITY_RUBRIC))),
    "regime_level",
)
NEWS_COLUMNS: tuple[str, ...] = (
    "expected_sentiment",
    "sentiment_confidence",
    *(f"sentiment_p{i}" for i in range(len(SENTIMENT_RUBRIC))),
    "p_downside_event",
    "p_volatility_event",
    "materiality",
    "materiality_confidence",
    *(f"materiality_p{i}" for i in range(len(MATERIALITY_RUBRIC))),
    *(f"event_{label}" for label in EVENT_LABELS),
    "news_signal",
    "has_news",
    "news_count_normalized",
)

CHANNELS: dict[str, dict] = {
    "technical": {
        "questions": TECHNICAL_QUESTIONS,
        "columns": TECHNICAL_COLUMNS,
        "read": read_technical,
        "labels": DIRECTION_LABELS,
        "distribution": ("p_down", "p_flat", "p_up"),
        "signal": "direction_signal",
    },
    "volatility": {
        "questions": VOLATILITY_QUESTIONS,
        "columns": VOLATILITY_COLUMNS,
        "read": read_volatility,
        "labels": REGIME_LABELS,
        "distribution": ("p_vol_low", "p_vol_normal", "p_vol_high", "p_vol_extreme"),
        "signal": "regime_level",
    },
    "news": {
        "questions": NEWS_QUESTIONS,
        "columns": NEWS_COLUMNS,
        "read": read_news,
        "labels": SENTIMENT_RUBRIC,
        "distribution": tuple(f"sentiment_p{i}" for i in range(len(SENTIMENT_RUBRIC))),
        "signal": "news_signal",
    },
}


def neutral(channel: str) -> dict[str, float]:
    """The deterministic no-evidence record.

    Used only where there is genuinely nothing to judge - an asset-day with no
    headlines. ``has_news = 0`` is what lets the gate tell "no evidence" apart
    from "neutral evidence"; every probability field is left at zero rather
    than uniform, because a uniform distribution is a *claim* and this is the
    absence of one.
    """
    record = dict.fromkeys(CHANNELS[channel]["columns"], 0.0)
    record["inference_status"] = InferenceStatus.NO_INPUT
    return record
