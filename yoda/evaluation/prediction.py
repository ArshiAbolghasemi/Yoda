"""Specialist-level prediction quality, scored before any portfolio is formed.

A probabilistic specialist has to be judged as a *forecaster* first: a channel
that is miscalibrated will still produce portfolio numbers, and those numbers
will be noise dressed as a result. This module scores the cached probability
distributions against realised outcomes, with no reference to weights or
returns.

Discrimination and calibration are reported separately on purpose. Accuracy,
macro-F1, balanced accuracy and ROC-AUC say whether the ranking is right; Brier
score and expected calibration error say whether the *numbers* can be believed.
A channel can be useful on one and useless on the other, and the Tail-VoI gate
cares about both.

Labels are constructed causally: every threshold comes from data that was
available before the outcome it labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from yoda.common.alignment import AlignedPanel
from yoda.common.logger import logger

DIRECTION_LABELS: tuple[str, ...] = ("down", "flat", "up")
REGIME_LABELS: tuple[str, ...] = ("low", "normal", "high", "extreme")


# ---- scoring primitives --------------------------------------------------


def brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error of the whole distribution."""
    if probabilities.size == 0:
        return float("nan")
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10
) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins.

    Zero means "when it says 70% it is right 70% of the time"; the bound is the
    weighted mean absolute gap, so a value of 0.1 is a ten-point overstatement.
    """
    if probabilities.size == 0:
        return float("nan")
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == outcomes).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error, total = 0.0, len(outcomes)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if not inside.any():
            continue
        error += (
            inside.sum()
            / total
            * abs(correct[inside].mean() - confidence[inside].mean())
        )
    return float(error)


def _auc(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Macro one-vs-rest ROC-AUC; NaN when a class is absent."""
    present = np.unique(outcomes)
    if len(present) < 2:
        return float("nan")
    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(outcomes, probabilities[:, 1]))
        return float(
            roc_auc_score(
                outcomes,
                probabilities / probabilities.sum(axis=1, keepdims=True),
                multi_class="ovr",
                average="macro",
                labels=list(range(probabilities.shape[1])),
            )
        )
    except ValueError:
        return float("nan")


def classification_report(
    probabilities: np.ndarray, outcomes: np.ndarray
) -> dict[str, float]:
    """Discrimination + calibration for one probabilistic channel."""
    if len(outcomes) == 0:
        return dict.fromkeys(
            (
                "n",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "brier",
                "ece",
            ),
            float("nan"),
        )
    predicted = probabilities.argmax(axis=1)
    return {
        "n": float(len(outcomes)),
        "accuracy": float(accuracy_score(outcomes, predicted)),
        "macro_f1": float(
            f1_score(outcomes, predicted, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(outcomes, predicted)),
        "roc_auc": _auc(probabilities, outcomes),
        "brier": brier(probabilities, outcomes),
        "ece": expected_calibration_error(probabilities, outcomes),
    }


# ---- causal label construction ------------------------------------------


def direction_labels(panel: AlignedPanel, flat_band: float = 0.25) -> np.ndarray:
    """down / flat / up, with the flat band scaled by trailing volatility.

    A fixed return threshold would label FX as "flat" almost always and BTC
    almost never. Scaling by each asset's own trailing volatility makes the
    three classes comparable across a mixed panel.
    """
    trailing = _trailing_volatility(panel)
    band = flat_band * trailing
    target = panel.targets
    labels = np.full(target.shape, 1, dtype=np.int64)  # flat
    labels[target < -band] = 0
    labels[target > band] = 2
    return labels


def regime_labels(panel: AlignedPanel, window: int = 252) -> np.ndarray:
    """low / normal / high / extreme from trailing quantiles of |r|.

    Thresholds come from the *previous* ``window`` observations only, so the
    label at ``t`` never depends on anything after ``t``.
    """
    realised = pd.DataFrame(np.abs(panel.targets))
    lower = realised.shift(1).rolling(window, min_periods=60).quantile(0.25).to_numpy()
    middle = realised.shift(1).rolling(window, min_periods=60).quantile(0.75).to_numpy()
    upper = realised.shift(1).rolling(window, min_periods=60).quantile(0.95).to_numpy()
    magnitude = np.abs(panel.targets)
    labels = np.full(magnitude.shape, -1, dtype=np.int64)
    known = np.isfinite(lower) & np.isfinite(middle) & np.isfinite(upper)
    labels[known & (magnitude <= lower)] = 0
    labels[known & (magnitude > lower) & (magnitude <= middle)] = 1
    labels[known & (magnitude > middle) & (magnitude <= upper)] = 2
    labels[known & (magnitude > upper)] = 3
    return labels


def _trailing_volatility(panel: AlignedPanel, window: int = 60) -> np.ndarray:
    returns = pd.DataFrame(panel.returns)
    volatility = returns.shift(1).rolling(window, min_periods=20).std(ddof=0).to_numpy()
    fallback = np.nanmedian(volatility)
    return np.where(np.isfinite(volatility), volatility, fallback)


# ---- channel evaluation --------------------------------------------------


def _aligned(
    table: pd.DataFrame,
    panel: AlignedPanel,
    columns: tuple[str, ...],
    labels: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Join a cached probability table onto the panel grid for ``rows``."""
    lookup = table.copy()
    lookup["date"] = pd.to_datetime(lookup["date"])
    lookup = lookup.drop_duplicates(["asset", "date"], keep="last").set_index(
        ["asset", "date"]
    )
    probabilities, outcomes = [], []
    assets = np.asarray(panel.assets)
    for row in rows:
        date = panel.dates[row]
        for index, asset in enumerate(assets):
            label = labels[row, index]
            if label < 0 or not panel.available[row, index]:
                continue
            try:
                record = lookup.loc[(asset, date)]
            except KeyError:
                continue
            probabilities.append([float(record[column]) for column in columns])
            outcomes.append(int(label))
    if not outcomes:
        return np.empty((0, len(columns))), np.empty(0, dtype=np.int64)
    matrix = np.asarray(probabilities, dtype=np.float64)
    return matrix, np.asarray(outcomes, dtype=np.int64)


def _confidence_weighted(signed: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """A signed score plus a confidence -> a three-class distribution.

    ``confidence = 0`` collapses to uniform: the agent said it does not know.
    ``confidence = 1`` puts the whole mass on the side the score points to.
    """
    magnitude = np.clip(np.abs(signed), 0.0, 1.0) * np.clip(confidence, 0.0, 1.0)
    flat = 1.0 - magnitude
    down = np.where(signed < 0, magnitude, 0.0)
    up = np.where(signed > 0, magnitude, 0.0)
    stacked = np.column_stack([down, flat, up])
    return stacked / stacked.sum(axis=1, keepdims=True)


def _signed_to_classes(table, panel, column, labels, rows):
    raw, outcomes = _aligned(table, panel, (column, "confidence"), labels, rows)
    if not raw.size:
        return raw.reshape(0, 3), outcomes
    return _confidence_weighted(raw[:, 0], raw[:, 1]), outcomes


def _score_to_regimes(table, panel, column, labels, rows):
    """A [0,1] magnitude -> four ordered volatility-regime buckets."""
    raw, outcomes = _aligned(table, panel, (column, "confidence"), labels, rows)
    if not raw.size:
        return raw.reshape(0, 4), outcomes
    score = np.clip(raw[:, 0], 0.0, 1.0)
    edges = np.array([0.125, 0.375, 0.625, 0.875])
    distance = np.abs(score[:, None] - edges[None, :])
    weight = np.exp(-distance / 0.15)
    return weight / weight.sum(axis=1, keepdims=True), outcomes


def evaluate_technical(
    table: pd.DataFrame, panel: AlignedPanel, rows: np.ndarray
) -> dict[str, float]:
    labels = direction_labels(panel)
    # The agents report a signed score, not a distribution, so it is bucketed
    # into down/flat/up before scoring. Magnitude and confidence widen the
    # implied flat band: a low-confidence score is a weaker directional claim.
    probabilities, outcomes = _signed_to_classes(
        table, panel, "expected_return_score", labels, rows
    )
    return classification_report(probabilities, outcomes)


def evaluate_volatility(
    table: pd.DataFrame, panel: AlignedPanel, rows: np.ndarray
) -> dict[str, float]:
    labels = regime_labels(panel)
    probabilities, outcomes = _score_to_regimes(
        table, panel, "expected_magnitude", labels, rows
    )
    return classification_report(probabilities, outcomes)


def evaluate_news(
    table: pd.DataFrame, panel: AlignedPanel, rows: np.ndarray
) -> dict[str, float]:
    """Direction metrics plus the sentiment/return relationship.

    There are no sentiment ground-truth labels in this dataset, so the honest
    test is whether sentiment moves with the forward return at all.
    """
    labels = direction_labels(panel)
    # The sentiment rubric has five levels; fold to down/flat/up so it can be
    # scored against the same directional label the technical channel uses.
    probabilities, outcomes = _signed_to_classes(
        table, panel, "expected_return_score", labels, rows
    )
    report = classification_report(probabilities, outcomes)
    if len(outcomes):
        sentiment = probabilities[:, 2] - probabilities[:, 0]
        direction = np.sign(outcomes - 1.0)
        report["sentiment_return_corr"] = (
            float(np.corrcoef(sentiment, direction)[0, 1])
            if sentiment.std() > 0
            else float("nan")
        )
    else:
        report["sentiment_return_corr"] = float("nan")
    return report


EVALUATORS = {
    "technical": evaluate_technical,
    "volatility": evaluate_volatility,
    "news": evaluate_news,
}


def evaluate_specialists(
    tables: dict[str, pd.DataFrame], panel: AlignedPanel, rows: np.ndarray
) -> pd.DataFrame:
    """One metric row per channel, scored on ``rows`` (normally the test split)."""
    report = {
        channel: EVALUATORS[channel](table, panel, rows)
        for channel, table in tables.items()
        if channel in EVALUATORS
    }
    frame = pd.DataFrame(report).T
    logger.info("specialist_prediction_scored channels=%s", ",".join(report))
    return frame
