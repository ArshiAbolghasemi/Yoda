"""OpenJev news specialist.

Extracts directional sentiment, event severity, volatility relevance and
downside-tail relevance from point-in-time headlines. It sees only the
headlines the dataset pipeline already aligned to that trading date, and never
future market reactions or future returns.

The decision bundle is two ``score`` questions (sentiment, materiality), two
``noul`` questions (downside event, volatility event) and one ``choice`` (event
type). Nothing asks for an explanation or a recommendation.

An asset-day with no headlines returns a deterministic neutral record without
calling the model, and carries ``has_news = 0`` so the gate can tell "no
evidence" apart from "neutral evidence".

``mu_hat_news`` comes from the head. Expected sentiment, materiality and
confidence are inputs to an empirically fitted mapping - never a fixed
conversion like "sentiment 0.5 means +5%".
"""

from __future__ import annotations

import numpy as np

from yoda.common.alignment import AlignedPanel
from yoda.config.settings import Config
from yoda.specialists.base import BaseSpecialist
from yoda.specialists.jev.features import build_jev_features


class NewsSpecialist(BaseSpecialist):
    """``mu_hat_news``: the headline view, scaled by materiality and confidence."""

    name = "news"


def build_news_features(
    panel: AlignedPanel, config: Config
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Materialise the news feature cube for the configured backend."""
    backend = config.research.news.backend
    if backend == "none":
        raise ValueError("NEWS__BACKEND=none has no news features by definition")
    if backend != "jev":
        raise ValueError(f"Unknown news backend: {backend!r}; expected 'jev' or 'none'")
    return build_jev_features(panel, config, "news")
