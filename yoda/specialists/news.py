"""The OpenJev news agent's specialist.

The agent reads only the point-in-time headlines for an asset-date and returns
a directional view with its tail and event assessment. This turns that view
into ``mu_hat_news``.

The mapping is fitted, never assigned. Expected sentiment, magnitude and
confidence are inputs to a ridge fit on the training window - there is no fixed
conversion such as "sentiment 0.5 means +5%".

An asset-date with no headlines returns a deterministic neutral record without
calling the model, with ``has_news = 0`` and maximal epistemic uncertainty, so
the gate can tell *no evidence* from *neutral evidence*.
"""

from __future__ import annotations

import numpy as np

from yoda.common.alignment import AlignedPanel
from yoda.config.settings import Config
from yoda.specialists.base import BaseSpecialist
from yoda.specialists.jev.features import build_jev_features


class NewsSpecialist(BaseSpecialist):
    """``mu_hat_news``: the headline view, in return units."""

    name = "news"


def build_news_features(
    panel: AlignedPanel, config: Config
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Materialise the news feature cube. OpenJev, or nothing."""
    backend = config.research.news.backend
    if backend == "none":
        raise ValueError("NEWS__BACKEND=none has no news features by definition")
    if backend != "jev":
        raise ValueError(f"Unknown news backend: {backend!r}; expected 'jev' or 'none'")
    return build_jev_features(panel, config, "news")
