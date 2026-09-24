"""The OpenJev technical agent's specialist.

The agent reads price, momentum, trend, volume and market structure at date
``t`` and returns a directional view. This turns that view into
``mu_hat_technical``.

``expected_return_score`` is a score on [-1, 1], not a percentage return. The
prompt says so explicitly, and the head is what supplies the missing scale: a
ridge fit onto the realised forward return over the training window.
"""

from __future__ import annotations

from yoda.specialists.base import BaseSpecialist


class TechnicalSpecialist(BaseSpecialist):
    """``mu_hat_technical``: the agent's directional view, in return units."""

    name = "technical"
