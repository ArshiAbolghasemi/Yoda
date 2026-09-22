"""Directional specialist over the 60 causal technical indicators."""

from __future__ import annotations

from yoda.specialists.base import BaseSpecialist


class TechnicalSpecialist(BaseSpecialist):
    """``mu_hat_technical``: forward-return view from trend/momentum/volume."""

    name = "technical"
