"""OpenJev technical specialist.

Estimates future directional behaviour from price and technical-analysis
information only. It never sees news, future returns, future prices, realised
future volatility, or any target label - its information set is built in
:mod:`yoda.specialists.jev.states` and contains nothing dated after ``t``.

The decision bundle is one ``choice`` (down / flat / up over the configured
horizon), one ``score`` (trend strength) and one ``noul`` (reversal risk).

``mu_hat_technical`` comes from the head, not from a hand-assigned conversion.
``P_up - P_down`` is a signal, not a percentage return; the head learns its
scale in return units on the training window.
"""

from __future__ import annotations

from yoda.specialists.base import BaseSpecialist


class TechnicalSpecialist(BaseSpecialist):
    """``mu_hat_technical``: forward-return view from price structure."""

    name = "technical"
    default_model = "mlp"
