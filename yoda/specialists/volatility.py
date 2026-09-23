"""OpenJev volatility specialist.

Estimates the future volatility and downside-risk regime. Its information set
is dispersion, drawdown and tail history - deliberately disjoint from the
technical channel's, so it cannot simply restate the directional prediction.

The decision bundle is one ``choice`` (low / normal / high / extreme), two
``noul`` questions (volatility spike, downside tail) and one ``score`` (risk
severity).

This channel informs portfolio risk and the Tail-VoI gate. It never enters
``mu`` and it never constructs weights: its view conditions the copula, and the
optimizer builds the book.

``predict`` returns a non-negative risk view. Where a log target is used, the
retransformation is corrected with Duan's smearing estimator - exponentiating a
log-space prediction gives the geometric mean, and understating volatility
understates CVaR, which is the one direction this system cannot be wrong in.
"""

from __future__ import annotations

import numpy as np

from yoda.specialists.base import BaseSpecialist

FLOOR = 1e-6


class VolatilitySpecialist(BaseSpecialist):
    name = "volatility"
    default_model = "mlp"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smearing = 1.0

    def _forward(self, y: np.ndarray) -> np.ndarray:
        """Fit in log volatility, where the target is roughly symmetric."""
        return np.log(np.abs(y) + FLOOR)

    def _calibrate(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Duan's smearing estimator for the log-to-level retransformation."""
        residual = target - prediction
        finite = np.isfinite(residual)
        if finite.any():
            self.smearing = float(np.mean(np.exp(residual[finite])))

    def _inverse(self, prediction: np.ndarray) -> np.ndarray:
        """Back to return units: exponentiate, de-bias, floor at zero."""
        level = np.exp(np.clip(prediction, -50, 10)) * self.smearing
        return np.maximum(level - FLOOR, 0.0)
