"""Risk specialist: ATR, band widths and realized volatility.

Realized volatility only - the dataset carries no implied-vol surface. The head
regresses the *magnitude* of the forward return, so ``predict`` returns
``sigma_hat`` rather than a directional view; it conditions the copula, it does
not enter ``mu``.
"""

from __future__ import annotations

import numpy as np

from yoda.specialists.base import BaseSpecialist


class VolatilitySpecialist(BaseSpecialist):
    name = "volatility"

    def _target(self, y: np.ndarray) -> np.ndarray:
        return np.abs(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward volatility view, floored at zero."""
        return np.maximum(super().predict(X), 0.0)
