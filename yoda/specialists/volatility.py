"""The OpenJev volatility and tail-risk agent's specialist.

The agent estimates the shape of the future return distribution - regime, tail
severity, skew, jump and liquidity risk - rather than its direction. This turns
that view into a non-negative risk forecast that conditions the copula.

It never enters ``mu``. Its directional field (``directional_bias``) is
deliberately secondary and reaches the gate through ``z``, not through the
expected-return path.

**Retransformation.** The target is ``log|r|``, because volatility is
right-skewed and a squared-error fit on raw ``|r|`` is dominated by a handful of
crisis days. Exponentiating a log-space prediction gives the geometric mean, so
Duan's smearing estimator corrects it non-parametrically - understating
volatility understates CVaR, which is the one direction this system cannot be
wrong in.
"""

from __future__ import annotations

import numpy as np

from yoda.specialists.base import BaseSpecialist

FLOOR = 1e-6  # keeps log(|r|) finite on exactly-zero returns


class VolatilitySpecialist(BaseSpecialist):
    name = "volatility"

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
