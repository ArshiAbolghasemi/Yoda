"""Shared machinery for the three OpenJev agents.

Each agent returns a structured view (see :mod:`yoda.specialists.prompts`). The
specialist turns that view into the two things the rest of the stack consumes:

``z_i`` — the representation
    The view's numeric fields, standardised and padded to ``z_dim``. The
    probability-like quantities are exposed directly rather than projected: the
    gate is meant to weigh *these* variables, and a rotation into principal
    components would hide which of them carries the information. All three
    channels put the same six core variables first, which is what lets the gate
    compare them at all.

``mu_hat_i`` — the view in return units
    A ridge fit from the view's fields onto the realised forward return,
    estimated on the training window. This is the **empirically fitted scale**
    the prompts call for: ``expected_return_score`` is a score on [-1, 1], not
    a percentage return, and the mapping between them is learned rather than
    assigned by hand.

There is no model selection here. The agents are OpenJev; the head is a scale
fit, not a competing predictor.
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from yoda.common.logger import logger
from yoda.common.types import Specialist


def flatten(X: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """(T, N, F) -> ((T*N, F), (T, N)); a 2-D input passes through unchanged."""
    if X.ndim == 2:
        return X, (X.shape[0],)
    return X.reshape(-1, X.shape[-1]), X.shape[:-1]


class BaseSpecialist(Specialist):
    """Turns one agent's structured view into ``z_i`` and ``mu_hat_i``."""

    name = "base"

    def __init__(self, z_dim: int = 28, ridge_alpha: float = 10.0, seed: int = 7):
        self.z_dim = z_dim
        self.seed = seed
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.head = Ridge(alpha=ridge_alpha)
        self.fitted = False

    # ---- target transform hooks -----------------------------------------

    def _forward(self, y: np.ndarray) -> np.ndarray:
        """What the head regresses on. The forward return, by default."""
        return y

    def _inverse(self, prediction: np.ndarray) -> np.ndarray:
        """Map the head's output back into return units."""
        return prediction

    def _calibrate(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Hook: fit any retransformation correction on training residuals."""

    # ---- fit / encode / predict -----------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        features, _ = flatten(X)
        target = self._forward(np.asarray(y, dtype=np.float64).reshape(-1))
        usable = np.isfinite(target) & np.isfinite(features).any(axis=1)
        if usable.sum() < 2:
            raise ValueError(
                f"{self.name}: too few usable rows to fit ({usable.sum()})"
            )
        prepared = self.scaler.fit_transform(
            self.imputer.fit_transform(features[usable])
        )
        self.head.fit(prepared, target[usable])
        self._calibrate(self.head.predict(prepared), target[usable])
        self.fitted = True
        logger.info(
            "agent_scale_fitted name=%s rows=%d view_fields=%d z_dim=%d",
            self.name,
            int(usable.sum()),
            features.shape[1],
            self.z_dim,
        )

    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name} specialist is not fitted")

    def _prepare(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(features))

    def encode(self, X: np.ndarray) -> np.ndarray:
        """The view's numeric fields, standardised, padded to ``z_dim``.

        Channels declare different numbers of fields, so narrow ones are
        zero-padded up to the contract and wide ones truncated. The gate sums
        representations across channels and the RL observation has a fixed
        layout; both need this to be exact.
        """
        self._check()
        features, shape = flatten(X)
        z = self._prepare(features)
        if z.shape[-1] < self.z_dim:
            z = np.hstack([z, np.zeros((len(z), self.z_dim - z.shape[-1]))])
        elif z.shape[-1] > self.z_dim:
            z = z[:, : self.z_dim]
        return np.nan_to_num(z).reshape(*shape, self.z_dim)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check()
        features, shape = flatten(X)
        view = self._inverse(self.head.predict(self._prepare(features)))
        return np.nan_to_num(view.reshape(shape), nan=0.0)
