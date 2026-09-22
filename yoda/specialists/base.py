"""Shared specialist machinery: impute-with-mask -> scale -> PCA -> ridge.

Every specialist turns a per-asset feature cube into a representation ``z_i``
and a view ``mu_hat_i``. The preprocessing is fitted on the training window
only, so nothing about the evaluation window leaks into the encoder.

FX rows have eight volume-derived indicators that are NaN *by design*. They are
never imputed to zero: the imputer appends a missing-indicator column so the
downstream model can tell "no volume data" apart from "volume was flat".
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yoda.common.logger import logger
from yoda.common.types import Specialist


def flatten(X: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """(T, N, F) -> ((T*N, F), (T, N)); a 2-D input passes through unchanged."""
    if X.ndim == 2:
        return X, (X.shape[0],)
    return X.reshape(-1, X.shape[-1]), X.shape[:-1]


class BaseSpecialist(Specialist):
    """Supervised encoder shared by the technical, volatility and news channels."""

    name = "base"

    def __init__(self, z_dim: int = 8, ridge_alpha: float = 10.0, seed: int = 7):
        self.z_dim = z_dim
        self.encoder = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("project", PCA(n_components=z_dim, random_state=seed)),
            ]
        )
        self.head = Ridge(alpha=ridge_alpha)
        self.fitted = False

    def _target(self, y: np.ndarray) -> np.ndarray:
        """Hook: what the head regresses on. Directional by default."""
        return y

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        features, _ = flatten(X)
        target = self._target(np.asarray(y, dtype=np.float64)).reshape(-1)
        usable = np.isfinite(target) & np.isfinite(features).any(axis=1)
        if usable.sum() < self.z_dim + 1:
            raise ValueError(
                f"{self.name}: too few usable rows to fit ({usable.sum()})"
            )
        components = min(self.z_dim, int(usable.sum()) - 1, features.shape[1])
        self.encoder.set_params(project__n_components=components)
        z = self.encoder.fit_transform(features[usable])
        self.head.fit(z, target[usable])
        self.fitted = True
        logger.info(
            "specialist_fitted name=%s rows=%d features=%d z_dim=%d",
            self.name,
            int(usable.sum()),
            features.shape[1],
            z.shape[1],
        )

    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name} specialist is not fitted")

    def encode(self, X: np.ndarray) -> np.ndarray:
        self._check()
        features, shape = flatten(X)
        z = self.encoder.transform(features)
        return z.reshape(*shape, z.shape[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check()
        features, shape = flatten(X)
        view = self.head.predict(self.encoder.transform(features))
        return np.nan_to_num(view.reshape(shape), nan=0.0)
