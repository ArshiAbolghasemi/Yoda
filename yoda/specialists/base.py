"""Shared specialist machinery.

Every specialist produces two things, and they come from two different
estimators on purpose:

``z_i`` — the representation
    The channel's probability vector, standardised and padded to ``z_dim``.
    The gate sums representations across channels and the RL observation has a
    fixed layout, so the width is part of the contract.

``mu_hat_i`` — the view
    A **per-specialist head** fitted on the channel's full feature matrix. This
    is where the empirical scale lives: a directional signal like
    ``P_up - P_down`` is not a percentage return, and the mapping from one to
    the other is learned on the training window rather than assigned by hand.

Subclasses choose their head with ``model=`` and may transform the target
through ``_forward`` / ``_inverse``. Everything else — preprocessing, shape
handling, the no-lookahead discipline — is shared.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from yoda.common.logger import logger
from yoda.common.types import Specialist

MODELS: tuple[str, ...] = ("mlp", "ridge", "gbm", "har", "pcr")


def flatten(X: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """(T, N, F) -> ((T*N, F), (T, N)); a 2-D input passes through unchanged."""
    if X.ndim == 2:
        return X, (X.shape[0],)
    return X.reshape(-1, X.shape[-1]), X.shape[:-1]


def _log_positive(X: np.ndarray) -> np.ndarray:
    """log1p of the non-negative part - the HAR model lives in log volatility."""
    return np.log1p(np.clip(X, 0.0, None))


def make_head(model: str, *, ridge_alpha: float, seed: int, z_dim: int):
    """The estimator mapping features to a view.

    ``mlp``
        A small dense network. The measured winner on this panel for both the
        technical and volatility channels.
    ``ridge``
        Regularised linear on the scaled features.
    ``gbm``
        Boosted trees - best in Gu, Kelly & Xiu (2020) on a far larger panel.
    ``har``
        Corsi (2009) in spirit: linear in *log* multi-horizon volatility.
    ``pcr``
        Ridge on PCA components, kept so the head choice stays ablatable.
    """
    if model == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model == "gbm":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
    if model == "har":
        return Pipeline(
            [
                ("log", FunctionTransformer(_log_positive)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha)),
            ]
        )
    if model == "ridge":
        return Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))]
        )
    if model == "pcr":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("project", PCA(n_components=z_dim, random_state=seed)),
                ("ridge", Ridge(alpha=ridge_alpha)),
            ]
        )
    raise ValueError(f"Unknown specialist model: {model!r}; expected one of {MODELS}")


class BaseSpecialist(Specialist):
    """Shared representation encoder plus a swappable per-specialist head."""

    name = "base"
    default_model = "mlp"

    def __init__(
        self,
        z_dim: int = 16,
        ridge_alpha: float = 10.0,
        seed: int = 7,
        model: str | None = None,
    ):
        self.z_dim = z_dim
        self.model = model or self.default_model
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.head = make_head(
            self.model, ridge_alpha=ridge_alpha, seed=seed, z_dim=z_dim
        )
        self.fitted = False

    # ---- target transform hooks -----------------------------------------

    def _forward(self, y: np.ndarray) -> np.ndarray:
        """What the head regresses on. Directional by default."""
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
        imputed = self.imputer.fit_transform(features[usable])
        self.scaler.fit(imputed)
        self.head.fit(imputed, target[usable])
        self._calibrate(self.head.predict(imputed), target[usable])
        self.fitted = True
        logger.info(
            "specialist_fitted name=%s model=%s rows=%d features=%d z_dim=%d",
            self.name,
            self.model,
            int(usable.sum()),
            features.shape[1],
            self.z_dim,
        )

    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name} specialist is not fitted")

    def encode(self, X: np.ndarray) -> np.ndarray:
        """The channel's own feature vector, standardised, padded to ``z_dim``.

        The probability vector is exposed directly rather than projected: the
        gate is meant to weigh *these* quantities, and a rotation into principal
        components would hide which of them carries the information. Channels
        differ in width, so narrow ones are zero-padded up to the contract and
        wide ones truncated.
        """
        self._check()
        features, shape = flatten(X)
        z = self.scaler.transform(self.imputer.transform(features))
        if z.shape[-1] < self.z_dim:
            z = np.hstack([z, np.zeros((len(z), self.z_dim - z.shape[-1]))])
        elif z.shape[-1] > self.z_dim:
            z = z[:, : self.z_dim]
        return np.nan_to_num(z).reshape(*shape, self.z_dim)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check()
        features, shape = flatten(X)
        view = self._inverse(self.head.predict(self.imputer.transform(features)))
        return np.nan_to_num(view.reshape(shape), nan=0.0)
