"""The learned Tail-VoI gate: ``G_phi(z) -> Delta_hat``, then a softmax over it.

The counterfactual generator is exact but far too expensive to run inside a
backtest. This is its amortisation: a small regressor trained offline on
``(state summary, Delta)`` pairs that predicts, from the current state alone,
how much tail risk each information source is worth right now.

Unfitted it degrades to an even gate rather than failing, so the stack can be
assembled before the targets exist.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yoda.common.logger import logger
from yoda.common.types import Gate, GateOutput, SpecialistOutput
from yoda.config.research import TailVoIConfig
from yoda.tailvoi.base import fuse, softmax, summarise
from yoda.tailvoi.counterfactual import CounterfactualTargets


class TailVoIGate(Gate):
    """Gate the information sources by their predicted counterfactual tail value."""

    name = "tailvoi"

    def __init__(self, config: TailVoIConfig | None = None):
        self.config = config or TailVoIConfig()
        self.sources: tuple[str, ...] = self.config.sources
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=tuple(self.config.hidden),
                        random_state=self.config.seed,
                        max_iter=2000,
                        early_stopping=False,
                    ),
                ),
            ]
        )
        self.scale = 1.0
        self.fitted = False

    def fit(self, targets: CounterfactualTargets) -> None:
        self.sources = targets.sources
        deltas = targets.deltas
        # Deltas are tail-risk differences (1e-3 scale); normalise so the softmax
        # temperature means the same thing regardless of the risk units.
        self.scale = float(np.std(deltas)) or 1.0
        self.model.fit(targets.summaries, deltas / self.scale)
        self.fitted = True
        predicted = self.model.predict(targets.summaries)
        logger.info(
            "tailvoi_gate_fitted states=%d sources=%s train_r=%.3f",
            len(deltas),
            ",".join(self.sources),
            float(np.corrcoef(predicted.ravel(), (deltas / self.scale).ravel())[0, 1]),
        )

    def weights(self, delta_hat: np.ndarray) -> np.ndarray:
        if self.config.mode == "threshold":
            keep = (delta_hat > self.config.threshold).astype(float)
            return (
                keep / keep.sum()
                if keep.sum()
                else np.full_like(delta_hat, 1 / len(delta_hat))
            )
        return softmax(delta_hat, self.config.temperature)

    def gate(self, spec: SpecialistOutput) -> GateOutput:
        sources = tuple(name for name in self.sources if name in spec.z)
        if not self.fitted:
            logger.warning("tailvoi_gate_unfitted falling back to an even gate")
            even = dict.fromkeys(sources, 1.0 / len(sources))
            return fuse(spec, even)
        delta_hat = np.asarray(
            self.model.predict(summarise(spec, sources).reshape(1, -1))
        ).reshape(-1)[: len(sources)]
        g = self.weights(delta_hat)
        return fuse(
            spec,
            dict(zip(sources, g, strict=True)),
            delta_hat=dict(zip(sources, delta_hat * self.scale, strict=True)),
        )
