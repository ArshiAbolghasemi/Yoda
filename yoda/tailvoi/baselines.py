"""Baseline gates - the ablation the Tail-VoI claim rests on.

All three answer "how much should each source count?" through the identical
:class:`~yoda.common.types.Gate` interface, so swapping one in changes nothing
else in the stack. If Tail-VoI does not beat these, it does not work.

``EqualWeightGate``
    No gating at all. The floor.
``AccuracyGate``
    Importance proportional to predictive skill (rank IC on the training
    window). Rewards being *right*, with no notion of what a source is worth in
    the tail.
``AttentionGate``
    A learned attention over the source summaries, trained on return prediction
    error. Learned, but supervised by the mean - not by risk.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from torch import nn

from yoda.common.logger import logger
from yoda.common.types import Gate, GateOutput, SpecialistOutput
from yoda.tailvoi.base import MU_SOURCES, fuse, softmax, summarise


class EqualWeightGate(Gate):
    """Uniform weights over whichever sources are present."""

    name = "equal_weight"

    def gate(self, spec: SpecialistOutput) -> GateOutput:
        sources = tuple(spec.z)
        return fuse(spec, dict.fromkeys(sources, 1.0 / len(sources)))


class AccuracyGate(Gate):
    """Importance = predictive skill, measured once on the training window."""

    name = "accuracy"

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.skill: dict[str, float] = {}

    def fit(self, mu_hat: dict[str, np.ndarray], realized: np.ndarray) -> None:
        """``mu_hat[source]`` and ``realized`` are (T, N) over the train window."""
        target = np.asarray(realized, dtype=np.float64)
        for name, view in mu_hat.items():
            # The volatility channel forecasts magnitude, so score it against |r|.
            truth = np.abs(target) if name not in MU_SOURCES else target
            left, right = np.asarray(view).ravel(), truth.ravel()
            usable = np.isfinite(left) & np.isfinite(right)
            self.skill[name] = (
                float(stats.spearmanr(left[usable], right[usable]).statistic)
                if usable.sum() > 2
                else 0.0
            )
        logger.info("accuracy_gate_fitted skill=%s", self.skill)

    def gate(self, spec: SpecialistOutput) -> GateOutput:
        sources = tuple(spec.z)
        scores = np.array(
            [np.nan_to_num(self.skill.get(name, 0.0)) for name in sources]
        )
        g = softmax(scores, self.temperature)
        return fuse(spec, dict(zip(sources, g, strict=True)))


class _Attention(nn.Module):
    """One shared scorer applied to each source's summary chunk."""

    def __init__(self, width: int, hidden: int = 16):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        # chunks: (batch, sources, width) -> attention weights (batch, sources)
        return torch.softmax(self.score(chunks).squeeze(-1), dim=-1)


class AttentionGate(Gate):
    """Learned attention over sources, supervised by forward-return error."""

    name = "attention"

    def __init__(self, seed: int = 13, epochs: int = 300, lr: float = 1e-2):
        self.seed, self.epochs, self.lr = seed, epochs, lr
        self.sources: tuple[str, ...] = ()
        self.model: _Attention | None = None

    def fit(
        self,
        summaries: np.ndarray,
        mu_stack: np.ndarray,
        realized: np.ndarray,
        sources: tuple[str, ...],
    ) -> None:
        """``summaries`` (M, K*width), ``mu_stack`` (M, K, N), ``realized`` (M, N)."""
        torch.manual_seed(self.seed)
        self.sources = sources
        width = summaries.shape[1] // len(sources)
        chunks = torch.tensor(
            summaries.reshape(len(summaries), len(sources), width), dtype=torch.float32
        )
        views = torch.tensor(np.nan_to_num(mu_stack), dtype=torch.float32)
        truth = torch.tensor(np.nan_to_num(realized), dtype=torch.float32)
        directional = torch.tensor(
            [1.0 if name in MU_SOURCES else 0.0 for name in sources]
        )

        self.model = _Attention(width)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            optimiser.zero_grad()
            g = self.model(chunks)
            mass = (g * directional).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            fused = ((g * directional).unsqueeze(-1) * views).sum(dim=1) / mass
            loss = nn.functional.mse_loss(fused, truth)
            loss.backward()
            optimiser.step()
        logger.info(
            "attention_gate_fitted states=%d loss=%.3e", len(summaries), loss.item()
        )

    def gate(self, spec: SpecialistOutput) -> GateOutput:
        sources = tuple(name for name in self.sources if name in spec.z) or tuple(
            spec.z
        )
        if self.model is None:
            return EqualWeightGate().gate(spec)
        summary = summarise(spec, sources)
        width = len(summary) // len(sources)
        chunks = torch.tensor(
            summary.reshape(1, len(sources), width), dtype=torch.float32
        )
        with torch.no_grad():
            g = self.model(chunks).numpy().reshape(-1)
        return fuse(spec, dict(zip(sources, g, strict=True)))
