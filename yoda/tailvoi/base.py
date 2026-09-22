"""Gate interface and the fusion rule every gate shares.

A gate answers one question: *how much should each information source count?*
The answer is used at the stage each source actually enters the stack, which is
not the same stage for all of them:

``technical``, ``news``
    Directional views. Their gate weights mix ``mu_hat_i`` into ``fused_mu``.
``volatility``
    A risk view. It never touches ``mu``; its gate weight blends the
    specialist's ``sigma_hat`` against the copula's unconditional scale, i.e. it
    controls how strongly the scenario set is conditioned on the current regime.

Keeping that split honest is what makes the counterfactual targets mean
something: ablating ``volatility`` has to perturb the scenarios, not the mean.
"""

from __future__ import annotations

import numpy as np

from yoda.common.types import Gate, GateOutput, SpecialistOutput

MU_SOURCES: tuple[str, ...] = ("technical", "news")
CONDITIONING_SOURCES: tuple[str, ...] = ("volatility",)


def available_sources(spec: SpecialistOutput) -> tuple[str, ...]:
    """Sources actually present in this run (the ``no-news`` arm drops one)."""
    return tuple(name for name in spec.z if name in spec.z)


def summarise(spec: SpecialistOutput, sources: tuple[str, ...]) -> np.ndarray:
    """Fixed-width state summary that the learned gates take as input.

    Cross-sectional statistics only, so the width is independent of how many
    assets are investable on a given day.
    """
    parts: list[np.ndarray] = []
    for name in sources:
        z = np.atleast_2d(np.asarray(spec.z[name], dtype=np.float64))
        mu = np.asarray(spec.mu_hat[name], dtype=np.float64).reshape(-1)
        parts.append(
            np.concatenate(
                [
                    z.mean(axis=0),
                    [float(np.abs(mu).mean()), float(mu.std()), float(z.std())],
                ]
            )
        )
    return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / max(temperature, 1e-8)
    shifted = np.exp(scaled - scaled.max())
    return shifted / shifted.sum()


def fuse(
    spec: SpecialistOutput,
    weights: dict[str, float],
    delta_hat: dict[str, float] | None = None,
) -> GateOutput:
    """Apply gate weights at the stage each source belongs to."""
    sources = tuple(spec.z)
    directional = [name for name in sources if name in MU_SOURCES]
    mu_mass = sum(weights.get(name, 0.0) for name in directional)
    n_assets = len(np.asarray(spec.mu_hat[sources[0]]).reshape(-1))

    fused_mu = np.zeros(n_assets, dtype=np.float64)
    for name in directional:
        share = weights.get(name, 0.0) / mu_mass if mu_mass > 0 else 0.0
        fused_mu += share * np.asarray(spec.mu_hat[name], dtype=np.float64).reshape(-1)

    z_dim = np.atleast_2d(spec.z[sources[0]]).shape[1]
    fused_z = np.zeros((n_assets, z_dim), dtype=np.float64)
    for name in sources:
        fused_z += weights.get(name, 0.0) * np.atleast_2d(
            np.asarray(spec.z[name], dtype=np.float64)
        )

    return GateOutput(
        g={name: float(weights.get(name, 0.0)) for name in sources},
        fused_mu=np.nan_to_num(fused_mu),
        fused_z=np.nan_to_num(fused_z),
        delta_hat=dict(delta_hat or {}),
    )


def conditioning_strength(gate: GateOutput) -> float:
    """How strongly the copula should be conditioned, in [0, 1].

    The volatility gate weight, renormalised against the directional mass: a gate
    that has decided the vol channel carries no Tail-VoI falls back to the
    unconditional scenario set.
    """
    total = sum(gate.g.values())
    if total <= 0:
        return 0.0
    share = sum(gate.g.get(name, 0.0) for name in CONDITIONING_SOURCES) / total
    # An even gate over k sources means "condition normally" -> 1.0.
    return float(np.clip(share * len(gate.g), 0.0, 1.0))


__all__ = [
    "CONDITIONING_SOURCES",
    "MU_SOURCES",
    "Gate",
    "GateOutput",
    "SpecialistOutput",
    "available_sources",
    "conditioning_strength",
    "fuse",
    "softmax",
    "summarise",
]
