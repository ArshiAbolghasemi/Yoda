"""The baselines and ablations the paper depends on (spec section 8).

Every arm runs on the *same* walk-forward splits, the same panel and the same
scoring intervals, so the comparison table is apples to apples by construction.

Four families:

``gate`` (headline)
    Tail-VoI vs Accuracy vs Attention vs EqualWeight, everything else fixed.
``system``
    Equal-Weight, Mean-Variance, Risk-Parity, static-CVaR (the DRO ball switched
    off) and a vol-targeted static policy.
``policy``
    the static risk policy vs the RL risk controller, same gate.
``news``
    ``no-news`` vs ``encoder`` vs ``llm_agent``, same gate and optimizer. The news
    channel is the thinnest one - FX headlines are keyword-matched - so the point
    is to let the gate decide whether ``z_news`` earns its place and to report it
    either way. ``encoder`` vs ``llm_agent`` doubles as the LLM leakage check.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from yoda.common.alignment import AlignedPanel, build_panel
from yoda.common.logger import logger
from yoda.common.types import DROCVaROptimizer
from yoda.config.settings import Config
from yoda.evaluation.report import Evaluation, evaluate
from yoda.optimizer.classical import EqualWeight, MeanVariance, RiskParity
from yoda.pipeline.tail_voli_risk import run_tail_voli_risk
from yoda.pipeline.tail_voli_risk_rl import run_tail_voli_risk_rl

FAMILIES: tuple[str, ...] = ("gate", "system", "policy", "news")


@dataclass(frozen=True)
class Arm:
    run_id: str
    label: str
    family: str
    gate: str = "tailvoi"
    rl: bool = False  # run the RL pipeline instead of the static one
    news_backend: str | None = None  # None = leave the configured backend alone
    optimizer: Callable[[], DROCVaROptimizer] | None = None
    overrides: dict = field(default_factory=dict)  # research-config replacements


def default_arms(gate: str = "tailvoi") -> list[Arm]:
    return [
        # -- headline gate ablation ------------------------------------------
        Arm("gate_tailvoi", "TailVoI", "gate", gate="tailvoi"),
        Arm("gate_accuracy", "Accuracy", "gate", gate="accuracy"),
        Arm("gate_attention", "Attention", "gate", gate="attention"),
        Arm("gate_equal", "EqualWeight", "gate", gate="equal_weight"),
        # -- classical system baselines ---------------------------------------
        Arm(
            "sys_equal_weight",
            "1/N",
            "system",
            gate="equal_weight",
            optimizer=EqualWeight,
        ),
        Arm(
            "sys_mean_variance",
            "Mean-Variance",
            "system",
            gate="equal_weight",
            optimizer=MeanVariance,
        ),
        Arm(
            "sys_risk_parity",
            "Risk-Parity",
            "system",
            gate="equal_weight",
            optimizer=RiskParity,
        ),
        Arm(
            "sys_static_cvar",
            "Static CVaR",
            "system",
            gate="equal_weight",
            overrides={"optimizer": {"dro": "none"}},
        ),
        Arm(
            "sys_vol_target",
            "Vol-target",
            "system",
            gate="equal_weight",
            overrides={"static_policy": {"vol_target": 0.02}},
        ),
        # -- static policy vs RL controller -------------------------------------
        Arm("tail_voli_risk", "Static policy", "policy", gate=gate),
        Arm("tail_voli_risk_rl", "RL policy", "policy", gate=gate, rl=True),
        # -- news channel -------------------------------------------------------
        Arm("news_none", "no-news", "news", gate=gate, news_backend="none"),
        Arm("news_encoder", "encoder", "news", gate=gate, news_backend="encoder"),
        Arm("news_llm", "llm_agent", "news", gate=gate, news_backend="llm_agent"),
    ]


def _apply(config: Config, arm: Arm) -> Config:
    research = config.research
    if arm.news_backend is not None:
        research = dataclasses.replace(
            research, news=dataclasses.replace(research.news, backend=arm.news_backend)
        )
        sources = tuple(
            name
            for name in research.tailvoi.sources
            if name != "news" or arm.news_backend != "none"
        )
        if "news" not in sources and arm.news_backend != "none":
            sources = (*sources, "news")
        research = dataclasses.replace(
            research, tailvoi=dataclasses.replace(research.tailvoi, sources=sources)
        )
    for section, changes in arm.overrides.items():
        research = dataclasses.replace(
            research,
            **{section: dataclasses.replace(getattr(research, section), **changes)},
        )
    return dataclasses.replace(config, research=research)


def _features(
    config: Config, panel: AlignedPanel, cache: dict
) -> dict[str, np.ndarray]:
    """Build each news backend's cube at most once and share it across arms."""
    backend = config.research.news.backend
    features = dict(panel.features)
    if backend == "none":
        return features
    if backend not in cache:
        from yoda.specialists.news import build_news_features

        cache[backend], _ = build_news_features(panel, config)
    features["news"] = cache[backend]
    return features


def run_experiments(
    config: Config,
    *,
    arms: list[Arm] | None = None,
    families: tuple[str, ...] = FAMILIES,
    panel: AlignedPanel | None = None,
    skip_failures: bool = True,
) -> Evaluation:
    """Run every selected arm and score them all on identical intervals."""
    panel = panel or build_panel(config)
    selected = [arm for arm in (arms or default_arms()) if arm.family in families]
    cache: dict[str, np.ndarray] = {}
    completed: list[str] = []

    for arm in selected:
        arm_config = _apply(config, arm)
        run = run_tail_voli_risk_rl if arm.rl else run_tail_voli_risk
        logger.info(
            "arm_start id=%s family=%s label=%s", arm.run_id, arm.family, arm.label
        )
        try:
            run(
                arm_config,
                run_id=arm.run_id,
                gate=arm.gate,
                panel=panel,
                features=_features(arm_config, panel, cache),
                optimizer=arm.optimizer() if arm.optimizer else None,
                label=arm.label,
                make_plots=False,
            )
            completed.append(arm.run_id)
        except Exception as error:  # noqa: BLE001 - one bad arm must not sink the sweep
            if not skip_failures:
                raise
            logger.warning("arm_failed id=%s error=%s", arm.run_id, error)

    if not completed:
        raise RuntimeError("Every experiment arm failed; see the log above")
    logger.info("experiments_done arms=%d", len(completed))
    return evaluate(config, completed, panel=panel, title="TailRiskFlow baselines")
