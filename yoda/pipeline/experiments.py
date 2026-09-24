"""The baseline, backend and ablation matrix.

Every arm runs on the same panel, the same walk-forward splits, the same
transaction costs and the same scoring intervals, so a difference between two
rows is a difference in the thing being varied and nothing else.

Families
--------
``gate``
    Tail-VoI vs Accuracy vs Attention vs EqualWeight, everything else pinned.
    The headline: does gating on *tail value* beat gating on accuracy, on
    learned attention, and on nothing?
``optimizer``
    The DRO robustification itself: Wasserstein, moment-based, or off. What is
    the distributional robustness actually worth?
``policy``
    The static risk rule vs the SAC controller - the only thing that varies
    inside the CIO.
``openjev``
    Conventional specialists versus OpenJev probabilistic specialists, one
    channel at a time and in every combination. **The primary research
    question.**
``sources``
    Agent-removal. Each arm *deletes* a channel from the Tail-VoI input rather
    than zeroing it, so the gate renormalises over what remains and the measured
    effect is that information's marginal contribution.
``horizon``
    The same stack at 1, 5, 10 and 20 trading-day prediction/rebalance horizons.

Nothing here assumes OpenJev helps. The matrix is built so a negative result is
as readable as a positive one, and every arm is persisted either way.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from yoda.backtest.artifacts import load_run
from yoda.common.alignment import AlignedPanel, build_panel
from yoda.common.logger import logger
from yoda.common.types import SOURCES, DROCVaROptimizer
from yoda.config.settings import Config
from yoda.evaluation.intervals import resolve, slice_ledger
from yoda.evaluation.report import Evaluation, evaluate
from yoda.pipeline.tail_voli_risk import run_tail_voli_risk
from yoda.pipeline.tail_voli_risk_rl import run_tail_voli_risk_rl
from yoda.stack import resolve_features

FAMILIES: tuple[str, ...] = (
    "gate",
    "optimizer",
    "policy",
    "openjev",
    "sources",
    "horizon",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "experiment",
    "tech_backend",
    "vol_backend",
    "news_backend",
    "return",
    "sharpe",
    "sortino",
    "calmar",
    "max_dd",
    "cvar",
    "turnover",
)


@dataclass(frozen=True)
class Arm:
    run_id: str
    label: str
    family: str
    gate: str | None = None  # None = the configured TAILVOI__GATE
    rl: bool = False
    sources: tuple[str, ...] | None = None  # None keeps the configured set
    backends: dict[str, str] = field(default_factory=dict)  # channel -> backend
    horizon: int | None = None
    optimizer: Callable[[], DROCVaROptimizer] | None = None
    overrides: dict = field(default_factory=dict)


def _jev_arms() -> list[Arm]:
    """Conventional specialists versus OpenJev, one channel at a time.

    Section 10's comparison. The conventional arm uses the numeric indicator
    cubes and the fitted heads; every OpenJev arm swaps one or more channels to
    calibrated probabilities and changes nothing else.
    """
    conventional = dict.fromkeys(("technical", "volatility"), "numeric")
    arms = [
        Arm(
            "jev_none",
            "Conventional",
            "openjev",
            backends={**conventional, "news": "none"},
        )
    ]
    for size in (1, 2, 3):
        for subset in itertools.combinations(SOURCES, size):
            backends = {**conventional, "news": "none"}
            for channel in subset:
                backends[channel] = "jev"
            arms.append(
                Arm(
                    run_id="jev_" + "_".join(subset),
                    label="OpenJev " + "+".join(subset),
                    family="openjev",
                    backends=backends,
                )
            )
    return arms


def _source_arms() -> list[Arm]:
    """All agents, each single removal, and every surviving subset."""
    arms = [Arm("src_all", "All agents", "sources", sources=SOURCES)]
    for dropped in SOURCES:
        arms.append(
            Arm(
                run_id=f"src_without_{dropped}",
                label=f"Without {dropped}",
                family="sources",
                sources=tuple(name for name in SOURCES if name != dropped),
            )
        )
    for size in (1, 2):
        for subset in itertools.combinations(SOURCES, size):
            run_id = "src_only_" + "_".join(subset)
            if any(arm.run_id == run_id for arm in arms):
                continue
            arms.append(
                Arm(
                    run_id=run_id,
                    label=" + ".join(subset) + " only",
                    family="sources",
                    sources=subset,
                )
            )
    return arms


def default_arms() -> list[Arm]:
    return [
        # -- headline gate ablation: what makes Tail-VoI a claim -----------------
        # Each control closes one escape route. equal_weight: does gating do
        # anything at all? accuracy: is this just accuracy weighting? attention:
        # is this just any fitted gate beating a fixed one?
        Arm("gate_tailvoi", "TailVoI", "gate", gate="tailvoi"),
        Arm("gate_accuracy", "Accuracy", "gate", gate="accuracy"),
        Arm("gate_attention", "Attention", "gate", gate="attention"),
        Arm("gate_equal", "EqualWeight", "gate", gate="equal_weight"),
        # -- what the distributional robustness is worth -------------------------
        Arm("opt_wasserstein", "Wasserstein DRO", "optimizer"),
        Arm(
            "opt_moment",
            "Moment DRO",
            "optimizer",
            overrides={"optimizer": {"dro": "moment"}},
        ),
        Arm(
            "opt_plain_cvar",
            "Plain CVaR",
            "optimizer",
            overrides={"optimizer": {"dro": "none"}},
        ),
        # -- static policy vs RL controller -------------------------------------
        Arm("policy_static", "Static policy", "policy"),
        Arm("policy_rl", "RL policy", "policy", rl=True),
        # -- conventional vs OpenJev specialists ---------------------------------
        *_jev_arms(),
        # -- agent-removal ablations --------------------------------------------
        *_source_arms(),
        # -- prediction / rebalance horizon ---------------------------------------
        *[
            Arm(f"horizon_{days}d", f"{days}-day horizon", "horizon", horizon=days)
            for days in (1, 5, 10, 20)
        ],
    ]


def expand_policies(
    arms: list[Arm], policies: tuple[str, ...] = ("static", "rl")
) -> list[Arm]:
    """Cross every arm with each risk policy.

    The policy is the one seam between the two pipelines, so running the whole
    matrix under both answers a question a single pass cannot: does an arm's
    effect survive the change of risk controller, or was it an artefact of the
    static rule?

    ``policy`` arms are left alone - they already vary exactly this, and
    duplicating them would only produce aliases of themselves.
    """
    unknown = set(policies) - {"static", "rl"}
    if unknown:
        raise ValueError(f"Unknown policies: {sorted(unknown)}")

    expanded: list[Arm] = []
    for arm in arms:
        if arm.family == "policy" or len(policies) == 1:
            expanded.append(
                dataclasses.replace(arm, rl=(policies[0] == "rl"))
                if arm.family != "policy" and len(policies) == 1
                else arm
            )
            continue
        for policy in policies:
            rl = policy == "rl"
            expanded.append(
                dataclasses.replace(
                    arm,
                    run_id=f"{arm.run_id}__{policy}",
                    label=f"{arm.label} [{policy}]",
                    rl=rl,
                )
            )
    return expanded


def _apply(config: Config, arm: Arm) -> Config:
    """Rewrite the config for one arm. Nothing else in the run may differ."""
    research = config.research

    if arm.sources is not None:
        research = dataclasses.replace(
            research, tailvoi=dataclasses.replace(research.tailvoi, sources=arm.sources)
        )
        if "news" not in arm.sources:
            research = dataclasses.replace(
                research, news=dataclasses.replace(research.news, backend="none")
            )

    changes = {
        f"{channel}_backend": backend
        for channel, backend in arm.backends.items()
        if channel != "news"
    }
    if changes:
        research = dataclasses.replace(
            research, specialists=dataclasses.replace(research.specialists, **changes)
        )
    if "news" in arm.backends:
        research = dataclasses.replace(
            research,
            news=dataclasses.replace(research.news, backend=arm.backends["news"]),
        )
        if arm.backends["news"] == "none":
            research = dataclasses.replace(
                research,
                tailvoi=dataclasses.replace(
                    research.tailvoi,
                    sources=tuple(c for c in research.tailvoi.sources if c != "news"),
                ),
            )

    if arm.horizon is not None:
        research = dataclasses.replace(
            research,
            panel=dataclasses.replace(research.panel, target_horizon=arm.horizon),
            backtest=dataclasses.replace(research.backtest, rebalance_days=arm.horizon),
        )

    for section, values in arm.overrides.items():
        research = dataclasses.replace(
            research,
            **{section: dataclasses.replace(getattr(research, section), **values)},
        )
    return dataclasses.replace(config, research=research)


def fingerprint(config: Config, arm: Arm) -> tuple:
    """Everything that can change a run's numbers.

    Several arms are legitimately the same experiment seen from different
    families - the shared control appears once per family, and with exactly
    three sources "without technical" and "volatility + news only" are the same
    set. They are worth reporting under both names but not worth *running*
    twice, so execution is deduplicated on this key and the summary aliases the
    extra labels onto the one run.
    """
    research = config.research
    return (
        arm.gate or config.research.tailvoi.gate,
        arm.rl,
        research.tailvoi.sources,
        research.panel.target_horizon,
        research.backtest.rebalance_days,
        research.news.backend,
        arm.optimizer.__name__ if arm.optimizer else "DROCVaR",
        research.optimizer.dro,
        research.static_policy.vol_target,
    )


def _backend_key(config: Config) -> tuple:
    research = config.research
    return (
        research.tailvoi.sources,
        research.news.backend,
        tuple(
            (channel, research.specialists.backend(channel))
            for channel in research.tailvoi.sources
            if channel != "news"
        ),
    )


def _shared_features(
    config: Config, panel: AlignedPanel, cache: dict
) -> dict[str, np.ndarray]:
    """Build each expensive cube once and share it across every arm that uses it."""
    key = (_backend_key(config), panel.horizon, len(panel.dates))
    if key not in cache:
        cache[key] = resolve_features(panel, config)
    return dict(cache[key])


def summary_table(
    config: Config,
    run_ids: list[str],
    interval: str = "test",
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """The headline comparison, one row per arm.

    ``aliases`` maps a label that was not run onto the run whose configuration
    it duplicates, so every requested arm appears in the table exactly once
    even though identical configurations executed only once.
    """
    evaluation = evaluate(config, run_ids, make_plots=False)
    table = evaluation.tables.get(interval)
    if table is None:
        table = evaluation.overall()
    root = config.path(config.research.backtest.runs)

    rows = []
    wanted = [(run_id, None) for run_id in run_ids]
    wanted += [(canonical, label) for label, canonical in (aliases or {}).items()]
    for run_id, alias_label in wanted:
        meta_path = root / run_id / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        label = meta.get("label", run_id)
        if label not in table.index:
            continue
        row = table.loc[label]
        backends = meta.get("backends", {})
        rows.append(
            {
                "experiment": alias_label or label,
                "tech_backend": backends.get("technical", "-"),
                "vol_backend": backends.get("volatility", "-"),
                "news_backend": meta.get("news_backend", "-"),
                "return": row["ARR"],
                "sharpe": row["Sharpe"],
                "sortino": row["Sortino"],
                "calmar": row["Calmar"],
                "max_dd": row["MaxDD"],
                "cvar": row["CVaR"],
                "turnover": row["Turnover"],
            }
        )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def gate_weights_by_regime(config: Config, run_ids: list[str]) -> pd.DataFrame:
    """Average Tail-VoI gate weight per channel, per named interval.

    This answers "does the volatility specialist earn more Tail-VoI during
    stress?" — gate weights are persisted at every rebalance, so the regime
    split is a groupby rather than another backtest.
    """
    root = config.path(config.research.backtest.runs)
    rows = []
    for run_id in run_ids:
        try:
            run = load_run(root, run_id)
        except FileNotFoundError:
            continue
        columns = [c for c in run.ledger.columns if c.startswith("gate_g_")]
        if not columns:
            continue
        intervals = resolve(
            run.ledger, config.research.split, config.research.evaluation
        )
        for interval in intervals:
            piece = slice_ledger(run.ledger, interval)
            if piece.empty:
                continue
            record = {
                "run": run.label,
                "interval": interval.name,
                "kind": interval.kind,
                "n": len(piece),
            }
            record.update(
                {
                    column.removeprefix("gate_g_"): float(piece[column].mean())
                    for column in columns
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def run_experiments(
    config: Config,
    *,
    arms: list[Arm] | None = None,
    families: tuple[str, ...] = FAMILIES,
    policies: tuple[str, ...] = ("static",),
    panel: AlignedPanel | None = None,
    skip_failures: bool = True,
) -> Evaluation:
    """Run every selected arm and score them all on identical intervals.

    ``policies`` crosses the matrix with each risk controller. ``("static",)``
    is the default because every RL arm trains a SAC agent per fold, and
    ``("static", "rl")`` therefore roughly triples the wall clock rather than
    doubling it.
    """
    panel = panel or build_panel(config)
    selected = [arm for arm in (arms or default_arms()) if arm.family in families]
    selected = expand_policies(selected, policies)
    logger.info(
        "experiments_planned arms=%d policies=%s families=%s",
        len(selected),
        ",".join(policies),
        ",".join(families),
    )
    cache: dict = {}
    completed: list[str] = []
    executed: dict[tuple, str] = {}
    aliases: dict[str, str] = {}
    panels: dict[int, AlignedPanel] = {panel.horizon: panel}

    for arm in selected:
        arm_config = _apply(config, arm)
        mark = fingerprint(arm_config, arm)
        if mark in executed:
            aliases[arm.label] = executed[mark]
            logger.info(
                "arm_aliased id=%s same_configuration_as=%s", arm.run_id, executed[mark]
            )
            continue
        horizon = arm_config.research.panel.target_horizon
        if horizon not in panels:
            panels[horizon] = build_panel(arm_config)
        arm_panel = panels[horizon]
        run = run_tail_voli_risk_rl if arm.rl else run_tail_voli_risk
        logger.info(
            "arm_start id=%s family=%s label=%s sources=%s",
            arm.run_id,
            arm.family,
            arm.label,
            ",".join(arm_config.research.tailvoi.sources),
        )
        try:
            run(
                arm_config,
                run_id=arm.run_id,
                panel=arm_panel,
                gate=arm.gate,
                features=_shared_features(arm_config, arm_panel, cache),
                optimizer=arm.optimizer() if arm.optimizer else None,
                label=arm.label,
                make_plots=False,
            )
            completed.append(arm.run_id)
            executed[mark] = arm.run_id
        except Exception as error:  # noqa: BLE001 - one bad arm must not sink the sweep
            if not skip_failures:
                raise
            logger.warning("arm_failed id=%s error=%s", arm.run_id, error)

    if not completed:
        raise RuntimeError("Every experiment arm failed; see the log above")

    root: Path = config.path(config.research.backtest.runs)
    root.mkdir(parents=True, exist_ok=True)
    summary_table(config, completed, aliases=aliases).to_csv(
        root / "summary.csv", index=False
    )
    gate_weights_by_regime(config, completed).to_csv(
        root / "gate_weights_by_regime.csv", index=False
    )
    logger.info(
        "experiments_done ran=%d aliased=%d summary=%s",
        len(completed),
        len(aliases),
        root / "summary.csv",
    )
    return evaluate(config, completed, panel=panel, title="TailRiskFlow experiments")
