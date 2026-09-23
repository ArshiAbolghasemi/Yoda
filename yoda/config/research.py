"""Configuration domains for the allocation research stack.

Mirrors ``yoda.data.settings``: frozen dataclasses built from the shared
Dynaconf settings, so a run is fully specified by ``.env``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from os import getenv
from pathlib import Path
from typing import Any

from dynaconf import Dynaconf

from yoda.common.types import SOURCES


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


@dataclass(frozen=True)
class PanelConfig:
    """Calendar alignment of the mixed-calendar panel."""

    dataset: str = ""  # resolved against storage when empty
    calendar: str = "equity"  # equity (NYSE) | intersection | union
    max_forward_fill: int = 5  # rows a stale price may be carried across a gap
    target_horizon: int = 1  # which future_return_Nd drives specialist training
    min_history: int = 250  # rows dropped from the head (indicator warm-up)
    # Corsi (2009) HAR horizons: daily, weekly, monthly.
    realized_vol_windows: tuple[int, ...] = (1, 5, 22)

    def __post_init__(self) -> None:
        if self.calendar not in {"equity", "intersection", "union"}:
            raise ValueError(f"Unknown calendar policy: {self.calendar}")


@dataclass(frozen=True)
class SplitConfig:
    """Contiguous, non-overlapping walk-forward boundaries."""

    train_start: str = "2017-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2026-09-18"
    refit_days: int = 0  # 0 => single fold; >0 => rolling refit stride

    def __post_init__(self) -> None:
        bounds = [
            ("train_start", self.train_start),
            ("train_end", self.train_end),
            ("val_start", self.val_start),
            ("val_end", self.val_end),
            ("test_start", self.test_start),
            ("test_end", self.test_end),
        ]
        values = [date.fromisoformat(_iso(value)) for _, value in bounds]
        for (name, _), (next_name, _), left, right in zip(
            bounds, bounds[1:], values, values[1:], strict=False
        ):
            if left >= right:
                raise ValueError(
                    f"Split boundaries must increase: {name} >= {next_name}"
                )

    @property
    def ranges(self) -> dict[str, tuple[str, str]]:
        return {
            "train": (self.train_start, self.train_end),
            "val": (self.val_start, self.val_end),
            "test": (self.test_start, self.test_end),
        }


@dataclass(frozen=True)
class SpecialistsConfig:
    # Wide enough for the widest channel (news, 27 columns) so no
    # probability is truncated out of z_i.
    z_dim: int = 28
    ridge_alpha: float = 10.0
    seed: int = 7
    # Numeric head per channel. See yoda/specialists/base.py::make_head.
    technical_model: str = "mlp"  # mlp | ridge | gbm | pcr
    volatility_model: str = "mlp"  # mlp | gbm | har | ridge | pcr
    news_model: str = "mlp"  # mlp | ridge | gbm | pcr
    # Feature backend per channel. OpenJev is the default; ``numeric`` keeps the
    # conventional-model arm the specialist ablation compares against.
    technical_backend: str = "jev"  # jev | numeric
    volatility_backend: str = "jev"  # jev | numeric

    def __post_init__(self) -> None:
        for channel in ("technical", "volatility"):
            backend = getattr(self, f"{channel}_backend")
            if backend not in {"jev", "numeric"}:
                raise ValueError(f"Unknown {channel} backend: {backend}")

    def backend(self, channel: str) -> str:
        return getattr(self, f"{channel}_backend", "jev")


@dataclass(frozen=True)
class JevConfig:
    """OpenJev decision-model access for the probabilistic specialists.

    OpenJev answers typed questions with calibrated probabilities instead of
    generating text, which is what makes Brier score and expected calibration
    error meaningful for these channels.

    The default endpoint is the local OpenJev decision shim
    (``scripts/serve-openjev.sh``), which speaks the TypeSafe System One
    protocol in front of a vLLM server. Pointing ``JEV__BASE_URL`` at the hosted
    TypeSafe API instead is a ``.env`` change and nothing else.
    """

    api_key: str = "x"  # the shim's SHIM_TOKEN; any value when it is unset
    base_url: str = "http://127.0.0.1:3000"  # API root; the SDK adds /v1/systemone
    model: str = ""  # pin a version; empty records whatever the shim reports
    prompt_version: str = "v1"  # part of the cache key
    max_concurrency: int = 8
    timeout: float = 120.0
    cache: str = "processed/jev"  # one parquet per channel
    # Deterministic fake answers instead of inference. For CI and dry runs
    # only: every run made this way is stamped ``synthetic`` in run_meta.json
    # and logs a warning, because its numbers mean nothing.
    synthetic: bool = False


@dataclass(frozen=True)
class CIOConfig:
    """The CIO agent: one decision that sets both the gate and the risk stance."""

    prompt_version: str = "v1"  # part of the decision cache key
    cache: str = "processed/cio"
    # The stance rubric is mapped onto the SAME action space SAC searches, so a
    # CIO run and an RL run are choosing from an identical set of options.
    levels: int = 5  # rubric 0 (maximally defensive) .. levels-1 (aggressive)
    stance_floor: float = 0.0  # clamp the usable stance range if needed
    stance_ceiling: float = 1.0


@dataclass(frozen=True)
class NewsConfig:
    """Headlines reach the portfolio through OpenJev, or not at all."""

    backend: str = "jev"  # jev | none

    def __post_init__(self) -> None:
        if self.backend not in {"jev", "none"}:
            raise ValueError(f"Unknown news backend: {self.backend}")


@dataclass(frozen=True)
class CopulaConfig:
    marginals: str = "empirical"  # empirical | garch_t
    shrinkage: float = 0.1  # ridge toward identity, keeps R well conditioned
    nu_grid: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)
    n_scenarios: int = 2000
    vol_halflife: int = 60  # EWMA halflife for the conditional vol scale
    corr_window: int = 250  # recent window blended into the conditional R
    corr_blend: float = 0.5
    seed: int = 11


@dataclass(frozen=True)
class OptimizerConfig:
    alpha: float = 0.05  # FIXED tail level, never an action
    dro: str = "wasserstein"  # wasserstein | moment | none
    radius: float = 0.001  # Wasserstein ball radius / moment ellipsoid kappa
    budget_mode: str = "constraint"  # constraint | penalty
    budget_penalty: float = 50.0  # hinge weight when budget_mode == "penalty"
    max_weight: float = 0.25
    solver: str = "CLARABEL"

    def __post_init__(self) -> None:
        if self.dro not in {"wasserstein", "moment", "none"}:
            raise ValueError(f"Unknown DRO mode: {self.dro}")
        if self.budget_mode not in {"constraint", "penalty"}:
            raise ValueError(f"Unknown budget mode: {self.budget_mode}")


@dataclass(frozen=True)
class TailVoIConfig:
    sources: tuple[str, ...] = SOURCES
    temperature: float = 1.0  # softmax temperature over predicted deltas
    mode: str = "softmax"  # softmax | threshold
    threshold: float = 0.0
    hidden: tuple[int, ...] = (32,)
    n_states: int = 250  # counterfactual states sampled from the train window
    counterfactual_scenarios: int = 500  # cheaper scenario set for the generator
    rho: str = "realized"  # realized | reference: the risk functional in Delta_i
    rho_window: int = 60  # forward days the realized CVaR is measured over
    seed: int = 13

    def __post_init__(self) -> None:
        if self.rho not in {"realized", "reference"}:
            raise ValueError(f"Unknown Tail-VoI risk functional: {self.rho}")
        if self.mode not in {"softmax", "threshold"}:
            raise ValueError(f"Unknown gate mode: {self.mode}")


@dataclass(frozen=True)
class StaticPolicyConfig:
    lam: float = 1.0
    budget: float = 0.05
    turnover_penalty: float = 0.001
    vol_target: float = 0.0  # >0 scales the budget toward a realized-CVaR target


@dataclass(frozen=True)
class RLConfig:
    total_timesteps: int = 5_000  # each step is a convex solve; raise deliberately
    env_scenarios: int = 500  # smaller scenario set inside the training loop
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 50_000
    learning_starts: int = 500
    seed: int = 17
    synthetic_episodes: int = 0  # copula-generated episodes appended to training
    lam_bounds: tuple[float, float] = (0.1, 10.0)
    budget_bounds: tuple[float, float] = (0.01, 0.20)
    turnover_bounds: tuple[float, float] = (0.0, 0.01)
    reward_lambda_cvar: float = 1.0  # FIXED shaping weight, distinct from action lam
    reward_lambda_dd: float = 0.5
    reward_turnover_cost: float = 0.001
    checkpoint: str = "processed/rl/sac_risk_policy"


@dataclass(frozen=True)
class BacktestConfig:
    rebalance_days: int = 5
    cost_bps: float = 5.0  # one-way transaction cost on turnover
    runs: str = "processed/runs"
    seed: int = 23


@dataclass(frozen=True)
class EvalConfig:
    intervals: tuple[dict[str, str], ...] = ()
    rolling_windows: tuple[int, ...] = (20, 60)
    periods_per_year: int = 252
    plots: bool = True


@dataclass(frozen=True)
class ResearchConfig:
    panel: PanelConfig
    split: SplitConfig
    specialists: SpecialistsConfig
    jev: JevConfig
    cio: CIOConfig
    news: NewsConfig
    copula: CopulaConfig
    optimizer: OptimizerConfig
    tailvoi: TailVoIConfig
    static_policy: StaticPolicyConfig
    rl: RLConfig
    backtest: BacktestConfig
    evaluation: EvalConfig


def _known(values: dict[str, Any], target: type) -> dict[str, Any]:
    """Keep only keys the target dataclass declares - see yoda/data/settings.py."""
    fields = {field.name for field in dataclasses.fields(target)}
    return {key: value for key, value in values.items() if key in fields}


def _section(settings: Dynaconf, name: str) -> dict[str, Any]:
    value = settings.get(name, {})
    if not value:
        return {}
    return {str(key).lower(): item for key, item in dict(value).items()}


def _tuple(values: Any, cast) -> tuple:
    return tuple(cast(value) for value in values)


def build_research_config(settings: Dynaconf) -> ResearchConfig:
    """Build every research domain from ``SECTION__KEY`` environment settings."""
    panel = _section(settings, "panel")
    if "realized_vol_windows" in panel:
        panel["realized_vol_windows"] = _tuple(panel["realized_vol_windows"], int)

    split = {key: _iso(value) for key, value in _section(settings, "split").items()}
    if "refit_days" in split:
        split["refit_days"] = int(split["refit_days"])

    jev = _section(settings, "jev")
    # Only let the environment override when it actually carries a value, or the
    # dataclass defaults (the local shim endpoint) would be clobbered by "".
    for key, variable in (
        ("api_key", "TYPESAFE_API_KEY"),
        ("base_url", "TYPESAFE_BASE_URL"),
    ):
        value = getenv(variable, "")
        if value and key not in jev:
            jev[key] = value

    news = _section(settings, "news")

    copula = _section(settings, "copula")
    if "nu_grid" in copula:
        copula["nu_grid"] = _tuple(copula["nu_grid"], float)

    tailvoi = _section(settings, "tailvoi")
    for key, cast in (("sources", str), ("hidden", int)):
        if key in tailvoi:
            tailvoi[key] = _tuple(tailvoi[key], cast)

    rl = _section(settings, "rl")
    for key in ("lam_bounds", "budget_bounds", "turnover_bounds"):
        if key in rl:
            low, high = _tuple(rl[key], float)
            rl[key] = (low, high)

    evaluation = _section(settings, "eval")
    if "intervals" in evaluation:
        evaluation["intervals"] = tuple(
            {str(k).lower(): _iso(v) for k, v in dict(item).items()}
            for item in evaluation["intervals"]
        )
    if "rolling_windows" in evaluation:
        evaluation["rolling_windows"] = _tuple(evaluation["rolling_windows"], int)

    return ResearchConfig(
        panel=PanelConfig(**panel),
        split=SplitConfig(**split),
        specialists=SpecialistsConfig(**_section(settings, "specialists")),
        jev=JevConfig(**_known(jev, JevConfig)),
        cio=CIOConfig(**_section(settings, "cio")),
        news=NewsConfig(**_known(news, NewsConfig)),
        copula=CopulaConfig(**copula),
        optimizer=OptimizerConfig(**_section(settings, "opt")),
        tailvoi=TailVoIConfig(**tailvoi),
        static_policy=StaticPolicyConfig(**_section(settings, "static_policy")),
        rl=RLConfig(**rl),
        backtest=BacktestConfig(**_section(settings, "backtest")),
        evaluation=EvalConfig(**evaluation),
    )


def resolve(root: Path, value: str) -> Path:
    """Resolve a storage-relative setting against the data root."""
    path = Path(value)
    return path if path.is_absolute() else root / path
