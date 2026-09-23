"""The CIO agent: one decision that sets both the gate and the risk stance.

A chief investment officer does two things a specialist cannot: decides **whom
to trust today**, and decides **how much risk the book should carry**. This
agent does exactly those two, in a single OpenJev call, and nothing else.

It fills both sockets:

* :class:`~yoda.common.types.Gate` — the trust distribution over information
  sources becomes ``g``, replacing the learned Tail-VoI gate.
* :class:`~yoda.common.types.RiskParamPolicy` — the risk stance becomes
  ``(lam, budget, turnover_penalty)``, replacing the static rule or SAC.

What it deliberately does **not** do is emit portfolio weights. The DRO-CVaR
solver still enforces ``w >= 0``, ``sum w = 1``, the per-asset cap and the CVaR
budget. A CIO that produced weights directly would turn every one of those
guarantees into a suggestion, and there would be no way to attribute the result
to anything.

The stance is mapped onto the *same* action space SAC searches, so
"CIO picks the risk parameters" and "SAC picks the risk parameters" are choices
from an identical option set and the comparison is clean.

Cost is small: one call per rebalance (~136 per run at a 5-day cadence), not one
per asset-day, so decisions are cached on demand rather than pre-built.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typesafe_sdk import Choice, Score

from yoda.common.logger import logger
from yoda.common.types import (
    Gate,
    GateOutput,
    MarketState,
    RiskParamPolicy,
    RiskParams,
    SpecialistOutput,
)
from yoda.config.research import CIOConfig, JevConfig, RLConfig, StaticPolicyConfig
from yoda.config.settings import Config
from yoda.policy.rl import action_to_params
from yoda.specialists.jev.cache import JevCache, state_hash
from yoda.specialists.jev.client import JevClient, resolve_model
from yoda.tailvoi.base import fuse, summarise

SOURCE_BRIEF: dict[str, str] = {
    "technical": "Trend, momentum and price-structure evidence.",
    "volatility": "How wide the return distribution is about to be.",
    "news": "What today's headlines imply for this market.",
}
STANCE_RUBRIC: tuple[str, ...] = (
    "Maximally defensive: protect capital, accept very little tail risk.",
    "Defensive: lean toward capital preservation.",
    "Neutral: balance return against tail risk as usual.",
    "Constructive: accept more tail risk to pursue return.",
    "Maximally aggressive: pursue return, tolerate a wide tail.",
)
COLUMNS: tuple[str, ...] = ("stance", "stance_confidence", "trust_confidence")


class CIOAgent(Gate, RiskParamPolicy):
    """Reads the desk's views, sets the gate and the risk stance."""

    name = "cio"

    def __init__(
        self,
        config: Config,
        sources: tuple[str, ...],
        cio: CIOConfig | None = None,
        jev: JevConfig | None = None,
        rl: RLConfig | None = None,
        static: StaticPolicyConfig | None = None,
    ):
        research = config.research
        self.config = config
        self.cio = cio or research.cio
        self.jev = jev or research.jev
        self.rl = rl or research.rl
        self.static = static or research.static_policy
        self.alpha = research.optimizer.alpha
        self.sources = tuple(sources)
        self._model: str | None = None  # resolved lazily; never at construction
        path: Path = config.path(self.cio.cache) / "decisions.parquet"
        self.cache = JevCache(path, flush_every=25)
        self._client: JevClient | None = None
        self._latest: dict[str, float] | None = None
        self._trust: dict[str, float] = dict.fromkeys(
            self.sources, 1.0 / len(self.sources)
        )

    # ---- the decision ----------------------------------------------------

    def _questions(self) -> dict:
        return {
            "trust": Choice(
                instructions=(
                    "Three research channels have reported on this market. Which "
                    "one should carry the most weight in today's allocation?"
                ),
                criteria={name: SOURCE_BRIEF[name] for name in self.sources},
            ),
            "stance": Score(
                instructions=(
                    "Given these views, how much tail risk should the book carry today?"
                ),
                criteria=list(STANCE_RUBRIC[: self.cio.levels]),
            ),
        }

    def _brief(self, spec: SpecialistOutput) -> dict:
        """A compact, point-in-time desk briefing - numbers only, no prose."""
        brief: dict[str, object] = {"channels": {}}
        for name in self.sources:
            view = np.asarray(spec.mu_hat[name], dtype=np.float64).reshape(-1)
            brief["channels"][name] = {
                "mean_view": round(float(np.nanmean(view)), 6),
                "dispersion": round(float(np.nanstd(view)), 6),
                "strongest": round(float(np.nanmax(np.abs(view))), 6),
            }
        summary = summarise(spec, self.sources)
        brief["state_digest"] = [round(float(v), 4) for v in summary[:12]]
        return brief

    def decide(self, spec: SpecialistOutput) -> dict[str, float]:
        """One call per distinct desk briefing; repeats come from the cache."""
        brief = self._brief(spec)
        if self._model is None:
            self._model = resolve_model(self.jev)
        key = "|".join(
            [state_hash(brief), self._model, self.cio.prompt_version, *self.sources]
        )
        if self.cache.has(key):
            row = self.cache.table(COLUMNS).loc[key]
            record = {column: float(row[column]) for column in COLUMNS}
            record.update(
                {
                    f"trust_{name}": float(row.get(f"trust_{name}", 0.0))
                    for name in self.sources
                }
            )
            return record

        if self._client is None:
            self._client = JevClient(self.jev)
        response = self._client.ask(brief, self._questions())
        trust = response.choices["trust"]
        stance = response.scores["stance"]
        record = {
            "stance": float(stance.score) / max(self.cio.levels - 1, 1),
            "stance_confidence": float(stance.confidence),
            "trust_confidence": float(trust.confidence),
            **{
                f"trust_{name}": float(trust.probabilities.get(name, 0.0))
                for name in self.sources
            },
        }
        self.cache.add(key, "portfolio", "n/a", record)
        return record

    # ---- Gate ------------------------------------------------------------

    def gate(self, spec: SpecialistOutput) -> GateOutput:
        sources = tuple(name for name in self.sources if name in spec.z)
        try:
            record = self.decide(spec)
        except Exception as error:  # noqa: BLE001 - never sink a backtest
            logger.warning(
                "cio_decision_failed error=%s; falling back to even gate", error
            )
            return fuse(spec, dict.fromkeys(sources, 1.0 / len(sources)))
        self._latest = record
        weights = {name: record.get(f"trust_{name}", 0.0) for name in sources}
        total = sum(weights.values())
        if total <= 0:
            weights = dict.fromkeys(sources, 1.0 / len(sources))
        else:
            weights = {name: value / total for name, value in weights.items()}
        self._trust = weights
        return fuse(spec, weights)

    # ---- RiskParamPolicy --------------------------------------------------

    def act(self, state: MarketState) -> RiskParams:
        """Map the stance onto the action space SAC searches.

        Aggressive means *less* risk aversion and a *wider* CVaR budget, so
        ``lam`` runs backwards against the stance while ``budget`` runs with it.
        Turnover cost is not a CIO judgement and stays at the configured value.
        """
        stance = 0.5 if self._latest is None else float(self._latest["stance"])
        stance = float(np.clip(stance, self.cio.stance_floor, self.cio.stance_ceiling))
        action = np.array([1.0 - 2.0 * stance, 2.0 * stance - 1.0, 0.0])
        params = action_to_params(action, self.rl, self.alpha)
        return RiskParams(
            lam=params.lam,
            budget=params.budget,
            turnover_penalty=self.static.turnover_penalty,
            alpha=self.alpha,
        )

    def close(self) -> None:
        self.cache.flush()
        if self._client is not None:
            self._client.close()
            self._client = None
