"""LangGraph workflow run once per asset-day of headlines.

    extract events -> {directional assessment, tail/vol relevance} -> aggregate

The two assessment nodes fan out from extraction and rejoin at aggregation,
which is pure Python - three model calls per asset-day, not four. Rows with no
headlines short-circuit to a neutral record without touching the model.

Point-in-time discipline: the prompts carry the headline text and the date and
nothing else, and instruct the model not to reason from post-date knowledge.
That cannot remove pretraining leakage, which is why the ``encoder`` backend
exists as the leakage-strict comparison (see the news ablation).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from yoda.specialists.news_agent.client import LLMClient
from yoda.specialists.news_agent.schema import (
    DirectionalView,
    EventList,
    NewsAssessment,
    TailView,
)

GUARD = (
    "You are a financial news analyst working strictly as of {date}. "
    "Use only the headlines supplied below. Do not use any knowledge of what "
    "happened after {date}. If the headlines do not support a judgement, return "
    "the neutral default. Reply with a single JSON object and nothing else."
)
EXTRACT = (
    "Extract the discrete market-relevant events for {asset}. "
    'JSON: {{"events": ["short label", ...]}} (at most 6, empty if none).'
)
DIRECTION = (
    "Assess the near-term (1-5 trading day) return impact on {asset}. "
    'JSON: {{"sentiment": -1..1, "confidence": 0..1, "rationale": "one sentence"}}.'
)
TAIL = (
    "Judge whether these events imply elevated downside-tail risk or a "
    "volatility spike for {asset}. "
    'JSON: {{"tail_risk_flag": true/false, "vol_flag": true/false}}.'
)


class AgentState(TypedDict, total=False):
    asset: str
    date: str
    headlines: str
    events: list[str]
    view: DirectionalView
    tail: TailView
    assessment: NewsAssessment


def _context(state: AgentState, extra: str = "") -> str:
    parts = [f"Asset: {state['asset']}", f"Date: {state['date']}"]
    if extra:
        parts.append(extra)
    parts.append(f"Headlines:\n{state['headlines']}")
    return "\n".join(parts)


def build_graph(client: LLMClient) -> Any:
    """Compile the per-asset-day agent graph."""

    def system(state: AgentState) -> str:
        return GUARD.format(date=state["date"])

    def extract(state: AgentState) -> AgentState:
        result = client.structured(
            system(state),
            _context(state, EXTRACT.format(asset=state["asset"])),
            EventList,
        )
        return {"events": result.events}

    def direction(state: AgentState) -> AgentState:
        question = DIRECTION.format(asset=state["asset"])
        extra = f"Events: {state.get('events', [])}\n{question}"
        return {
            "view": client.structured(
                system(state), _context(state, extra), DirectionalView
            )
        }

    def tail(state: AgentState) -> AgentState:
        extra = (
            f"Events: {state.get('events', [])}\n{TAIL.format(asset=state['asset'])}"
        )
        return {
            "tail": client.structured(system(state), _context(state, extra), TailView)
        }

    def aggregate(state: AgentState) -> AgentState:
        view = state.get("view") or DirectionalView()
        flags = state.get("tail") or TailView()
        return {
            "assessment": NewsAssessment(
                sentiment=view.sentiment,
                confidence=view.confidence,
                tail_risk_flag=flags.tail_risk_flag,
                vol_flag=flags.vol_flag,
                event_tags=state.get("events", []),
                rationale=view.rationale,
            )
        }

    graph = StateGraph(AgentState)
    for name, node in (
        ("extract", extract),
        ("direction", direction),
        ("tail", tail),
        ("aggregate", aggregate),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "direction")
    graph.add_edge("extract", "tail")
    graph.add_edge("direction", "aggregate")
    graph.add_edge("tail", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


def assess(graph: Any, asset: str, date: str, headlines: str) -> NewsAssessment:
    """Run the graph for one asset-day; empty headlines never call the model."""
    if not headlines.strip():
        return NewsAssessment.neutral()
    state = graph.invoke(
        {"asset": asset, "date": date, "headlines": headlines, "events": []}
    )
    return state.get("assessment") or NewsAssessment.neutral()
