"""Agent prompts and the structured views they return."""

from yoda.specialists.prompts.agents import (
    NEWS_PROMPT,
    PROMPT_VERSION,
    PROMPTS,
    TECHNICAL_PROMPT,
    VOLATILITY_PROMPT,
)
from yoda.specialists.prompts.schema import (
    CORE,
    VIEWS,
    AgentView,
    NewsView,
    RiskScenario,
    TechnicalView,
    VolatilityView,
)

__all__ = [
    "CORE",
    "NEWS_PROMPT",
    "PROMPTS",
    "PROMPT_VERSION",
    "TECHNICAL_PROMPT",
    "VIEWS",
    "VOLATILITY_PROMPT",
    "AgentView",
    "NewsView",
    "RiskScenario",
    "TechnicalView",
    "VolatilityView",
]
