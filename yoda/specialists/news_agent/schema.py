"""Structured output contract for the news agent.

The model is asked for exactly this record; anything it cannot ground in the
supplied headlines must come back neutral rather than invented.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

NEUTRAL_RATIONALE = "no headlines"


class NewsAssessment(BaseModel):
    """One asset-day verdict, the unit the feature cache stores."""

    sentiment: float = Field(
        0.0, ge=-1.0, le=1.0, description="Signed near-term return view."
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="How strongly the headlines support the view."
    )
    tail_risk_flag: bool = Field(
        False, description="Headlines imply elevated downside-tail risk."
    )
    vol_flag: bool = Field(
        False, description="Headlines imply a near-term volatility spike."
    )
    event_tags: list[str] = Field(
        default_factory=list, description="Short discrete event labels."
    )
    rationale: str = Field("", description="One sentence, grounded in the headlines.")

    @classmethod
    def neutral(cls) -> NewsAssessment:
        return cls(rationale=NEUTRAL_RATIONALE)

    def features(self) -> dict[str, float]:
        """The numeric fields that become part of ``z_news``."""
        return {
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "tail_risk_flag": float(self.tail_risk_flag),
            "vol_flag": float(self.vol_flag),
            "event_count": float(len(self.event_tags)),
        }


class EventList(BaseModel):
    """Node 1 output: discrete events pulled out of the joined headlines."""

    events: list[str] = Field(default_factory=list)


class DirectionalView(BaseModel):
    """Node 2 output: signed near-term view with a confidence."""

    sentiment: float = Field(0.0, ge=-1.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""


class TailView(BaseModel):
    """Node 3 output: what the Tail-VoI gate ultimately cares about."""

    tail_risk_flag: bool = False
    vol_flag: bool = False
