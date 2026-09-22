"""The seam.

Everything below this interface is shared. ``tail_voli_risk`` injects
:class:`~yoda.policy.static.StaticRiskPolicy`, ``tail_voli_risk_rl`` injects
:class:`~yoda.policy.rl.RLRiskPolicy`, and the optimizer cannot tell them apart.

A direct-weight RL policy is a later drop-in at this same level: it replaces
the policy *and* the allocation call, and still touches nothing below.
"""

from __future__ import annotations

from yoda.common.types import MarketState, RiskParamPolicy, RiskParams

__all__ = ["MarketState", "RiskParamPolicy", "RiskParams"]
