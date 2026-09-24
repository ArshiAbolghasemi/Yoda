"""Inference status and the feature layout each channel persists.

The agents return structured views (see :mod:`yoda.specialists.prompts`); this
module says which of their numeric fields become the feature cube, and records
how an inference ended.
"""

from __future__ import annotations

from yoda.specialists.prompts.schema import CORE, VIEWS


class InferenceStatus:
    OK = "ok"
    NO_INPUT = "no_input"
    RETRY_EXHAUSTED = "retry_exhausted"
    INVALID_RESPONSE = "invalid_response"


def _columns(channel: str) -> tuple[str, ...]:
    """Every numeric field the channel's view declares, core variables first."""
    model = VIEWS[channel]
    numeric = [
        name
        for name, field in model.model_fields.items()
        if field.annotation in (float, int) and name not in CORE
    ]
    return (*CORE, *sorted(numeric))


CHANNELS: dict[str, dict] = {
    channel: {
        "columns": _columns(channel),
        # The six shared variables, in the order the gate compares them.
        "core": CORE,
    }
    for channel in VIEWS
}


def neutral(channel: str) -> dict:
    """The deterministic no-evidence record.

    Used only where there is genuinely nothing to judge - an asset-day with no
    headlines. Uncertainty is maximal rather than zero: the absence of evidence
    is a statement about what we do not know, not a confident neutral view.
    """
    record = dict.fromkeys(CHANNELS[channel]["columns"], 0.0)
    record["epistemic_uncertainty"] = 1.0
    record["has_news"] = 0.0
    record["view_summary"] = ""
    record["narrative"] = "{}"
    record["inference_status"] = InferenceStatus.NO_INPUT
    return record
