"""Access to the locally served OpenJev decision API.

Thin wrapper over ``typesafe_sdk``, which is the protocol the OpenJev shim
speaks: ``POST /v1/systemone`` with ``model``, ``state`` and ``questions``. It
returns the raw ``SystemOneResponse`` so the caller keeps the calibrated
probabilities rather than a thresholded point estimate.

The endpoint lives in configuration (``JEV__BASE_URL``, ``JEV__API_KEY``,
``JEV__MODEL``) and never in a specialist implementation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typesafe_sdk import (
    JSONContent,
    Noul,
    Question,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeClient,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from yoda.common.logger import logger
from yoda.config.research import JevConfig
from yoda.specialists.prompts.schema import AgentView

TRANSIENT = (
    TypeSafeRateLimitError,
    TypeSafeAPITimeoutError,
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
)
UNRESOLVED = "unresolved"
_PING = Noul(instructions="Is this a test?")


class JevClient:
    """One client per feature build; safe to share across worker threads."""

    def __init__(self, config: JevConfig):
        self.config = config
        self.client = TypeSafeClient(
            api_key=config.api_key or "x",
            base_url=config.base_url or None,
            timeout=config.timeout,
        )
        self.calls = 0

        # The shim proxies /v1/chat/completions through to vLLM, which is how
        # the agent prompts are served. Decision questions and generation share
        # one endpoint and one set of credentials.
        self.chat = OpenAI(
            base_url=f"{(config.base_url or '').rstrip('/')}/v1",
            api_key=config.api_key or "x",
            timeout=config.timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type(TRANSIENT),
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=20),
        reraise=True,
    )
    def ask(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> SystemOneResponse:
        """One state, every question for that specialist, one request."""
        response = self.client.system_one(
            state=state,
            questions=questions,
            model=self.config.model or None,
        )
        self.calls += 1
        return response

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=15),
        reraise=True,
    )
    def view(self, prompt: str, state: dict, schema: type[AgentView]) -> AgentView:
        """Run an agent prompt and validate the JSON it returns.

        The prompts ask for a strict JSON object and nothing else. A response
        that will not parse or will not validate raises, so the caller records
        an explicit ``inference_status`` instead of quietly accepting a
        malformed view.
        """
        response = self.chat.chat.completions.create(
            model=self.config.model or "qwen",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(state, sort_keys=True)},
            ],
        )
        self.calls += 1
        text = response.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}")
        if not 0 <= start < end:
            raise ValueError("agent returned no JSON object")
        return schema.model_validate(json.loads(text[start : end + 1]))


def resolve_model(config: JevConfig) -> str:
    """The model version recorded in the cache key.

    An explicit pin is preferred; without one the served default is probed once
    and frozen for the run, so a mid-run model rotation cannot silently mix two
    models into one table.

    An unreachable endpoint returns ``"unresolved"`` rather than raising:
    callers must stay constructable offline. Keys written under it cannot
    collide with real-model keys, which is the safe direction to fail in.
    """
    if config.model:
        return config.model
    try:
        with TypeSafeClient(
            api_key=config.api_key or "x",
            base_url=config.base_url or None,
            timeout=config.timeout,
        ) as client:
            response = client.system_one(
                state="ping", questions={"ok": _PING}, model=None
            )
        logger.info("jev_model_resolved model=%s", response.model)
        return response.model
    except Exception as error:  # noqa: BLE001 - offline is a valid state here
        logger.warning(
            "jev_model_unresolved error=%s; cache keys will use %r",
            error.__class__.__name__,
            UNRESOLVED,
        )
        return UNRESOLVED
