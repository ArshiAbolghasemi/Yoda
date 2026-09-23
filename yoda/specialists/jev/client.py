"""Access to the locally served OpenJev decision API.

Thin wrapper over ``typesafe_sdk``, which is the protocol the OpenJev shim
speaks: ``POST /v1/systemone`` with ``model``, ``state`` and ``questions``. It
returns the raw ``SystemOneResponse`` so the caller keeps the calibrated
probabilities rather than a thresholded point estimate.

The endpoint lives in configuration (``JEV__BASE_URL``, ``JEV__API_KEY``,
``JEV__MODEL``) and never in a specialist implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

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
