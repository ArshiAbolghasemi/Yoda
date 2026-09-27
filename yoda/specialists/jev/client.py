"""Access to the OpenJev decision API.

Thin wrapper over ``typesafe_sdk``, which is the protocol OpenJev speaks:
``POST /v1/systemone`` with ``model``, ``state`` and ``questions``. It returns
the raw ``SystemOneResponse`` so the caller keeps the calibrated probabilities
rather than a thresholded point estimate.

**There is no generative path.** ``jev-1.13`` is a decisions model and rejects
``/v1/chat/completions`` outright::

    400 - jev-1.13 is a decisions model and cannot be used with the
          chat/completions endpoint.

Which is the right constraint to be held to: a JSON blob of self-assessed
numbers is not the thing the calibration metrics are supposed to measure.

The endpoint lives in configuration (``JEV__BASE_URL``, ``JEV__API_KEY``,
``JEV__MODEL``) and never in a specialist implementation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

from typesafe_sdk import (
    JSONContent,
    Noul,
    Question,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeClient,
)

from yoda.common.logger import logger
from yoda.config.research import JevConfig

UNRESOLVED = "unresolved"
_PING = Noul(instructions="Is this a test?")


def retry_policy(config: JevConfig) -> RetryPolicy:
    """The SDK's own retry, rather than tenacity around it.

    It backs off with jitter *and* honours ``Retry-After``, which a wrapper
    cannot do because it never sees the response header. Stacking tenacity on
    top would multiply the attempts rather than add to them - three outer
    tries over three inner ones is nine requests per failure.
    """
    return RetryPolicy(
        max_retries=config.max_retries,
        backoff_initial=0.5,
        backoff_max=20.0,
        backoff_jitter=0.5,
        respect_retry_after=True,
    )


class RateLimiter:
    """Evenly paced client-side cap, shared across the worker threads.

    The endpoint allows a fixed number of requests per minute::

        limited to 2000 requests per minute. Please retry shortly

    The feature build runs a thread pool, so without a cap ``max_concurrency``
    workers times a sub-second round trip sails past that and the whole pool
    starts collecting 429s at once - the retries then arrive together and the
    burst repeats.

    Pacing rather than bursting: each caller reserves the next slot, so the
    issue rate is the limit rather than the limit on average. The reservation
    is taken under the lock and the sleep happens outside it, so threads wait
    on the clock instead of on each other.
    """

    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


class JevClient:
    """One client per feature build; safe to share across worker threads."""

    def __init__(self, config: JevConfig):
        self.config = config
        self.client = TypeSafeClient(
            api_key=config.api_key or "x",
            base_url=config.base_url or None,
            timeout=config.timeout,
            retry=retry_policy(config),
        )
        self.limiter = RateLimiter(config.requests_per_minute)
        self.calls = 0

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ask(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> SystemOneResponse:
        """One state, every question for that specialist, one request."""
        self.limiter.acquire()
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
