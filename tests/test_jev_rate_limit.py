"""The client-side cap on OpenJev requests.

The endpoint allows a fixed number per minute ("limited to 2000 requests per
minute"). The feature build runs a thread pool, so the cap has to hold across
threads, not just within one.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from yoda.specialists.jev.client import RateLimiter


def test_spacing_matches_the_configured_rate():
    limiter = RateLimiter(per_minute=6000)  # 100/s -> 10ms apart
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    elapsed = time.monotonic() - start
    # Four gaps of 10ms after the first (free) slot.
    assert 0.035 <= elapsed < 0.15, elapsed


def test_the_cap_holds_across_threads():
    """Sixteen workers must not each get their own allowance."""
    limiter = RateLimiter(per_minute=6000)
    stamps: list[float] = []
    guard = threading.Lock()

    def call(_):
        limiter.acquire()
        with guard:
            stamps.append(time.monotonic())

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(call, range(32)))

    stamps.sort()
    span = stamps[-1] - stamps[0]
    assert span >= 0.28, f"32 requests at 100/s cannot finish in {span:.3f}s"
    # No two issued closer than the interval, allowing for clock granularity.
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    assert min(gaps) > 0.005, min(gaps)


def test_zero_disables_the_limiter():
    limiter = RateLimiter(per_minute=0)
    start = time.monotonic()
    for _ in range(1000):
        limiter.acquire()
    assert time.monotonic() - start < 0.05


def test_the_first_call_is_not_delayed():
    """A cold limiter must not make the very first request wait a full slot."""
    limiter = RateLimiter(per_minute=60)  # one per second
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.05
