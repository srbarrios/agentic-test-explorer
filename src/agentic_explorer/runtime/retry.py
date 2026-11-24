"""Retry policy for transient LLM/API failures (503s, quota and rate limits).

Replaces the identical retry loops that used to be inlined in ``main.py`` for
the mission run and for report generation.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, Optional, TypeVar

from agentic_explorer.utils import console

T = TypeVar("T")

# 529 / "overloaded_error" is Anthropic's temporary-overload response: retry with the
# normal exponential backoff (it is not a quota, so no long rate-limit wait).
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RATE_LIMIT", "RESOURCE_EXHAUSTED", "QUOTA", "529", "OVERLOADED")
_RATE_LIMIT_MARKERS = ("429", "RATE_LIMIT", "RESOURCE_EXHAUSTED", "QUOTA")


def is_transient_error(exc: Exception) -> bool:
    """True for errors worth retrying: service unavailable, quota or rate limiting."""
    msg = str(exc).upper()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def is_rate_limit(exc: Exception) -> bool:
    """True when the transient error is specifically a quota/rate limit."""
    msg = str(exc).upper()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def retry_delay(attempt: int, exc: Exception, base_delay: int = 2) -> int:
    """Seconds to wait after failure number ``attempt`` (0-based).

    Rate limits wait long (``30 + 15 * attempt`` plus 0-10 s of jitter);
    other transient errors back off exponentially (``base_delay * 2**attempt``).
    """
    if is_rate_limit(exc):
        return 30 + attempt * 15 + random.randint(0, 10)
    return base_delay * (2 ** attempt)


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    base_delay: int = 2,
    fail_message: str = "Failed after {max_attempts} attempts.",
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
) -> T:
    """Await ``fn()``, retrying transient errors up to ``max_attempts`` times.

    Non-transient errors, and the last transient one once attempts run out,
    are re-raised unchanged. ``on_retry(attempt, exc)`` runs before each wait,
    with ``attempt`` 1-based. ``sleep`` is injectable for tests.
    """
    do_sleep = sleep or asyncio.sleep
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            if attempt >= max_attempts - 1:
                console.fail(fail_message.format(max_attempts=max_attempts))
                raise
            if is_rate_limit(exc):
                console.warn(f"Rate limit hit: {str(exc)[:120]}")
            delay = retry_delay(attempt, exc, base_delay)
            console.retry(attempt + 1, max_attempts, delay)
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            await do_sleep(delay)
    raise AssertionError("unreachable: max_attempts must be >= 1")  # pragma: no cover
