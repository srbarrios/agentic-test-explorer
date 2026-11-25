"""Memory failures are never silent: log them, count them, tell the person once.

Long-term memory is best-effort by design: a broken store must not abort a mission.
But swallowing the error with ``except Exception: pass`` meant a mission could run
for an hour without learning anything and nobody would know. ``guarded`` keeps the
"carry on" behaviour and adds the missing visibility:

* every failure is logged with its traceback to the ``agentic_explorer.memory``
  logger (silent unless the application configures logging);
* every failure is counted, and the mission report shows the total;
* the console warns about the first failure of each kind, not every repetition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Set, TypeVar

from agentic_explorer.utils import console

T = TypeVar("T")

logger = logging.getLogger("agentic_explorer.memory")
logger.addHandler(logging.NullHandler())  # library etiquette: no output unless configured


@dataclass(frozen=True)
class MemoryFailure:
    operation: str
    error: str


class MemoryHealth:
    """Failures of memory operations seen so far in this process."""

    def __init__(self) -> None:
        self.errors: List[MemoryFailure] = []
        self._warned: Set[str] = set()

    @property
    def count(self) -> int:
        return len(self.errors)

    def count_since(self, mark: int) -> int:
        """Failures recorded after ``mark`` (a previous value of ``count``)."""
        return self.count - mark

    def record(self, operation: str, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"
        self.errors.append(MemoryFailure(operation, error))
        if operation not in self._warned:
            self._warned.add(operation)
            console.warn(
                f"Memory: {operation} failed ({error}). Learning from this run may be "
                "incomplete; further failures of this kind are only counted."
            )

    def reset(self) -> None:
        self.errors.clear()
        self._warned.clear()


memory_health = MemoryHealth()


async def guarded(
    operation: str,
    factory: Callable[[], Awaitable[T]],
    *,
    default: Optional[T] = None,
    health: MemoryHealth = memory_health,
) -> Optional[T]:
    """Await ``factory()``; if it fails, log and count the error and return ``default``.

    ``factory`` is called inside the guard, so errors raised while *building* the
    call are covered too. Only ``Exception`` is caught: cancellation and
    ``KeyboardInterrupt`` still propagate.
    """
    try:
        return await factory()
    except Exception as exc:
        logger.warning("Memory operation failed: %s", operation, exc_info=True)
        health.record(operation, exc)
        return default
