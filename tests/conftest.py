"""Shared fixtures."""

import pytest

from agentic_explorer.memory_health import memory_health


@pytest.fixture(autouse=True)
def _isolate_memory_health():
    """The failure counter is process-wide; a test that breaks the store must not leak into the next."""
    memory_health.reset()
    yield
    memory_health.reset()
