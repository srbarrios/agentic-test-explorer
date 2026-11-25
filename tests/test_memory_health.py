"""Tests for ``agentic_explorer.memory_health``: memory failures are logged and counted, never silent."""

import asyncio
import logging

import pytest

from agentic_explorer.memory_health import MemoryHealth, guarded, memory_health


@pytest.fixture
def health() -> MemoryHealth:
    return MemoryHealth()


async def _boom(message="disk full"):
    raise RuntimeError(message)


async def test_a_successful_operation_returns_its_value_and_counts_nothing(health):
    async def ok():
        return "stored"

    assert await guarded("write episode", ok, health=health) == "stored"
    assert health.count == 0


async def test_a_failing_operation_returns_the_default_and_is_counted(health):
    result = await guarded("write episode", lambda: _boom("disk full"), default="fallback", health=health)

    assert result == "fallback"
    assert health.count == 1
    assert health.errors[0].operation == "write episode"
    assert "disk full" in health.errors[0].error
    assert "RuntimeError" in health.errors[0].error


async def test_the_default_default_is_none(health):
    assert await guarded("read context", lambda: _boom(), health=health) is None


async def test_failures_are_logged_with_the_traceback(health, caplog):
    with caplog.at_level(logging.WARNING, logger="agentic_explorer.memory"):
        await guarded("write episode", lambda: _boom("disk full"), health=health)

    record = caplog.records[0]
    assert "write episode" in record.getMessage()
    assert record.exc_info is not None  # the traceback is kept for whoever configures logging


async def test_the_console_warns_once_per_operation_but_every_failure_is_counted(health, capsys):
    for _ in range(3):
        await guarded("write semantic memory", lambda: _boom(), health=health)
    await guarded("read context", lambda: _boom(), health=health)

    out = capsys.readouterr().out
    assert out.count("write semantic memory") == 1  # no wall of identical warnings
    assert out.count("read context") == 1
    assert health.count == 4


async def test_an_error_raised_while_building_the_awaitable_is_guarded_too(health):
    def broken_factory():
        raise ValueError("could not even build the call")

    assert await guarded("reflect", broken_factory, default=0, health=health) == 0
    assert health.count == 1


async def test_cancellation_and_exit_requests_are_not_swallowed(health):
    async def cancelled():
        raise asyncio.CancelledError()

    async def interrupted():
        raise KeyboardInterrupt()

    with pytest.raises(asyncio.CancelledError):
        await guarded("write", cancelled, health=health)
    with pytest.raises(KeyboardInterrupt):
        await guarded("write", interrupted, health=health)
    assert health.count == 0


def test_count_since_reports_what_happened_after_a_mark(health):
    health.record("a", RuntimeError("x"))
    mark = health.count
    health.record("b", RuntimeError("y"))
    health.record("c", RuntimeError("z"))

    assert health.count_since(mark) == 2


def test_reset_clears_errors_and_the_warned_operations(health, capsys):
    health.record("a-operation", RuntimeError("x"))
    health.reset()

    assert health.count == 0
    capsys.readouterr()  # drop the first warning
    health.record("a-operation", RuntimeError("x"))
    assert "a-operation" in capsys.readouterr().out  # warns again after a reset


def test_the_module_logger_is_silent_unless_someone_configures_it():
    logger = logging.getLogger("agentic_explorer.memory")

    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_a_process_wide_instance_is_shared():
    assert isinstance(memory_health, MemoryHealth)
