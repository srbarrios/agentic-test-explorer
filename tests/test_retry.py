"""Tests for ``agentic_explorer.runtime.retry``.

The delays and the transient/rate-limit classification are the ones that were
duplicated inline in ``main.py`` (mission run and report generation).
"""

import pytest

from agentic_explorer.runtime import retry
from agentic_explorer.runtime.retry import (
    is_rate_limit,
    is_transient_error,
    retry_delay,
    with_retry,
)


@pytest.fixture
def no_jitter(monkeypatch):
    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 0

    monkeypatch.setattr(retry.random, "randint", fake_randint)
    return calls


class _Recorder:
    """An async callable that fails with the given exceptions, then returns 'ok'."""

    def __init__(self, *failures: Exception):
        self.failures = list(failures)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def _sleeper():
    delays: list = []

    async def sleep(seconds):
        delays.append(seconds)

    return delays, sleep


# ---------------------------------------------------------------- classification

@pytest.mark.parametrize("message", [
    "503 Service Unavailable", "UNAVAILABLE", "Error 429", "rate_limit_error",
    "RESOURCE_EXHAUSTED: quota", "quota exceeded",
])
def test_transient_errors(message):
    assert is_transient_error(Exception(message)) is True


@pytest.mark.parametrize("message", ["401 Unauthorized", "bad json", "KeyError: 'x'"])
def test_non_transient_errors(message):
    assert is_transient_error(Exception(message)) is False


@pytest.mark.parametrize("message, expected", [
    ("429 Too Many Requests", True),
    ("rate_limit_error", True),
    ("RESOURCE_EXHAUSTED", True),
    ("quota exceeded", True),
    ("503 Service Unavailable", False),  # transient, but not a rate limit
])
def test_rate_limit_errors(message, expected):
    assert is_rate_limit(Exception(message)) is expected


# ---------------------------------------------------------------- delays

def test_exponential_delay_for_plain_transient_errors():
    exc = Exception("503 unavailable")
    assert [retry_delay(a, exc) for a in range(4)] == [2, 4, 8, 16]
    assert retry_delay(2, exc, base_delay=3) == 12


def test_rate_limit_delay_is_long_and_grows_with_jitter(no_jitter):
    exc = Exception("429 rate limit")
    assert [retry_delay(a, exc) for a in range(3)] == [30, 45, 60]
    assert set(no_jitter) == {(0, 10)}  # jitter is randint(0, 10)


# ---------------------------------------------------------------- with_retry

async def test_returns_value_without_sleeping_when_first_call_succeeds():
    delays, sleep = _sleeper()
    fn = _Recorder()

    assert await with_retry(fn, sleep=sleep) == "ok"
    assert fn.calls == 1
    assert delays == []


async def test_succeeds_after_transient_failures():
    delays, sleep = _sleeper()
    fn = _Recorder(Exception("503 unavailable"), Exception("503 unavailable"))

    assert await with_retry(fn, sleep=sleep) == "ok"
    assert fn.calls == 3
    assert delays == [2, 4]


async def test_non_transient_error_is_raised_immediately():
    delays, sleep = _sleeper()
    boom = ValueError("bad input")
    fn = _Recorder(boom)

    with pytest.raises(ValueError) as info:
        await with_retry(fn, sleep=sleep)

    assert info.value is boom
    assert fn.calls == 1
    assert delays == []


async def test_gives_up_after_max_attempts_and_reraises_last_error(capsys):
    delays, sleep = _sleeper()
    last = Exception("503 last")
    fn = _Recorder(Exception("503 a"), Exception("503 b"), last)

    with pytest.raises(Exception) as info:
        await with_retry(fn, max_attempts=3, sleep=sleep)

    assert info.value is last
    assert fn.calls == 3
    assert delays == [2, 4]  # no sleep after the final failure
    assert "Failed after 3 attempts." in capsys.readouterr().out


async def test_custom_fail_message(capsys):
    _, sleep = _sleeper()
    fn = _Recorder(Exception("503 a"), Exception("503 b"))

    with pytest.raises(Exception):
        await with_retry(
            fn, max_attempts=2, sleep=sleep,
            fail_message="Report generation failed after {max_attempts} attempts.",
        )

    assert "Report generation failed after 2 attempts." in capsys.readouterr().out


async def test_rate_limit_uses_long_delay_and_warns(no_jitter, capsys):
    delays, sleep = _sleeper()
    fn = _Recorder(Exception("429 Too Many Requests"))

    assert await with_retry(fn, sleep=sleep) == "ok"

    assert delays == [30]
    out = capsys.readouterr().out
    assert "Rate limit hit" in out
    assert "attempt 1/5" in out


async def test_on_retry_hook_receives_attempt_number_and_error():
    _, sleep = _sleeper()
    seen = []
    err = Exception("503 a")
    fn = _Recorder(err)

    await with_retry(fn, sleep=sleep, on_retry=lambda attempt, exc: seen.append((attempt, exc)))

    assert seen == [(1, err)]
