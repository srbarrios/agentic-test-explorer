"""Graceful stop: the first signal asks LangGraph to drain, a second one aborts.

Real signals are used, but ``SIGUSR1`` instead of ``SIGINT`` so pytest's own
Ctrl-C handling is never touched.
"""

import asyncio
import os
import signal

import pytest
from langgraph.runtime import RunControl

from agentic_explorer.runtime.drain import install_drain_handlers


async def _deliver(sig) -> None:
    os.kill(os.getpid(), sig)
    await asyncio.sleep(0.05)  # let the event loop run the handler


async def test_first_signal_requests_a_drain_and_explains_how_to_resume(capsys):
    control = RunControl()
    uninstall = install_drain_handlers(control, signals=(signal.SIGUSR1,))
    try:
        await _deliver(signal.SIGUSR1)
    finally:
        uninstall()

    assert control.drain_requested is True
    assert control.drain_reason == "sigusr1"
    out = capsys.readouterr().out
    assert "finishing the current step" in out
    assert "Run the same command again to resume" in out
    assert "again to abort" in out


async def test_the_handler_removes_itself_so_a_second_signal_aborts():
    control = RunControl()
    install_drain_handlers(control, signals=(signal.SIGUSR1,))

    await _deliver(signal.SIGUSR1)

    loop = asyncio.get_running_loop()
    assert loop.remove_signal_handler(signal.SIGUSR1) is False  # already gone: default behaviour is back


async def test_uninstall_removes_the_handlers_without_a_signal():
    control = RunControl()
    uninstall = install_drain_handlers(control, signals=(signal.SIGUSR1,))

    uninstall()

    assert asyncio.get_running_loop().remove_signal_handler(signal.SIGUSR1) is False
    assert control.drain_requested is False


async def test_uninstall_is_safe_to_call_twice():
    uninstall = install_drain_handlers(RunControl(), signals=(signal.SIGUSR1,))

    uninstall()
    uninstall()


async def test_platforms_without_signal_handlers_degrade_to_a_noop():
    class _NoSignalsLoop:
        def add_signal_handler(self, *args):
            raise NotImplementedError  # e.g. the Windows event loop

        def remove_signal_handler(self, *args):
            return False

    uninstall = install_drain_handlers(RunControl(), signals=(signal.SIGUSR1,), loop=_NoSignalsLoop())

    uninstall()  # nothing to undo, and nothing raised


@pytest.mark.parametrize("sig, reason", [(signal.SIGINT, "sigint"), (signal.SIGTERM, "sigterm")])
def test_defaults_cover_ctrl_c_and_termination(sig, reason):
    from agentic_explorer.runtime import drain

    assert sig in drain.DEFAULT_SIGNALS
    assert signal.Signals(sig).name.lower() == reason
