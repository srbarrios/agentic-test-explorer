"""Graceful stop on Ctrl-C / SIGTERM, using LangGraph's cooperative drain.

The first signal asks the running graph to stop at the next superstep boundary:
the checkpoint is saved, ``GraphDrained`` is raised, and running the same command
again resumes the mission. A second signal gets the default behaviour (an
immediate abort), because the handler removes itself after the first one.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Callable, Iterable, List, Optional

from langgraph.runtime import RunControl

from agentic_explorer.utils import console

DEFAULT_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def install_drain_handlers(
    control: RunControl,
    *,
    signals: Iterable[int] = DEFAULT_SIGNALS,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Callable[[], None]:
    """Make ``signals`` request a drain on ``control``; returns a function that undoes it.

    Must be called from the running event loop. Where the loop cannot install
    signal handlers (Windows, non-main thread) nothing is installed and the
    returned function is a no-op, so Ctrl-C keeps its default behaviour.
    """
    loop = loop or asyncio.get_running_loop()
    installed: List[int] = []

    def uninstall() -> None:
        while installed:
            loop.remove_signal_handler(installed.pop())

    def on_signal(sig: int) -> None:
        name = signal.Signals(sig).name
        control.request_drain(name.lower())
        console.warn(
            f"{name} received: finishing the current step, then stopping. "
            "Run the same command again to resume. Press Ctrl-C again to abort now."
        )
        uninstall()  # a second signal falls through to the default handler

    for sig in signals:
        try:
            loop.add_signal_handler(sig, on_signal, sig)
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        installed.append(sig)

    return uninstall
