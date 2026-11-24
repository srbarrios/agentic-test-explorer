"""Start and stop the optional Streamlit dashboard (``--visual``)."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
import webbrowser
from typing import Optional

from agentic_explorer.ui import state_emitter
from agentic_explorer.utils import console

DASHBOARD_URL = "http://localhost:8501"
_DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.py")


def start_dashboard(*, app_url: str, provider: str, model_name: str) -> subprocess.Popen:
    """Reset Visual Mode state, launch Streamlit and open it in the browser.

    The process is terminated at interpreter exit and on Ctrl-C (the previous
    SIGINT handler still runs afterwards).
    """
    # Clear any stale state from previous runs.
    state_emitter.cleanup()
    state_emitter.enable()
    state_emitter.update(app_url=app_url, provider=provider, model_name=model_name)

    proc = subprocess.Popen(
        [
            "streamlit", "run", _DASHBOARD_PATH,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    atexit.register(stop_dashboard, proc)
    previous_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(sig, frame):
        stop_dashboard(proc)
        if callable(previous_handler):
            previous_handler(sig, frame)

    signal.signal(signal.SIGINT, _on_sigint)
    console.success("Visual Mode dashboard launched (Streamlit)")

    # Give Streamlit a moment to start, then open the browser.
    time.sleep(2)
    webbrowser.open(DASHBOARD_URL)
    return proc


def stop_dashboard(proc: Optional[subprocess.Popen]) -> None:
    """Terminate the dashboard if it is still running.

    State files are kept on purpose so the dashboard can show completion.
    """
    if proc and proc.poll() is None:
        proc.terminate()
