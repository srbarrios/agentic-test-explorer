"""Start and stop the optional Streamlit dashboard (``--visual``)."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
import webbrowser
from typing import Optional

from agentic_explorer.ui import state_emitter
from agentic_explorer.utils import console

DASHBOARD_URL = "http://localhost:8501"
_DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.py")


def start_dashboard(*, app_url: str, provider: str, model_name: str) -> subprocess.Popen:
    """Reset Visual Mode state, launch Streamlit and open it in the browser.

    The process is terminated when the interpreter exits.
    """
    # Clear any stale state from previous runs.
    state_emitter.cleanup()
    state_emitter.enable()
    state_emitter.update(app_url=app_url, provider=provider, model_name=model_name)

    # The interpreter running this process, not whatever "streamlit" the PATH finds first: with
    # the venv not activated that is missing or the wrong file (PermissionError), and the
    # dashboard must run with the same environment as the swarm.
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", _DASHBOARD_PATH,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Ctrl-C is handled by runtime/drain.py (stop at a safe point), so the dashboard
    # only needs to be stopped when the process exits, however that happens.
    atexit.register(stop_dashboard, proc)
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
