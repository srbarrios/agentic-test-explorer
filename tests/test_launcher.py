"""Tests for ``agentic_explorer.ui.launcher`` (Streamlit dashboard lifecycle).

Everything external (subprocess, browser, sleeping) is patched so no real
dashboard starts.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from agentic_explorer.ui import launcher
from agentic_explorer.ui.launcher import start_dashboard, stop_dashboard


class _FakeProc:
    def __init__(self, exit_code=None):
        self.exit_code = exit_code
        self.terminated = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True


class _FakeEmitter:
    def __init__(self):
        self.calls = []

    def cleanup(self):
        self.calls.append(("cleanup",))

    def enable(self):
        self.calls.append(("enable",))

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))


@pytest.fixture
def env(monkeypatch):
    rec = {"popen": [], "atexit": [], "sleeps": [], "urls": [], "proc": _FakeProc()}
    rec["emitter"] = _FakeEmitter()

    def popen(cmd, **kwargs):
        rec["popen"].append((cmd, kwargs))
        return rec["proc"]

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "state_emitter", rec["emitter"])
    monkeypatch.setattr(launcher.atexit, "register", lambda fn, *a: rec["atexit"].append((fn, a)))
    monkeypatch.setattr(launcher.time, "sleep", lambda s: rec["sleeps"].append(s))
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: rec["urls"].append(url))
    return rec


def test_start_launches_streamlit_headless_and_returns_the_process(env):
    proc = start_dashboard(app_url="https://app.test", provider="claude", model_name="m")

    assert proc is env["proc"]
    cmd, kwargs = env["popen"][0]
    # Run through the current interpreter, never a "streamlit" found on PATH (see the test below).
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert Path(cmd[4]).name == "dashboard.py" and Path(cmd[4]).parent.name == "ui"
    assert cmd[5:] == ["--server.headless", "true", "--browser.gatherUsageStats", "false"]
    assert kwargs == {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


def test_start_resets_and_seeds_the_state_emitter_in_order(env):
    start_dashboard(app_url="https://app.test", provider="claude", model_name="m")

    assert env["emitter"].calls == [
        ("cleanup",),
        ("enable",),
        ("update", {"app_url": "https://app.test", "provider": "claude", "model_name": "m"}),
    ]


def test_start_waits_then_opens_the_browser(env):
    start_dashboard(app_url="u", provider="p", model_name="m")

    assert env["sleeps"] == [2]
    assert env["urls"] == ["http://localhost:8501"]


def test_start_registers_cleanup_at_exit(env):
    proc = start_dashboard(app_url="u", provider="p", model_name="m")

    assert env["atexit"] == [(stop_dashboard, (proc,))]


def test_start_leaves_signal_handling_to_the_drain_handlers(env, monkeypatch):
    # Ctrl-C now means "stop at a safe point" (runtime/drain.py). If the dashboard
    # killed itself on the first Ctrl-C, the live view would vanish mid-step.
    import signal

    installed = []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.append(sig))

    start_dashboard(app_url="u", provider="p", model_name="m")

    assert installed == []
    assert env["atexit"], "the dashboard is still stopped when the process exits"


def test_stop_terminates_a_running_dashboard():
    proc = _FakeProc(exit_code=None)
    stop_dashboard(proc)
    assert proc.terminated is True


def test_stop_ignores_an_already_finished_dashboard():
    proc = _FakeProc(exit_code=0)
    stop_dashboard(proc)
    assert proc.terminated is False


def test_stop_accepts_none():
    stop_dashboard(None)


def test_the_command_does_not_depend_on_the_path(env, monkeypatch):
    # With the venv not activated "streamlit" may be missing or be some other file on PATH.
    monkeypatch.setenv("PATH", "/nonexistent")
    start_dashboard(app_url="u", provider="p", model_name="m")
    cmd, _ = env["popen"][0]
    assert Path(cmd[0]).is_absolute() and cmd[1:3] == ["-m", "streamlit"]
