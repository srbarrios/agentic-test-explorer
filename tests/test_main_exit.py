"""Exit behaviour of the ``agent-explorer`` entry point."""

import pytest
from langgraph.errors import GraphDrained

from agentic_explorer import main as main_module


def test_main_exits_130_and_explains_how_to_resume_after_a_drain(monkeypatch, capsys):
    async def drained():
        raise GraphDrained("sigint")

    monkeypatch.setattr(main_module, "run_missions", drained)

    with pytest.raises(SystemExit) as exit_info:
        main_module.main()

    assert exit_info.value.code == 130  # the conventional "interrupted by Ctrl-C" status
    out = capsys.readouterr().out
    assert "Stopped (sigint) at a safe point" in out
    assert "Run the same command again to resume" in out


def test_main_returns_normally_when_the_run_completes(monkeypatch):
    async def done():
        return None

    monkeypatch.setattr(main_module, "run_missions", done)

    assert main_module.main() is None


def test_main_does_not_hide_unrelated_errors(monkeypatch):
    async def broken():
        raise ValueError("App URL is not configured")

    monkeypatch.setattr(main_module, "run_missions", broken)

    with pytest.raises(ValueError, match="App URL is not configured"):
        main_module.main()
