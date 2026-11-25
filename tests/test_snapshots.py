"""Per-step snapshots: what the agent saw at each step, bounded, optional, referenced from the tape."""

import json
import os

import pytest

from agentic_explorer.tools.browser import engine, snapshots


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_SNAPSHOT_KEEP", raising=False)
    yield tmp_path
    engine._ACTION_TAPES.clear()


def _files(workdir, thread="t1"):
    folder = workdir / f"report_{thread}" / "snapshots"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_persists_one_file_per_step_and_returns_a_relative_ref(workdir):
    ref = snapshots.persist_snapshot("t1", "URL: https://app.example\n* [0] role=button name='Save'")
    assert ref == "snapshots/step_0001.txt"
    assert (workdir / "report_t1" / ref).read_text().startswith("URL: https://app.example")
    assert snapshots.persist_snapshot("t1", "second") == "snapshots/step_0002.txt"


def test_only_the_most_recent_snapshots_are_kept(workdir, monkeypatch):
    monkeypatch.setenv("AGENT_SNAPSHOT_KEEP", "3")
    for i in range(5):
        snapshots.persist_snapshot("t1", f"snap {i}")
    assert _files(workdir) == ["step_0003.txt", "step_0004.txt", "step_0005.txt"]


def test_zero_disables_persistence_completely(workdir, monkeypatch):
    monkeypatch.setenv("AGENT_SNAPSHOT_KEEP", "0")
    assert snapshots.persist_snapshot("t1", "anything") is None
    assert not (workdir / "report_t1").exists()


@pytest.mark.parametrize("raw", ["", "abc", "-4", "1.5"])
def test_invalid_setting_falls_back_to_the_default(workdir, monkeypatch, raw):
    monkeypatch.setenv("AGENT_SNAPSHOT_KEEP", raw)
    assert snapshots.snapshot_keep() == snapshots.DEFAULT_KEEP


def test_a_single_snapshot_is_size_bounded(workdir):
    ref = snapshots.persist_snapshot("t1", "x" * 100_000)
    text = (workdir / "report_t1" / ref).read_text()
    assert len(text) <= snapshots.MAX_SNAPSHOT_CHARS + 200
    assert "truncated" in text


def test_numbering_continues_after_a_restart(workdir):
    snapshots.persist_snapshot("t1", "a")
    snapshots.persist_snapshot("t1", "b")
    assert snapshots.persist_snapshot("t1", "c") == "snapshots/step_0003.txt"


def test_a_failing_disk_never_raises(workdir, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(snapshots.os, "makedirs", boom)
    assert snapshots.persist_snapshot("t1", "a") is None


# --- wired into the command tool ---------------------------------------------

async def test_command_tool_links_the_tape_entry_to_the_snapshot_it_showed_the_agent(workdir, monkeypatch):
    async def fake_dispatch(page, action, params):
        return "snapshot"

    async def fake_snapshot(page, limit=150):
        return {"url": "https://app.example/cart", "title": "Cart", "source": "dom",
                "elements": [{"tag": "button", "role": "button", "selector": "[data-test-subj='save']",
                              "interactive": True, "testSubj": "save"}]}

    monkeypatch.setattr(engine, "_dispatch", fake_dispatch)
    monkeypatch.setattr(engine, "extract_dom_snapshot", fake_snapshot)
    tool = engine.get_browser_command_tool(page=object())

    out = await tool.ainvoke({"command_json": '{"action": "snapshot"}'}, config={"configurable": {"thread_id": "t1"}})

    [entry] = engine.get_action_tape("t1")
    assert entry["snapshot_ref"] == "snapshots/step_0001.txt"
    saved = (workdir / "report_t1" / entry["snapshot_ref"]).read_text()
    assert saved in out  # exactly what the agent was shown
    assert "snapshot_ref" in (workdir / "report_t1" / "action_tape.jsonl").read_text()
    assert json.loads((workdir / "report_t1" / "action_tape.jsonl").read_text().splitlines()[0])["snapshot_ref"]


async def test_command_tool_without_snapshots_leaves_the_tape_entry_unchanged(workdir, monkeypatch):
    monkeypatch.setenv("AGENT_SNAPSHOT_KEEP", "0")

    async def fake_dispatch(page, action, params):
        return "snapshot"

    async def fake_snapshot(page, limit=150):
        return {"url": "u", "title": "t", "source": "dom", "elements": []}

    monkeypatch.setattr(engine, "_dispatch", fake_dispatch)
    monkeypatch.setattr(engine, "extract_dom_snapshot", fake_snapshot)
    tool = engine.get_browser_command_tool(page=object())
    await tool.ainvoke({"command_json": '{"action": "snapshot"}'}, config={"configurable": {"thread_id": "t1"}})
    [entry] = engine.get_action_tape("t1")
    assert "snapshot_ref" not in entry
    assert not os.path.exists(workdir / "report_t1" / "snapshots")
