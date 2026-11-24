"""Tests for ``agentic_explorer.runtime.runner.run_mission``.

A fake compiled graph stands in for LangGraph: it replays scripted stream
events, can fail mid-stream, and serves the final state. The report LLM and the
retry sleep are injected so nothing waits or touches the network.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.store.memory import InMemoryStore

from agentic_explorer.memory import app_url_hash
from agentic_explorer.runtime import runner
from agentic_explorer.runtime.runner import MissionResult, run_mission

APP_URL = "https://app.test"
MISSION = {"thread_id": "test_new_user_01", "prompt": "explore the cart"}

STEP = ("updates", {"Supervisor": {"next_agent": "new_user_agent", "step_count": 1}})
AGENT_UPDATE = (
    "updates",
    {"new_user_agent": {
        "messages": [AIMessage(content="clicked add to cart", id="ai-1")],
        "bugs_found": ["Empty cart lacks messaging: no guidance text"],
        "explored_paths": ["https://app.test/cart"],
    }},
)
# A distinct id: a message already seen in an "updates" event is skipped in "messages".
THOUGHT = ("messages", (AIMessage(content="thinking about the cart", id="ai-2"), {"langgraph_node": "new_user_agent"}))


class FakeGraph:
    """``attempts`` holds one script per astream call: a list of events, optionally ending in an Exception."""

    def __init__(self, attempts, *, final_values, existing_values=None):
        self.attempts = list(attempts)
        self.final_values = final_values
        self.existing_values = existing_values or {}
        self.initial_states = []
        self.state_calls = 0

    async def aget_state(self, config):
        self.state_calls += 1
        return SimpleNamespace(values=self.existing_values if self.state_calls == 1 else self.final_values)

    async def astream(self, initial_state, *, config, stream_mode):
        self.initial_states.append(initial_state)
        assert config == {"configurable": {"thread_id": MISSION["thread_id"]}}
        assert stream_mode == ["updates", "messages"]
        for event in self.attempts.pop(0):
            if isinstance(event, Exception):
                raise event
            yield event


class FakeReportLLM:
    def __init__(self, text="## Report\nPASS"):
        self.text = text
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        return AIMessage(content=self.text)


async def _no_sleep(_seconds):
    return None


FINAL = {
    "messages": [HumanMessage(content="explore the cart"), AIMessage(content="clicked add to cart")],
    "bugs_found": ["Empty cart lacks messaging: no guidance text"],
    "explored_paths": ["https://app.test/cart"],
}


@pytest.fixture(autouse=True)
def in_tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def _run(graph, *, llm=None, store=None, **kwargs):
    llm = llm or FakeReportLLM()
    return await run_mission(
        graph, MISSION, mission_type="STANDARD", app_url=APP_URL, store=store,
        report_llm_factory=lambda: llm, sleep=_no_sleep, **kwargs,
    ), llm


# ---------------------------------------------------------------- happy path

async def test_runs_a_new_mission_and_writes_all_artifacts(in_tmp_cwd):
    graph = FakeGraph([[STEP, AGENT_UPDATE, THOUGHT]], final_values=FINAL)

    result, llm = await _run(graph)

    assert result == MissionResult(
        thread_id="test_new_user_01", bugs=["Empty cart lacks messaging: no guidance text"],
        paths=["https://app.test/cart"], tape_len=0,
    )
    assert graph.initial_states[0]["messages"][0].content == "explore the cart"
    assert {k: v for k, v in graph.initial_states[0].items() if k != "messages"} == {
        "next_agent": "", "step_count": 0, "action_tape": [], "bugs_found": [], "explored_paths": [],
    }

    out = in_tmp_cwd / "report_test_new_user_01"
    traces = (out / "traces.log").read_text(encoding="utf-8")
    assert traces.startswith("=== TRACES: test_new_user_01 ===")
    assert "STATE UPDATE FROM: new_user_agent" in traces
    assert "clicked add to cart" in traces

    report = (out / "test_report.md").read_text(encoding="utf-8")
    assert report.startswith("\n## Report\nPASS\n\n---\n")
    assert "- **Recorded steps:** 0" in report and "- **Bugs captured:** 1" in report
    assert "mission 'test_new_user_01'" in llm.prompts[0]
    assert "BUGS CAPTURED (1):" in llm.prompts[0]


async def test_existing_checkpoint_resumes_without_initial_state():
    graph = FakeGraph([[STEP]], final_values=FINAL, existing_values={"messages": [HumanMessage(content="old")]})

    await _run(graph)

    assert graph.initial_states == [None]


async def test_bugs_are_deduplicated_before_the_report():
    final = {**FINAL, "bugs_found": [
        "Empty cart lacks messaging: first wording",
        "Empty cart lacks messaging: second wording",
        "Login race: other bug",
    ]}
    graph = FakeGraph([[STEP]], final_values=final)

    result, llm = await _run(graph)

    assert len(result.bugs) == 2
    assert "BUGS CAPTURED (2):" in llm.prompts[0]


# ---------------------------------------------------------------- traces

async def test_trace_skips_chunks_and_duplicate_message_ids(in_tmp_cwd):
    chunk = ("messages", (AIMessageChunk(content="partial", id="c1"), {"langgraph_node": "new_user_agent"}))
    graph = FakeGraph([[chunk, THOUGHT, THOUGHT]], final_values=FINAL)

    await _run(graph)

    traces = (in_tmp_cwd / "report_test_new_user_01" / "traces.log").read_text(encoding="utf-8")
    assert "partial" not in traces
    assert traces.count("[new_user_agent] AIMessage") == 1


# ---------------------------------------------------------------- retries

async def test_transient_failure_retries_resuming_from_the_checkpoint(capsys):
    graph = FakeGraph(
        [[STEP, Exception("503 service unavailable")], [AGENT_UPDATE]],
        final_values=FINAL,
    )

    result, _ = await _run(graph)

    assert len(graph.initial_states) == 2
    assert graph.initial_states[0] is not None  # fresh mission
    assert graph.initial_states[1] is None  # the retry resumes from the saved checkpoint
    assert result.bugs == ["Empty cart lacks messaging: no guidance text"]
    assert "attempt 1/5" in capsys.readouterr().out


async def test_non_transient_error_propagates_and_no_report_is_written(in_tmp_cwd):
    graph = FakeGraph([[ValueError("bad graph")]], final_values=FINAL)

    with pytest.raises(ValueError, match="bad graph"):
        await _run(graph)

    assert not (in_tmp_cwd / "report_test_new_user_01" / "test_report.md").exists()


# ---------------------------------------------------------------- episodic memory

async def test_episode_memory_is_written_to_the_store():
    store = InMemoryStore()
    graph = FakeGraph([[STEP]], final_values=FINAL)

    await _run(graph, store=store)

    bugs = await store.asearch(("episodes", app_url_hash(APP_URL), "bugs"))
    assert len(bugs) == 1
    assert bugs[0].value["page"] == "/cart"
    assert bugs[0].value["sessions"] == ["test_new_user_01"]


async def test_a_failing_store_does_not_abort_the_mission():
    class BrokenStore:
        async def aput(self, *args, **kwargs):
            raise RuntimeError("disk full")

        async def aget(self, *args, **kwargs):
            raise RuntimeError("disk full")

    result, _ = await _run(FakeGraph([[STEP]], final_values=FINAL), store=BrokenStore())

    assert result.thread_id == "test_new_user_01"


# ---------------------------------------------------------------- visual mode

class _Emitter:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


async def test_visual_mode_reports_progress_to_the_emitter(monkeypatch):
    emitter = _Emitter()
    monkeypatch.setattr(runner, "state_emitter", emitter)
    graph = FakeGraph([[STEP, AGENT_UPDATE, THOUGHT]], final_values=FINAL)

    await _run(graph, visual=True)

    calls = [(name, args) for name, args, _ in emitter.calls]
    assert ("start_mission", ("test_new_user_01", "explore the cart", "STANDARD")) in calls
    assert ("add_bugs", (["Empty cart lacks messaging: no guidance text"],)) in calls
    assert ("add_paths", (["https://app.test/cart"],)) in calls
    assert ("append_thought", ("new_user_agent", "thinking about the cart")) in calls
    assert ("end_mission", ("test_new_user_01",)) in calls
    updates = [kw for name, _, kw in emitter.calls if name == "update"]
    assert {"active_node": "new_user_agent", "step_count": 1} in updates  # Supervisor routed to the agent


async def test_visual_off_never_touches_the_emitter(monkeypatch):
    emitter = _Emitter()
    monkeypatch.setattr(runner, "state_emitter", emitter)

    await _run(FakeGraph([[STEP, AGENT_UPDATE, THOUGHT]], final_values=FINAL))

    assert emitter.calls == []
