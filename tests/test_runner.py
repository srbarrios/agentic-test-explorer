"""Tests for ``agentic_explorer.runtime.runner.run_mission``.

A fake compiled graph stands in for LangGraph: it replays scripted stream
events, can fail mid-stream, and serves the final state. The report LLM and the
retry sleep are injected so nothing waits or touches the network.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.store.memory import InMemoryStore

from agentic_explorer.memory import app_url_hash
from agentic_explorer.runtime import runner
from agentic_explorer.runtime.runner import MissionResult, run_mission, run_missions_batch

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

    def __init__(self, attempts, *, final_values, existing_values=None, thread_id=None, on_stream=None):
        self.thread_id = thread_id or MISSION["thread_id"]
        self.on_stream = on_stream  # hook run when a stream starts (e.g. to request a drain)
        self.attempts = list(attempts)
        self.final_values = final_values
        self.existing_values = existing_values or {}
        self.initial_states = []
        self.controls = []
        self.state_calls = 0

    async def aget_state(self, config):
        self.state_calls += 1
        return SimpleNamespace(values=self.existing_values if self.state_calls == 1 else self.final_values)

    async def astream(self, initial_state, *, config, stream_mode, control=None):
        self.initial_states.append(initial_state)
        self.controls.append(control)
        if self.on_stream:
            self.on_stream(control)
        assert config == {"configurable": {"thread_id": self.thread_id}}
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


# ---------------------------------------------------------------- graceful stop

async def test_the_run_control_is_passed_to_every_stream():
    control = RunControl()
    graph = FakeGraph([[STEP]], final_values=FINAL)

    await _run(graph, control=control)

    assert graph.controls == [control]


async def test_a_drained_run_propagates_without_retry_or_report(in_tmp_cwd):
    graph = FakeGraph([[STEP, GraphDrained("sigint")]], final_values=FINAL)

    with pytest.raises(GraphDrained) as drained:
        await _run(graph, control=RunControl())

    assert drained.value.reason == "sigint"
    assert len(graph.initial_states) == 1  # a drain is not a transient error: no retry
    assert not (in_tmp_cwd / "report_test_new_user_01" / "test_report.md").exists()


async def test_resuming_keeps_the_earlier_trace_and_marks_the_resume(in_tmp_cwd):
    out = in_tmp_cwd / "report_test_new_user_01"
    out.mkdir()
    (out / "traces.log").write_text("=== TRACES: test_new_user_01 ===\nEARLIER STEP\n", encoding="utf-8")
    graph = FakeGraph([[AGENT_UPDATE]], final_values=FINAL, existing_values={"messages": [HumanMessage(content="old")]})

    await _run(graph)

    traces = (out / "traces.log").read_text(encoding="utf-8")
    assert "EARLIER STEP" in traces  # a stopped mission must not lose its first half
    assert "=== RESUMED: test_new_user_01 ===" in traces
    assert traces.index("EARLIER STEP") < traces.index("=== RESUMED") < traces.index("clicked add to cart")


async def test_a_brand_new_mission_starts_a_fresh_trace(in_tmp_cwd):
    out = in_tmp_cwd / "report_test_new_user_01"
    out.mkdir()
    (out / "traces.log").write_text("STALE TRACE FROM A CLEARED RUN\n", encoding="utf-8")

    await _run(FakeGraph([[AGENT_UPDATE]], final_values=FINAL))

    traces = (out / "traces.log").read_text(encoding="utf-8")
    assert "STALE TRACE" not in traces
    assert "RESUMED" not in traces


# ---------------------------------------------------------------- batch

def _missions():
    return [
        {"thread_id": "m_standard", "prompt": "first"},
        {"thread_id": "m_advanced", "prompt": "second"},
    ]


async def _batch(graphs, missions, tiers, *, control=None, cooldown_s=0, sleeps=None):
    async def sleep(seconds):
        if sleeps is not None:
            sleeps.append(seconds)

    return await run_missions_batch(
        graphs, missions, tiers, app_url=APP_URL, store=None, control=control or RunControl(),
        cooldown_s=cooldown_s, report_llm_factory=lambda: FakeReportLLM(), sleep=sleep,
    )


async def test_batch_runs_each_mission_on_the_graph_of_its_tier():
    standard = FakeGraph([[STEP]], final_values=FINAL, thread_id="m_standard")
    advanced = FakeGraph([[STEP]], final_values=FINAL, thread_id="m_advanced")

    results = await _batch({"standard": standard, "advanced": advanced}, _missions(), ["standard", "advanced"])

    assert [r.thread_id for r in results] == ["m_standard", "m_advanced"]
    assert len(standard.initial_states) == 1 and len(advanced.initial_states) == 1


async def test_batch_labels_the_mission_type_from_its_tier(capsys):
    graphs = {"standard": FakeGraph([[STEP]], final_values=FINAL, thread_id="m_standard"),
              "advanced": FakeGraph([[STEP]], final_values=FINAL, thread_id="m_advanced")}

    await _batch(graphs, _missions(), ["standard", "advanced"])

    out = capsys.readouterr().out
    assert "MISSION [STANDARD] m_standard" in out
    assert "MISSION [ADVANCED] m_advanced" in out


async def test_batch_pauses_between_missions_but_not_after_the_last():
    graphs = {"standard": FakeGraph([[STEP]], final_values=FINAL, thread_id="m_standard"),
              "advanced": FakeGraph([[STEP]], final_values=FINAL, thread_id="m_advanced")}
    sleeps = []

    await _batch(graphs, _missions(), ["standard", "advanced"], cooldown_s=5, sleeps=sleeps)

    assert sleeps == [5]


async def test_batch_pauses_correctly_when_two_missions_are_identical():
    # The old loop compared `mission != missions[-1]`, so equal missions skipped the pause.
    twin = {"thread_id": "m_standard", "prompt": "same"}
    graph = FakeGraph([[STEP], [STEP]], final_values=FINAL, thread_id="m_standard")
    sleeps = []

    await _batch({"standard": graph}, [twin, dict(twin)], ["standard", "standard"], cooldown_s=3, sleeps=sleeps)

    assert sleeps == [3]


async def test_batch_without_cooldown_never_sleeps():
    graphs = {"standard": FakeGraph([[STEP]], final_values=FINAL, thread_id="m_standard"),
              "advanced": FakeGraph([[STEP]], final_values=FINAL, thread_id="m_advanced")}
    sleeps = []

    await _batch(graphs, _missions(), ["standard", "advanced"], cooldown_s=0, sleeps=sleeps)

    assert sleeps == []


async def test_batch_stops_before_the_next_mission_when_a_drain_was_requested(capsys):
    control = RunControl()
    standard = FakeGraph(
        [[STEP]], final_values=FINAL, thread_id="m_standard",
        on_stream=lambda ctl: ctl.request_drain("sigint"),  # Ctrl-C arrives during the first mission
    )
    advanced = FakeGraph([[STEP]], final_values=FINAL, thread_id="m_advanced")
    sleeps = []

    with pytest.raises(GraphDrained) as drained:
        await _batch(
            {"standard": standard, "advanced": advanced}, _missions(), ["standard", "advanced"],
            control=control, cooldown_s=5, sleeps=sleeps,
        )

    assert drained.value.reason == "sigint"
    assert advanced.initial_states == []  # the second mission never started
    assert sleeps == []  # and we did not wait out the cooldown first


async def test_batch_does_not_start_when_a_drain_was_already_requested():
    control = RunControl()
    control.request_drain("sigterm")
    graph = FakeGraph([[STEP]], final_values=FINAL, thread_id="m_standard")

    with pytest.raises(GraphDrained):
        await _batch({"standard": graph}, _missions()[:1], ["standard"], control=control)

    assert graph.initial_states == []


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
