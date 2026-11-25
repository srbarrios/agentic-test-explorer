"""Node-level resilience: hung nodes time out, recover, or become a message.

The graphs here use the real ``compile_swarm`` wiring (agent -> Summarizer ->
Supervisor) with fake nodes, so what is tested is the production topology plus
the policies from ``orchestration.resilience``.
"""

import asyncio

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained, NodeTimeoutError
from langgraph.graph import StateGraph
from langgraph.runtime import RunControl

from agentic_explorer.config import RuntimeConfig
from agentic_explorer.orchestration import graph_base
from agentic_explorer.orchestration.graph_base import AgentState, compile_swarm
from agentic_explorer.orchestration.resilience import (
    agent_node_options,
    supervisor_node_options,
)


def _initial_state() -> dict:
    return {
        "messages": [HumanMessage(content="mission")],
        "next_agent": "", "action_tape": [], "step_count": 0,
        "bugs_found": [], "explored_paths": [],
    }


def _swarm(agent, runtime: RuntimeConfig):
    """Supervisor routes to ``agent`` once, then finishes."""
    calls = {"supervisor": 0}

    async def supervisor(state):
        calls["supervisor"] += 1
        return {"next_agent": "agent" if calls["supervisor"] == 1 else "FINISH", "step_count": calls["supervisor"]}

    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor, **supervisor_node_options(runtime))
    workflow.add_node("agent", agent, **agent_node_options(runtime))
    graph = compile_swarm(workflow, {"agent": None}, InMemorySaver())
    return graph, calls


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def _texts(out) -> list:
    return [m.content for m in out["messages"]]


# ---------------------------------------------------------------- policies from config

def test_default_runtime_uses_an_idle_timeout_and_one_retry():
    options = agent_node_options(RuntimeConfig())

    assert options["timeout"].idle_timeout == 300.0
    assert options["timeout"].run_timeout is None
    assert options["retry_policy"].max_attempts == 2  # 1 retry = 2 attempts
    assert options["error_handler"] is not None


def test_run_timeout_is_set_only_when_configured():
    options = agent_node_options(RuntimeConfig(node_idle_timeout_s=0, node_run_timeout_s=60))

    assert options["timeout"].idle_timeout is None
    assert options["timeout"].run_timeout == 60.0


def test_zero_timeouts_disable_timeout_and_retry():
    options = agent_node_options(RuntimeConfig(node_idle_timeout_s=0, node_run_timeout_s=0))

    assert options["timeout"] is None
    assert options["retry_policy"] is None  # nothing can time out, so nothing to retry
    assert options["error_handler"] is not None  # other failures are still handled


def test_zero_retries_disable_the_retry_policy():
    assert agent_node_options(RuntimeConfig(node_retries=0))["retry_policy"] is None


def test_supervisor_gets_timeout_and_retry_but_no_error_handler():
    options = supervisor_node_options(RuntimeConfig())

    assert options["timeout"].idle_timeout == 300.0
    assert options["retry_policy"].max_attempts == 2
    assert "error_handler" not in options


def test_only_a_timeout_is_retried_by_the_node_policy():
    retry_on = agent_node_options(RuntimeConfig())["retry_policy"].retry_on

    assert retry_on(NodeTimeoutError("agent", 1.0, kind="idle", idle_timeout=1.0)) is True
    # API errors are left to the mission-level retry, which waits long enough for quotas.
    assert retry_on(Exception("429 rate limit")) is False
    assert retry_on(ValueError("bad")) is False


# ---------------------------------------------------------------- behaviour in a graph

async def test_a_hung_agent_times_out_and_the_run_goes_on():
    async def hang(state):
        await asyncio.sleep(30)

    graph, calls = _swarm(hang, RuntimeConfig(node_idle_timeout_s=0.3, node_retries=0))

    out = await asyncio.wait_for(graph.ainvoke(_initial_state(), _config("hang")), timeout=10)

    failed = [t for t in _texts(out) if t.startswith("[NODE FAILED]")]
    assert len(failed) == 1
    assert "agent" in failed[0] and "NodeTimeoutError" in failed[0]
    assert calls["supervisor"] == 2  # the supervisor got control back and could finish


async def test_a_busy_agent_is_not_timed_out_while_it_keeps_working():
    async def slow_step(x):
        await asyncio.sleep(0.12)
        return x

    chain = RunnableLambda(slow_step)

    async def busy(state, config):
        for _ in range(5):  # 0.6 s in total, but never 0.3 s without activity
            await chain.ainvoke(1, config)
        return {"explored_paths": ["https://app.test/done"]}

    graph, _ = _swarm(busy, RuntimeConfig(node_idle_timeout_s=0.3, node_retries=0))

    out = await graph.ainvoke(_initial_state(), _config("busy"))

    assert out["explored_paths"] == ["https://app.test/done"]
    assert not any(t.startswith("[NODE FAILED]") for t in _texts(out))


async def test_a_timed_out_agent_is_retried_and_can_recover():
    attempts = {"n": 0}

    async def slow_then_fast(state):
        attempts["n"] += 1
        if attempts["n"] == 1:
            await asyncio.sleep(30)
        return {"explored_paths": ["https://app.test/recovered"]}

    graph, _ = _swarm(slow_then_fast, RuntimeConfig(node_idle_timeout_s=0.3, node_retries=1))

    out = await asyncio.wait_for(graph.ainvoke(_initial_state(), _config("retry")), timeout=10)

    assert attempts["n"] == 2
    assert out["explored_paths"] == ["https://app.test/recovered"]
    assert not any(t.startswith("[NODE FAILED]") for t in _texts(out))


async def test_a_transient_api_error_is_not_swallowed():
    async def overloaded(state):
        raise Exception("Error code: 529 - overloaded_error")

    graph, calls = _swarm(overloaded, RuntimeConfig())

    # It must reach the mission-level retry, which resumes from the checkpoint and waits.
    with pytest.raises(Exception, match="529"):
        await graph.ainvoke(_initial_state(), _config("transient"))

    assert calls["supervisor"] == 1


async def test_a_non_transient_failure_becomes_a_message_for_the_supervisor():
    async def broken(state):
        raise KeyError("missing selector " + "x" * 600)

    graph, calls = _swarm(broken, RuntimeConfig())

    out = await graph.ainvoke(_initial_state(), _config("broken"))

    failed = [t for t in _texts(out) if t.startswith("[NODE FAILED]")]
    assert len(failed) == 1
    assert "KeyError" in failed[0] and "missing selector" in failed[0]
    assert "different approach" in failed[0]
    assert len(failed[0]) < 600  # the error text is clipped
    assert calls["supervisor"] == 2


async def test_failure_is_routed_through_the_summarizer(monkeypatch):
    visited = []
    original = graph_base.make_summarizer_node

    def spying_summarizer(**kwargs):
        node = original(**kwargs)

        async def spy(state):
            visited.append("Summarizer")
            return await node(state)

        return spy

    monkeypatch.setattr(graph_base, "make_summarizer_node", spying_summarizer)

    async def broken(state):
        raise RuntimeError("boom")

    graph, _ = _swarm(broken, RuntimeConfig())
    await graph.ainvoke(_initial_state(), _config("via-summarizer"))

    assert visited, "the recovery path should keep compacting the history"


# ---------------------------------------------------------------- graceful stop

async def test_drain_stops_at_a_safe_point_and_resume_does_not_redo_the_agent():
    control = RunControl()
    runs = {"agent": 0}

    async def agent(state):
        runs["agent"] += 1
        control.request_drain("sigint")  # as if Ctrl-C arrived during this step
        return {"bugs_found": ["Login race: filled field flagged"], "explored_paths": ["https://app.test/login"]}

    graph, calls = _swarm(agent, RuntimeConfig())

    with pytest.raises(GraphDrained) as drained:
        await graph.ainvoke(_initial_state(), _config("drain"), control=control)

    assert drained.value.reason == "sigint"
    snapshot = await graph.aget_state(_config("drain"))
    assert snapshot.next, "work remains, so the mission can be resumed"
    assert snapshot.values["bugs_found"] == ["Login race: filled field flagged"]  # nothing was lost

    out = await graph.ainvoke(None, _config("drain"))  # running the same command again

    assert runs["agent"] == 1  # the finished step was not repeated
    assert out["bugs_found"] == ["Login race: filled field flagged"]  # and nothing was duplicated
    assert out["explored_paths"] == ["https://app.test/login"]
    assert calls["supervisor"] == 2  # the supervisor resumed and finished the mission


async def test_without_a_drain_request_the_run_completes_normally():
    async def agent(state):
        return {"explored_paths": ["https://app.test/home"]}

    graph, _ = _swarm(agent, RuntimeConfig())

    out = await graph.ainvoke(_initial_state(), _config("no-drain"), control=RunControl())

    assert out["explored_paths"] == ["https://app.test/home"]
