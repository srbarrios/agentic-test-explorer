"""Characterization tests for the Supervisor and Summarizer nodes.

These pin the behaviour that existed before the LangGraph 1.2 refactor so the
upcoming changes cannot alter it silently.
"""

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from agentic_explorer.orchestration.graph_base import (
    AgentState,
    make_agent_node,
    make_summarizer_node,
    make_supervisor_node,
)

AGENTS = ("new_user_agent", "power_user_agent")


def _history(n: int) -> list:
    """First message is the mission prompt; the rest alternate Human/AI. All have ids."""
    msgs = [HumanMessage(content="mission prompt", id="m0")]
    for i in range(1, n):
        cls = AIMessage if i % 2 else HumanMessage
        msgs.append(cls(content=f"msg {i}", id=f"m{i}"))
    return msgs


def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage(content="mission prompt")],
        "next_agent": "",
        "action_tape": [],
        "step_count": 0,
        "bugs_found": [],
        "explored_paths": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- Summarizer

async def test_summarizer_is_noop_when_history_is_short():
    node = make_summarizer_node(keep_recent=20)
    assert await node({"messages": _history(21)}) == {}


async def test_summarizer_is_noop_when_messages_have_no_ids():
    node = make_summarizer_node(keep_recent=20)
    msgs = [HumanMessage(content=f"m{i}") for i in range(60)]
    assert await node({"messages": msgs}) == {}


async def test_summarizer_keeps_mission_prompt_and_last_20():
    node = make_summarizer_node(keep_recent=20)
    out = await node({"messages": _history(60)})

    removals = [m for m in out["messages"] if isinstance(m, RemoveMessage)]
    summary = out["messages"][-1]

    removed_ids = {m.id for m in removals}
    assert removed_ids == {f"m{i}" for i in range(1, 40)}  # 39 old messages
    assert "m0" not in removed_ids  # mission prompt survives
    assert not removed_ids & {f"m{i}" for i in range(40, 60)}  # last 20 survive
    assert isinstance(summary, HumanMessage)
    assert summary.content.startswith("PREVIOUS PROGRESS SUMMARY (39 messages compacted):")


async def test_summarizer_actually_shrinks_state_through_agent_state_reducer():
    # Regression: with operator.add the RemoveMessage entries were appended
    # instead of deleting history (60 -> 100 messages).
    graph = StateGraph(AgentState)
    graph.add_node("Summarizer", make_summarizer_node(keep_recent=20))
    graph.set_entry_point("Summarizer")
    graph.add_edge("Summarizer", END)

    out = await graph.compile().ainvoke(_state(messages=_history(60)))
    messages = out["messages"]

    assert len(messages) == 22  # mission + 20 recent + 1 summary
    assert not any(isinstance(m, RemoveMessage) for m in messages)
    assert messages[0].id == "m0"  # mission prompt survives
    assert [m.id for m in messages[1:21]] == [f"m{i}" for i in range(40, 60)]
    assert messages[-1].content.startswith("PREVIOUS PROGRESS SUMMARY (39 messages compacted):")


async def test_agent_summarizer_loop_keeps_history_bounded():
    # Regression: with operator.add this loop grew to >6000 messages in 16 steps
    # (each agent turn re-emits the inner history and nothing was ever removed).
    llm = GenericFakeChatModel(messages=iter(AIMessage(content=f"reply {i}") for i in range(200)))
    agent = create_agent(llm, tools=[], system_prompt="x")

    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(agent, name="agent", quiet=True))
    graph.add_node("Summarizer", make_summarizer_node())
    graph.set_entry_point("agent")
    graph.add_edge("agent", "Summarizer")
    graph.add_edge("Summarizer", "agent")

    sizes = []
    try:
        async for values in graph.compile().astream(
            _state(), config={"recursion_limit": 24}, stream_mode="values"
        ):
            sizes.append(len(values["messages"]))
    except GraphRecursionError:
        pass  # the loop is intentionally unbounded; we only sample 24 supersteps

    assert len(sizes) >= 20
    assert max(sizes) <= 50, sizes


# ---------------------------------------------------------------- Supervisor

class _FakeRoutingLLM:
    """Stands in for a chat model: with_structured_output(...) returns self."""

    def __init__(self, choice: str):
        self.choice = choice
        self.calls: list = []

    def with_structured_output(self, **_kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return {"next": self.choice}


async def test_supervisor_routes_and_increments_step():
    llm = _FakeRoutingLLM("power_user_agent")
    node = make_supervisor_node(llm, AGENTS, "https://app.test", max_steps=30)

    out = await node(_state(step_count=3))

    assert out == {"next_agent": "power_user_agent", "step_count": 4}


async def test_supervisor_passes_finish_through():
    node = make_supervisor_node(_FakeRoutingLLM("FINISH"), AGENTS, "https://app.test", max_steps=30)

    out = await node(_state())

    assert out["next_agent"] == "FINISH"


async def test_supervisor_resets_after_max_steps():
    node = make_supervisor_node(_FakeRoutingLLM("new_user_agent"), AGENTS, "https://app.test", max_steps=5)

    out = await node(_state(step_count=5, bugs_found=["Login race: something"]))

    assert out["step_count"] == 1
    reset = out["messages"][0]
    assert isinstance(reset, HumanMessage)
    assert reset.content.startswith("[STEP LIMIT")
    assert "https://app.test" in reset.content
    assert "Bugs discovered so far: 1" in reset.content


async def test_supervisor_routing_prompt_contains_compact_context():
    llm = _FakeRoutingLLM("new_user_agent")
    node = make_supervisor_node(llm, AGENTS, "https://app.test", max_steps=30)

    await node(_state(
        bugs_found=["Empty cart lacks messaging"],
        explored_paths=["https://app.test/cart"],
    ))

    system, request = llm.calls[0]
    assert "QA Orchestrator" in system.content
    for section in ("MISSION:", "RECENT_PROGRESS:", "BUGS_FOUND:", "EXPLORED_URLS:"):
        assert section in request.content
    assert "Empty cart lacks messaging" in request.content
    assert "https://app.test/cart" in request.content
