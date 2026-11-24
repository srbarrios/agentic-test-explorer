"""Characterization tests for the Supervisor and Summarizer nodes.

These pin the behaviour that existed before the LangGraph 1.2 refactor so the
upcoming changes cannot alter it silently.
"""

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
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


async def test_summarizer_replaces_old_history_with_one_summary_in_place():
    node = make_summarizer_node(keep_recent=20)
    out = await node({"messages": _history(60)})

    summary = next(m for m in out["messages"] if isinstance(m, HumanMessage))
    removed_ids = {m.id for m in out["messages"] if isinstance(m, RemoveMessage)}

    # The summary reuses the id of the first old message, so add_messages puts it
    # exactly where the old history was; the other 38 old messages are removed.
    assert summary.id == "m1"
    assert removed_ids == {f"m{i}" for i in range(2, 40)}
    assert not removed_ids & {"m0", *{f"m{i}" for i in range(40, 60)}}  # mission prompt and last 20 survive
    assert summary.content.startswith("PREVIOUS PROGRESS SUMMARY (39 messages compacted):")


async def test_summarizer_uses_the_first_old_message_that_has_an_id():
    messages = _history(60)
    messages[1] = HumanMessage(content="no id")  # first old message cannot be addressed

    out = await make_summarizer_node(keep_recent=20)({"messages": messages})

    summary = next(m for m in out["messages"] if isinstance(m, HumanMessage))
    assert summary.id == "m2"


async def test_summarizer_actually_shrinks_state_through_agent_state_reducer():
    # Regression: with operator.add the RemoveMessage entries were appended
    # instead of deleting history (60 -> 100 messages).
    graph = StateGraph(AgentState)
    graph.add_node("Summarizer", make_summarizer_node(keep_recent=20))
    graph.set_entry_point("Summarizer")
    graph.add_edge("Summarizer", END)

    out = await graph.compile().ainvoke(_state(messages=_history(60)))
    messages = out["messages"]

    assert len(messages) == 22  # mission + summary + 20 recent
    assert not any(isinstance(m, RemoveMessage) for m in messages)
    assert messages[0].id == "m0"  # mission prompt survives
    # Chronological order: mission, then the summary of what came before, then the recent messages.
    assert messages[1].content.startswith("PREVIOUS PROGRESS SUMMARY (39 messages compacted):")
    assert [m.id for m in messages[2:]] == [f"m{i}" for i in range(40, 60)]


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
            _state(), config={"recursion_limit": 100}, stream_mode="values"
        ):
            sizes.append(len(values["messages"]))
    except GraphRecursionError:
        pass  # the loop is intentionally unbounded; we only sample 100 supersteps

    assert len(sizes) >= 40
    # History grows one message per turn, so the summarizer must kick in and cut it back.
    assert any(after < before for before, after in zip(sizes, sizes[1:])), sizes
    assert max(sizes) <= 24, sizes  # mission + summary + 20 recent + what one agent turn adds


# ---------------------------------------------------------------- Agent node

class _EchoAgent:
    """Mimics ``create_agent``: streams the whole state (input + new messages), like a compiled graph."""

    def __init__(self, *new):
        self.new = list(new)
        self.inputs = []

    async def astream(self, state, config=None):
        self.inputs.append(list(state["messages"]))
        yield {"messages": list(state["messages"])}
        yield {"messages": [*state["messages"], *self.new]}


def _bug_message(title: str, msg_id: str) -> AIMessage:
    return AIMessage(
        content=f'turn done <bugs_found>[{{"title": "{title}", "summary": "s"}}]</bugs_found>', id=msg_id,
    )


async def test_agent_node_returns_only_the_messages_the_agent_produced():
    old = _bug_message("Old bug in the cart page", "a0")
    new = _bug_message("New bug in the login page", "a1")
    agent = _EchoAgent(new)
    history = [HumanMessage(content="mission", id="h0"), old]

    out = await make_agent_node(agent, name="agent", quiet=True)(_state(messages=history))

    assert [m.content for m in out["messages"]] == [new.content]


async def test_agent_node_extracts_bugs_only_from_its_own_new_messages():
    agent = _EchoAgent(_bug_message("New bug in the login page", "a1"))
    history = [HumanMessage(content="mission", id="h0"), _bug_message("Old bug in the cart page", "a0")]

    out = await make_agent_node(agent, name="agent", quiet=True)(_state(messages=history))

    # The old bug was already counted when it was produced; replaying history must not count it again.
    assert out["bugs_found"] == ["New bug in the login page: s"]


async def test_agent_node_feeds_the_agent_text_only_history():
    agent = _EchoAgent(AIMessage(content="ok", id="a1"))
    history = [
        HumanMessage(content="mission", id="h0"),
        AIMessage(content="", id="a0", tool_calls=[{"name": "get_dom_snapshot", "args": {}, "id": "t1"}]),
        ToolMessage(content="<html/>", tool_call_id="t1", name="get_dom_snapshot", id="x"),
        AIMessage(content="saw a form", id="a2"),
    ]

    await make_agent_node(agent, name="agent", quiet=True)(_state(messages=history))

    assert [m.content for m in agent.inputs[0]] == ["mission", "saw a form"]


async def test_agent_node_with_no_new_output_returns_nothing():
    out = await make_agent_node(_EchoAgent(), name="agent", quiet=True)(
        _state(messages=[HumanMessage(content="mission", id="h0"), AIMessage(content="hi", id="a0")])
    )

    assert out["messages"] == []
    assert out["bugs_found"] == []


async def test_bug_counter_counts_each_bug_once_across_turns():
    # Regression: each turn used to re-extract every earlier bug from the replayed
    # history, so 6 distinct bugs showed up as 21 entries in bugs_found.
    replies = (_bug_message(f"Bug number {i:02d} on the cart page", f"r{i}") for i in range(100))
    agent = create_agent(GenericFakeChatModel(messages=iter(replies)), tools=[], system_prompt="x")

    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(agent, name="agent", quiet=True))
    graph.add_node("Summarizer", make_summarizer_node())
    graph.set_entry_point("agent")
    graph.add_edge("agent", "Summarizer")
    graph.add_edge("Summarizer", "agent")

    counts = []
    try:
        async for values in graph.compile().astream(
            _state(), config={"recursion_limit": 24}, stream_mode="values"
        ):
            counts.append((len(values["bugs_found"]), len(set(values["bugs_found"]))))
    except GraphRecursionError:
        pass

    assert all(total == distinct for total, distinct in counts), counts
    assert counts[-1][0] >= 10  # many turns ran, one new bug each


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
