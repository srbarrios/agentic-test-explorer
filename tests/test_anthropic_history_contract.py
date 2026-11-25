"""Why ``_sanitize_messages_for_model`` must stay.

Anthropic rejects histories where a ``tool_result`` has no matching ``tool_use``
in the turn before it, where a ``tool_use`` has no result, where a ``system``
message is not first, or where the conversation does not start with the user.
The Summarizer removes an arbitrary range of old messages, so the raw history
it leaves behind can split an AI(tool call) / Tool(result) pair.

These tests encode those documented rules as a local validator (no API call is
made, so real-API acceptance is not verified here) and show that:
  * raw histories cut by the summarizer can violate them, and
  * sanitized histories never do, for every possible cut.
"""

from itertools import combinations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_explorer.orchestration.graph_base import _sanitize_messages_for_model


def contract_violations(messages: list[BaseMessage]) -> list[str]:
    """Return the Anthropic message-ordering rules that ``messages`` would break."""
    problems: list[str] = []
    body = list(messages)

    if body and isinstance(body[0], SystemMessage):
        body = body[1:]  # a leading system prompt is sent separately and is fine
    if any(isinstance(m, SystemMessage) for m in body):
        problems.append("system message that is not first")
    if body and not isinstance(body[0], HumanMessage):
        problems.append("conversation does not start with the user")

    open_calls: set[str] = set()
    for msg in body:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id not in open_calls:
                problems.append(f"tool_result {msg.tool_call_id!r} without a preceding tool_use")
            open_calls.discard(msg.tool_call_id)
            continue
        if open_calls:
            problems.append(f"tool_use {sorted(open_calls)} without a tool_result before the next turn")
            open_calls = set()
        if isinstance(msg, AIMessage):
            open_calls = {call["id"] for call in msg.tool_calls}
    if open_calls:
        problems.append(f"tool_use {sorted(open_calls)} without a tool_result at the end")
    return problems


def _history() -> list[BaseMessage]:
    """What the outer state holds after a few agent turns."""
    return [
        HumanMessage(content="explore the cart", id="m0"),
        AIMessage(content="", id="m1", tool_calls=[{"name": "get_dom_snapshot", "args": {}, "id": "t1"}]),
        ToolMessage(content="<html>…</html>", tool_call_id="t1", name="get_dom_snapshot", id="m2"),
        AIMessage(content="found a form", id="m3"),
        HumanMessage(content="[STEP LIMIT] reset", id="m4"),
        AIMessage(content="clicking", id="m5", tool_calls=[
            {"name": "execute_browser_command", "args": {}, "id": "t2"},
            {"name": "capture_bug_screenshot", "args": {}, "id": "t3"},
        ]),
        ToolMessage(content="ok", tool_call_id="t2", name="execute_browser_command", id="m6"),
        ToolMessage(content="saved", tool_call_id="t3", name="capture_bug_screenshot", id="m7"),
        AIMessage(content="done <bugs_found>[]</bugs_found>", id="m8"),
    ]


def _cuts(history: list[BaseMessage]):
    """Every history the summarizer could leave: the mission prompt kept, a middle range removed."""
    n = len(history)
    for start, end in combinations(range(1, n + 1), 2):
        yield history[:start] + history[end:]


def test_the_validator_accepts_the_untouched_history():
    assert contract_violations(_history()) == []


def test_the_validator_rejects_an_orphaned_tool_result():
    broken = [_history()[0], *_history()[2:3]]  # Human, Tool(t1) with its tool_use removed
    assert any("without a preceding tool_use" in p for p in contract_violations(broken))


def test_raw_history_cut_by_the_summarizer_can_break_the_contract():
    broken = [cut for cut in _cuts(_history()) if contract_violations(cut)]

    assert broken, "expected at least one summarizer cut to orphan a tool call or result"


def test_sanitized_history_satisfies_the_contract_for_every_cut():
    for cut in _cuts(_history()):
        sanitized = _sanitize_messages_for_model(cut)
        assert contract_violations(sanitized) == [], [m.content for m in cut]


def test_sanitized_history_drops_system_and_tool_messages_and_keeps_text():
    history = [SystemMessage(content="sys"), *_history()]

    sanitized = _sanitize_messages_for_model(history)

    # Only text survives: the empty tool-call AI message and every ToolMessage are gone.
    assert [m.content for m in sanitized] == [
        "explore the cart", "found a form", "[STEP LIMIT] reset", "clicking",
        "done <bugs_found>[]</bugs_found>",
    ]
    assert [type(m).__name__ for m in sanitized] == [
        "HumanMessage", "AIMessage", "HumanMessage", "AIMessage", "AIMessage",
    ]
    assert not any(getattr(m, "tool_calls", None) for m in sanitized)
