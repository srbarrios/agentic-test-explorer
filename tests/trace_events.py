"""A scripted stream that exercises every branch of the trace writer.

Used to pin the exact bytes of ``traces.log``: ``tests/fixtures/legacy_traces.log``
was captured from the code *before* the trace writer was extracted and the stream
moved to the v2 format, so any change in what gets written shows up as a diff.

Events are ``(mode, payload)`` pairs, the v1 shape; the fake graph in
``test_runner.py`` wraps them into v2 parts.
"""

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

NODE = {"langgraph_node": "new_user_agent"}


GOLDEN_EVENTS = [
    # A supervisor update without messages: no trace output at all.
    ("updates", {"Supervisor": {"next_agent": "new_user_agent", "step_count": 1}}),
    # An agent update with two messages; the first carries an `extras` block to strip.
    ("updates", {"new_user_agent": {
        "messages": [
            AIMessage(content=[{"type": "text", "text": "clicked add to cart", "extras": {"signature": "abc"}}], id="ai-1"),
            HumanMessage(content="[STEP LIMIT] reset", id="h-1"),
        ],
        "bugs_found": ["Empty cart lacks messaging: no guidance text"],
    }}),
    # Streaming chunks and system prompts are never traced.
    ("messages", (AIMessageChunk(content="partial", id="c1"), NODE)),
    ("messages", (SystemMessage(content="static system prompt", id="s1"), NODE)),
    # A new complete message, with an `extras` block to strip.
    ("messages", (
        AIMessage(content=[{"type": "text", "text": "thinking about the cart", "extras": {"signature": "zzz"}}], id="ai-2"),
        NODE,
    )),
    # The same message again: already traced, so skipped.
    ("messages", (
        AIMessage(content=[{"type": "text", "text": "thinking about the cart", "extras": {"signature": "zzz"}}], id="ai-2"),
        NODE,
    )),
    # An update whose only message was already traced: the header is still written.
    ("updates", {"new_user_agent": {"messages": [AIMessage(content="clicked add to cart", id="ai-1")]}}),
    # Messages without an id cannot be de-duplicated, so each one is written.
    ("messages", (AIMessage(content="no id yet"), NODE)),
    ("messages", (AIMessage(content="no id yet"), NODE)),
    # A node name missing from the metadata falls back to the current node.
    ("messages", (AIMessage(content="from nowhere", id="ai-3"), {})),
]
