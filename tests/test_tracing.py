"""Tests for ``agentic_explorer.runtime.tracing.TraceWriter`` (the ``traces.log`` writer).

The exact bytes for a whole run are pinned in ``test_runner.py``; these tests pin
each rule on its own.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from agentic_explorer.runtime.tracing import TraceWriter, strip_extras


@pytest.fixture
def log(tmp_path):
    return tmp_path / "traces.log"


def _text(path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- node updates

def test_node_update_writes_a_header_then_each_message(log):
    ai = AIMessage(content="hi", id="a1")
    human = HumanMessage(content="yo", id="h1")

    TraceWriter(str(log)).node_update("agent", [ai, human])

    assert _text(log) == f"\nSTATE UPDATE FROM: agent\n{ai.pretty_repr()}\n{human.pretty_repr()}\n"


def test_node_update_skips_messages_already_traced(log):
    seen = AIMessage(content="old", id="a1")
    new = AIMessage(content="new", id="a2")
    writer = TraceWriter(str(log))
    writer.node_update("agent", [seen])

    writer.node_update("agent", [seen, new])

    text = _text(log)
    assert text.count("old") == 1
    assert text.count("new") == 1


def test_node_update_writes_the_header_even_when_everything_is_a_duplicate(log):
    msg = AIMessage(content="old", id="a1")
    writer = TraceWriter(str(log))
    writer.node_update("agent", [msg])

    writer.node_update("agent", [msg])

    assert _text(log).count("STATE UPDATE FROM: agent") == 2
    assert _text(log).count("old") == 1


def test_node_update_without_messages_writes_only_the_header(log):
    TraceWriter(str(log)).node_update("Supervisor", [])

    assert _text(log) == "\nSTATE UPDATE FROM: Supervisor\n"


# ---------------------------------------------------------------- single messages

def test_message_writes_the_node_and_type_header_and_returns_true(log):
    msg = AIMessage(content="thinking", id="a1")

    assert TraceWriter(str(log)).message("agent", msg) is True

    assert _text(log) == f"\n[agent] AIMessage\n{msg.pretty_repr()}\n"


def test_message_skips_streaming_chunks_and_system_prompts(log):
    writer = TraceWriter(str(log))

    assert writer.message("agent", AIMessageChunk(content="part", id="c1")) is False
    assert writer.message("agent", SystemMessage(content="static", id="s1")) is False

    assert not log.exists()


def test_message_skips_a_duplicate_id(log):
    writer = TraceWriter(str(log))
    assert writer.message("agent", AIMessage(content="once", id="a1")) is True

    assert writer.message("agent", AIMessage(content="once", id="a1")) is False

    assert _text(log).count("once") == 1


def test_dedupe_is_shared_between_updates_and_messages(log):
    writer = TraceWriter(str(log))
    writer.node_update("agent", [AIMessage(content="shared", id="a1")])

    assert writer.message("agent", AIMessage(content="shared", id="a1")) is False


def test_messages_without_an_id_cannot_be_deduplicated(log):
    writer = TraceWriter(str(log))

    assert writer.message("agent", AIMessage(content="no id")) is True
    assert writer.message("agent", AIMessage(content="no id")) is True

    assert _text(log).count("no id") == 2


# ---------------------------------------------------------------- extras and files

def test_strip_extras_removes_the_noisy_block_in_place():
    msg = AIMessage(content=[
        {"type": "text", "text": "hello", "extras": {"signature": "abc"}},
        "plain string block",
        {"type": "text", "text": "no extras here"},
    ])

    strip_extras(msg)

    assert msg.content == [
        {"type": "text", "text": "hello"},
        "plain string block",
        {"type": "text", "text": "no extras here"},
    ]


def test_strip_extras_ignores_plain_string_content():
    msg = AIMessage(content="just text")

    strip_extras(msg)

    assert msg.content == "just text"


def test_traced_messages_never_contain_extras(log):
    msg = AIMessage(content=[{"type": "text", "text": "hi", "extras": {"signature": "SECRET"}}], id="a1")

    TraceWriter(str(log)).message("agent", msg)

    assert "SECRET" not in _text(log)


def test_each_writer_starts_with_an_empty_seen_set(log):
    # One writer per stream attempt: after a retry the same message is traced again.
    msg = AIMessage(content="again", id="a1")
    TraceWriter(str(log)).message("agent", msg)

    assert TraceWriter(str(log)).message("agent", msg) is True

    assert _text(log).count("again") == 2


def test_the_writer_appends_and_never_truncates(log):
    log.write_text("=== TRACES: t ===\n", encoding="utf-8")

    TraceWriter(str(log)).message("agent", AIMessage(content="x", id="a1"))

    assert _text(log).startswith("=== TRACES: t ===\n")
