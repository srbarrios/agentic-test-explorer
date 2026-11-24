"""Tests for the shared message helpers in ``agentic_explorer.utils.messages``.

These replace the copies that used to live in ``main.py`` (``_message_text``,
``_clip_for_prompt``) and ``orchestration/graph_base.py`` (``_msg_text``,
``_clip_text``).
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agentic_explorer.utils.messages import clip_text, message_text


def test_message_text_returns_plain_string_content():
    assert message_text(HumanMessage(content="hello")) == "hello"


def test_message_text_joins_text_blocks_and_skips_others():
    msg = AIMessage(content=[
        {"type": "text", "text": "Found "},
        {"type": "tool_use", "id": "t1", "name": "x", "input": {}},  # no "text" key
        {"type": "text", "text": "a bug"},
        "stray string",  # non-dict blocks are ignored
    ])
    assert message_text(msg) == "Found a bug"


def test_message_text_empty_block_list_is_empty_string():
    assert message_text(AIMessage(content=[])) == ""


def test_clip_text_returns_short_text_unchanged_but_stripped():
    assert clip_text("  short  ", 100) == "short"


def test_clip_text_truncates_with_marker():
    clipped = clip_text("x" * 100, 40)
    assert clipped == "x" * 16 + " … [truncated]"


@pytest.mark.parametrize("max_chars, expected", [(10, "x" * 10), (24, "x" * 24), (0, "")])
def test_clip_text_hard_cuts_when_budget_is_tiny(max_chars, expected):
    assert clip_text("x" * 100, max_chars) == expected
