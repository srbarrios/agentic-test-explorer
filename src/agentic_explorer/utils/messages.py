"""Small helpers for turning LangChain messages and long strings into prompt-safe text."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

_TRUNCATION_MARKER = " … [truncated]"
_MARKER_RESERVE = 24


def message_text(msg: BaseMessage) -> str:
    """Flatten ``msg.content`` (plain string or list of content blocks) to text.

    Only blocks that are dicts with a ``"text"`` key contribute; tool-use and
    other non-text blocks are skipped.
    """
    content = msg.content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return str(content)


def clip_text(text: str, max_chars: int) -> str:
    """Strip ``text`` and cap it for LLM routing/report prompts.

    Text over budget is cut and suffixed with `` … [truncated]``; with a tiny
    budget (``max_chars <= 24``) it is hard-cut without the marker.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= _MARKER_RESERVE:
        return text[:max(0, max_chars)]
    return text[: max_chars - _MARKER_RESERVE].rstrip() + _TRUNCATION_MARKER
