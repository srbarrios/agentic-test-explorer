"""Writer for ``report_<thread_id>/traces.log``: every message of a mission, once.

The same message reaches the stream twice (inside a node's state update and again
as a complete ``messages`` event), so each writer remembers the ids it has written.
Create one writer per stream attempt: after a retry resumes from the checkpoint
the messages of the failed attempt may legitimately be traced again.
"""

from __future__ import annotations

from typing import Iterable, Set

from langchain_core.messages import AIMessageChunk, BaseMessage, SystemMessage


def strip_extras(msg: BaseMessage) -> None:
    """Remove the noisy ``extras`` block from list-content messages, in place."""
    if isinstance(msg.content, list):
        for block in msg.content:
            if isinstance(block, dict) and "extras" in block:
                del block["extras"]


class TraceWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self._seen_ids: Set[str] = set()

    def _is_new(self, msg: BaseMessage) -> bool:
        """True the first time an id is seen; messages without an id are always new."""
        msg_id = getattr(msg, "id", None)
        if msg_id and msg_id in self._seen_ids:
            return False
        if msg_id:
            self._seen_ids.add(msg_id)
        return True

    def node_update(self, node: str, messages: Iterable[BaseMessage]) -> None:
        """Write a ``STATE UPDATE FROM`` block for ``node`` with its not-yet-traced messages.

        The header is written even if every message was already traced.
        """
        with open(self.path, "a", encoding="utf-8") as trace_file:
            trace_file.write(f"\nSTATE UPDATE FROM: {node}\n")
            for msg in messages:
                strip_extras(msg)
                if self._is_new(msg):
                    trace_file.write(msg.pretty_repr() + "\n")

    def message(self, node: str, msg: BaseMessage) -> bool:
        """Write one complete message; return whether it was new.

        Streaming chunks (the final message follows) and system prompts (static
        boilerplate) are never written, and an id already traced is skipped.
        """
        if isinstance(msg, (AIMessageChunk, SystemMessage)):
            return False
        if not self._is_new(msg):
            return False

        strip_extras(msg)
        with open(self.path, "a", encoding="utf-8") as trace_file:
            trace_file.write(f"\n[{node}] {msg.__class__.__name__}\n")
            trace_file.write(msg.pretty_repr() + "\n")
        return True
