"""Mission report: transcript -> prompt -> LLM -> ``report_<thread_id>/test_report.md``."""

from __future__ import annotations

import os
from typing import Awaitable, Callable, List, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from agentic_explorer.runtime.retry import with_retry
from agentic_explorer.utils.llm import make_llm
from agentic_explorer.utils.messages import clip_text, message_text

REPORT_TRANSCRIPT_MAX_CHARS = 35_000
REPORT_TRANSCRIPT_HEAD_CHARS = 6_000
_MESSAGE_CLIP_CHARS = 2_000
_BUG_CLIP_CHARS = 200


def report_dir(thread_id: str) -> str:
    """Directory holding every artifact of a mission."""
    return f"report_{thread_id}"


def bound_transcript(transcript: str) -> str:
    """Keep report context within a predictable prompt budget.

    Reports need the mission start plus the latest outcomes most. Preserve both
    ends and disclose how much was omitted instead of sending unbounded history.
    """
    if len(transcript) <= REPORT_TRANSCRIPT_MAX_CHARS:
        return transcript
    tail_chars = REPORT_TRANSCRIPT_MAX_CHARS - REPORT_TRANSCRIPT_HEAD_CHARS - 160
    omitted = len(transcript) - REPORT_TRANSCRIPT_HEAD_CHARS - tail_chars
    return (
        transcript[:REPORT_TRANSCRIPT_HEAD_CHARS].rstrip()
        + f"\n\n... [omitted {omitted:,} chars of middle transcript for context budget] ...\n\n"
        + transcript[-tail_chars:].lstrip()
    )


def build_transcript(messages: Sequence[BaseMessage]) -> str:
    """One ``ROLE: text [Action: tools]`` line per non-empty message, bounded for the prompt."""
    lines: List[str] = []
    for msg in messages:
        text = message_text(msg)
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            text += f" [Action: {', '.join(tc['name'] for tc in tool_calls)}]"
        if text.strip():
            lines.append(f"{msg.type.upper()}: {clip_text(text, _MESSAGE_CLIP_CHARS)}")
    return bound_transcript("\n".join(lines))


def build_report_prompt(thread_id: str, transcript: str, bugs: Sequence[str]) -> str:
    bugs_section = ""
    if bugs:
        bugs_section = (
            f"\n\nBUGS CAPTURED ({len(bugs)}):\n"
            + "\n".join(f"  - {b[:_BUG_CLIP_CHARS]}" for b in bugs)
        )
    return (
        f"You are the Lead QA Engineer. Review the following agent test transcript for mission '{thread_id}'.\n\n"
        f"--- TEST TRANSCRIPT ---\n{transcript}{bugs_section}\n-----------------------\n\n"
        "Write a concise report formatted in Markdown. Include ONLY these sections:\n"
        "- **Mission ID & Objective**\n"
        "- **Actions Taken** (Brief summary)\n"
        "- **Issues Found** (Any UI errors, visual anomalies, or tool failures)\n"
        "- **Final Status** (PASS or FAIL based on whether the objective was achieved)\n\n"
        "Output ONLY plain Markdown text."
    )


def make_report_llm():
    """LLM for reports; ``CLAUDE_REPORT_MODEL`` / ``GEMINI_REPORT_MODEL`` override the default model."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    model = (
        os.getenv("CLAUDE_REPORT_MODEL") if provider == "claude"
        else os.getenv("GEMINI_REPORT_MODEL")
    ) or None
    return make_llm(temperature=0, model_name=model)


async def generate_report(
    thread_id: str,
    transcript: str,
    bugs: Sequence[str],
    llm,
    *,
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
) -> str:
    """Ask ``llm`` for the Markdown report, retrying transient API errors."""
    prompt = HumanMessage(content=build_report_prompt(thread_id, transcript, bugs))
    response = await with_retry(
        lambda: llm.ainvoke([prompt]),
        fail_message="Report generation failed after {max_attempts} attempts.",
        sleep=sleep,
    )
    return message_text(response)


def write_report(thread_id: str, text: str) -> str:
    """Write the LLM report and return its path."""
    path = os.path.join(report_dir(thread_id), "test_report.md")
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write(f"\n{text}\n\n---\n")
    return path


def append_action_tape_summary(
    thread_id: str, *, tape_len: int, bugs_count: int, memory_errors: int = 0,
) -> None:
    path = os.path.join(report_dir(thread_id), "test_report.md")
    memory_line = (
        f"- **Memory errors:** {memory_errors} — learning from this run may be incomplete "
        "(the console warned about each kind; configure logging for `agentic_explorer.memory` for details).\n"
        if memory_errors else ""
    )
    with open(path, "a", encoding="utf-8") as report_file:
        report_file.write(
            f"\n## Action Tape\n\n"
            f"- **Recorded steps:** {tape_len}\n"
            f"- **Bugs captured:** {bugs_count}\n"
            f"{memory_line}"
            f"- **Tape log:** `action_tape.jsonl`\n"
            f"- **Reproductions:** any `reproduction_*.spec.ts` in this folder can be run with `npx playwright test`.\n"
        )
