"""Mission report: transcript -> prompt -> LLM -> ``report_<thread_id>/test_report.md``."""

from __future__ import annotations

import os
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from agentic_explorer.orchestration.findings import find_evidence, one_line
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


def build_report_prompt(
    thread_id: str, transcript: str, bugs: Sequence[str], uncertain: Sequence[str] = (),
) -> str:
    bugs_section = ""
    if bugs:
        bugs_section = (
            f"\n\nBUGS CAPTURED ({len(bugs)}):\n"
            + "\n".join(f"  - {b[:_BUG_CLIP_CHARS]}" for b in bugs)
        )
    if uncertain:
        bugs_section += (
            f"\n\nNOT DETERMINED ({len(uncertain)}) — the agent could not confirm these; "
            "do not report these as confirmed bugs, mention them only as open questions:\n"
            + "\n".join(f"  - {b[:_BUG_CLIP_CHARS]}" for b in uncertain)
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
    uncertain: Sequence[str] = (),
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
) -> str:
    """Ask ``llm`` for the Markdown report, retrying transient API errors."""
    prompt = HumanMessage(content=build_report_prompt(thread_id, transcript, bugs, uncertain))
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


def _finding_lines(index: int, bug: str, record: Optional[Dict[str, str]]) -> List[str]:
    record = record or {}
    lines = [f"{index}. **{bug}**"]
    lines.append(f"   - Observed: {one_line(record['observed'])}" if record.get("observed")
                 else "   - Observed: no observed value recorded")
    if record.get("expected"):
        lines.append(f"   - Expected: {one_line(record['expected'])}")
    where = record.get("observed_at") or record.get("location")
    if where:
        lines.append(f"   - Where: {one_line(where)}")
    if record.get("severity"):
        lines.append(f"   - Severity: {record['severity']}")
    return lines


def render_findings_section(
    confirmed: Sequence[str], uncertain: Sequence[str], evidence: Sequence[Dict[str, str]],
) -> str:
    """Markdown for the findings: confirmed bugs with their evidence, then a separate "Not determined" list.

    Rendered by code, not by the LLM, so the evidence is exactly what the agent recorded.
    Returns an empty string when there is nothing to show.
    """
    parts: List[str] = []
    if confirmed:
        lines = [f"## Confirmed bugs ({len(confirmed)})", ""]
        for i, bug in enumerate(confirmed, 1):
            lines += _finding_lines(i, bug, find_evidence(bug, evidence))
        parts.append("\n".join(lines))
    if uncertain:
        lines = [
            f"## Not determined ({len(uncertain)})",
            "",
            "The agent could not confirm these. They are not counted as bugs.",
            "",
        ]
        for i, bug in enumerate(uncertain, 1):
            lines += _finding_lines(i, bug, find_evidence(bug, evidence))
        parts.append("\n".join(lines))
    return ("\n\n".join(parts) + "\n") if parts else ""


def append_findings_section(
    thread_id: str, confirmed: Sequence[str], uncertain: Sequence[str], evidence: Sequence[Dict[str, str]],
) -> None:
    """Append the findings (see ``render_findings_section``) to ``test_report.md``."""
    section = render_findings_section(confirmed, uncertain, evidence)
    if not section:
        return
    path = os.path.join(report_dir(thread_id), "test_report.md")
    with open(path, "a", encoding="utf-8") as report_file:
        report_file.write("\n" + section)


def append_action_tape_summary(
    thread_id: str, *, tape_len: int, bugs_count: int, memory_errors: int = 0, uncertain_count: int = 0,
) -> None:
    path = os.path.join(report_dir(thread_id), "test_report.md")
    uncertain_line = (
        f"- **Not determined:** {uncertain_count} (listed apart, not counted as bugs)\n" if uncertain_count else ""
    )
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
            f"{uncertain_line}"
            f"{memory_line}"
            f"- **Tape log:** `action_tape.jsonl`\n"
            f"- **Reproductions:** any `reproduction_*.spec.ts` in this folder can be run with `npx playwright test`.\n"
        )
