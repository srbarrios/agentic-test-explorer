"""Tests for ``agentic_explorer.reporting`` (mission transcript -> test_report.md)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic_explorer import reporting
from agentic_explorer.reporting import (
    REPORT_TRANSCRIPT_HEAD_CHARS,
    REPORT_TRANSCRIPT_MAX_CHARS,
    append_action_tape_summary,
    bound_transcript,
    build_report_prompt,
    build_transcript,
    generate_report,
    make_report_llm,
    report_dir,
    write_report,
)


# ---------------------------------------------------------------- bound_transcript

def test_bound_transcript_leaves_short_text_alone():
    assert bound_transcript("short") == "short"


def test_bound_transcript_keeps_both_ends_and_discloses_omission():
    text = "H" * REPORT_TRANSCRIPT_HEAD_CHARS + "M" * 60_000 + "T" * 10_000

    bounded = bound_transcript(text)

    assert len(bounded) <= REPORT_TRANSCRIPT_MAX_CHARS
    assert bounded.startswith("H" * 100)
    assert bounded.endswith("T" * 100)
    assert "omitted" in bounded and "middle transcript for context budget" in bounded


# ---------------------------------------------------------------- build_transcript

def test_build_transcript_labels_roles_and_lists_tool_actions():
    messages = [
        HumanMessage(content="explore the cart"),
        AIMessage(content="clicking", tool_calls=[
            {"name": "get_dom_snapshot", "args": {}, "id": "1"},
            {"name": "execute_browser_command", "args": {}, "id": "2"},
        ]),
        ToolMessage(content="clicked", tool_call_id="2", name="execute_browser_command"),
    ]

    transcript = build_transcript(messages)

    assert transcript.splitlines() == [
        "HUMAN: explore the cart",
        "AI: clicking [Action: get_dom_snapshot, execute_browser_command]",
        "TOOL: clicked",
    ]


def test_build_transcript_skips_empty_messages_and_flattens_blocks():
    messages = [
        AIMessage(content=""),
        AIMessage(content=[{"type": "text", "text": "found "}, {"type": "text", "text": "it"}]),
    ]

    assert build_transcript(messages) == "AI: found it"


def test_build_transcript_clips_each_message_to_2000_chars():
    transcript = build_transcript([AIMessage(content="x" * 5000)])

    assert transcript.startswith("AI: ")
    assert len(transcript) - len("AI: ") <= 2000
    assert transcript.endswith("[truncated]")


# ---------------------------------------------------------------- build_report_prompt

def test_prompt_contains_mission_transcript_and_required_sections():
    prompt = build_report_prompt("test_new_user_01", "HUMAN: hi", [])

    assert "mission 'test_new_user_01'" in prompt
    assert "--- TEST TRANSCRIPT ---\nHUMAN: hi\n" in prompt
    for section in ("Mission ID & Objective", "Actions Taken", "Issues Found", "Final Status"):
        assert section in prompt
    assert "BUGS CAPTURED" not in prompt


def test_prompt_lists_bugs_clipped_to_200_chars():
    prompt = build_report_prompt("t", "x", ["Login race", "y" * 500])

    assert "BUGS CAPTURED (2):" in prompt
    assert "  - Login race" in prompt
    assert "  - " + "y" * 200 + "\n" in prompt
    assert "y" * 201 not in prompt


# ---------------------------------------------------------------- report llm

def test_make_report_llm_uses_claude_report_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(reporting, "make_llm", lambda **kw: seen.update(kw) or "llm")
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("CLAUDE_REPORT_MODEL", "claude-report")
    monkeypatch.setenv("GEMINI_REPORT_MODEL", "gemini-report")

    assert make_report_llm() == "llm"
    assert seen == {"temperature": 0, "model_name": "claude-report"}


def test_make_report_llm_falls_back_to_gemini_model_and_none(monkeypatch):
    seen = {}
    monkeypatch.setattr(reporting, "make_llm", lambda **kw: seen.update(kw) or "llm")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_REPORT_MODEL", "gemini-report")
    make_report_llm()
    assert seen["model_name"] == "gemini-report"

    monkeypatch.delenv("GEMINI_REPORT_MODEL")
    make_report_llm()
    assert seen["model_name"] is None


# ---------------------------------------------------------------- generate_report

class _FakeLLM:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0].content)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _no_sleep(_seconds):
    return None


async def test_generate_report_returns_plain_markdown():
    llm = _FakeLLM(AIMessage(content="## Report"))

    assert await generate_report("t1", "HUMAN: hi", ["bug"], llm, sleep=_no_sleep) == "## Report"
    assert "BUGS CAPTURED (1):" in llm.prompts[0]


async def test_generate_report_joins_text_blocks():
    llm = _FakeLLM(AIMessage(content=[{"type": "text", "text": "## A"}, {"type": "text", "text": "B"}]))

    assert await generate_report("t1", "x", [], llm, sleep=_no_sleep) == "## AB"


async def test_generate_report_retries_transient_errors():
    llm = _FakeLLM(Exception("503 unavailable"), AIMessage(content="ok"))

    assert await generate_report("t1", "x", [], llm, sleep=_no_sleep) == "ok"
    assert len(llm.prompts) == 2


async def test_generate_report_reports_its_own_failure_message(capsys):
    llm = _FakeLLM(*[Exception("503 unavailable")] * 5)

    with pytest.raises(Exception):
        await generate_report("t1", "x", [], llm, sleep=_no_sleep)

    assert "Report generation failed after 5 attempts." in capsys.readouterr().out


# ---------------------------------------------------------------- files

def test_report_dir_name():
    assert report_dir("mission_1") == "report_mission_1"


def test_write_report_and_append_tape_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "report_m1").mkdir()

    path = write_report("m1", "## Result")
    append_action_tape_summary("m1", tape_len=12, bugs_count=3)

    content = (tmp_path / "report_m1" / "test_report.md").read_text(encoding="utf-8")
    assert path == "report_m1/test_report.md"
    assert content.startswith("\n## Result\n\n---\n")
    assert "## Action Tape" in content
    assert "- **Recorded steps:** 12" in content
    assert "- **Bugs captured:** 3" in content
    assert "`action_tape.jsonl`" in content
    assert "reproduction_*.spec.ts" in content
    assert "Memory errors" not in content  # nothing failed, so nothing to say


def test_memory_errors_are_reported_when_there_were_any(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "report_m1").mkdir()
    write_report("m1", "## Result")

    append_action_tape_summary("m1", tape_len=12, bugs_count=3, memory_errors=2)

    content = (tmp_path / "report_m1" / "test_report.md").read_text(encoding="utf-8")
    assert "- **Memory errors:** 2" in content
    assert "learning from this run may be incomplete" in content
