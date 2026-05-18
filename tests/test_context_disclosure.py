import unittest
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_explorer.config import AppMeta
from agentic_explorer.main import _bound_transcript_for_report, REPORT_TRANSCRIPT_MAX_CHARS
from agentic_explorer.orchestration.graph_base import AgentState, _build_routing_context, _sanitize_messages_for_model
from agentic_explorer.pr_analyzer import (
    PRData,
    PR_GENERATED_MISSION_PROMPT_MAX_CHARS,
    _build_diff_excerpt,
    _build_human_message,
    _validate_missions,
)


class ContextDisclosureTests(unittest.TestCase):
    def test_sanitize_messages_for_model_removes_system_and_tool_messages(self):
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="mission"),
            AIMessage(content="thinking"),
            ToolMessage(content="tool output", name="execute_browser_command", tool_call_id="call-1"),
        ]

        sanitized = _sanitize_messages_for_model(messages)

        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[0], HumanMessage)
        self.assertEqual(sanitized[0].content, "mission")
        self.assertIsInstance(sanitized[1], AIMessage)
        self.assertEqual(sanitized[1].content, "thinking")

    def test_sanitize_messages_for_model_drops_non_text_content_blocks(self):
        messages = [
            HumanMessage(content=[{"type": "tool_result", "tool_use_id": "x", "content": "y"}]),
            AIMessage(content="next step"),
        ]

        sanitized = _sanitize_messages_for_model(messages)

        self.assertEqual(len(sanitized), 1)
        self.assertIsInstance(sanitized[0], AIMessage)
        self.assertEqual(sanitized[0].content, "next step")

    def test_report_transcript_at_limit_is_unchanged(self):
        transcript = "x" * REPORT_TRANSCRIPT_MAX_CHARS
        self.assertEqual(_bound_transcript_for_report(transcript), transcript)

    def test_report_transcript_is_bounded(self):
        transcript = "start\n" + ("middle\n" * 10_000) + "end"
        bounded = _bound_transcript_for_report(transcript)
        self.assertLessEqual(len(bounded), REPORT_TRANSCRIPT_MAX_CHARS)
        self.assertIn("start", bounded)
        self.assertIn("end", bounded)
        self.assertIn("omitted", bounded)

    def test_supervisor_context_uses_recent_progress_not_full_history(self):
        old = HumanMessage(content="old detail " * 500)
        recent = AIMessage(content="recent decision")
        tool = ToolMessage(content="clicked stable selector", name="execute_browser_command", tool_call_id="1")
        state = {
            "messages": [HumanMessage(content="mission objective"), old, recent, tool],
            "next_agent": "",
            "step_count": 1,
            "action_tape": [{"action": "click", "params": {"selector": "[data-test-subj='x']"}, "ok": True}],
            "bugs_found": [],
            "explored_paths": ["https://app.example/path"],
        }
        context = _build_routing_context(cast(AgentState, cast(object, state)), [])
        self.assertIn("MISSION", context)
        self.assertIn("RECENT_PROGRESS", context)
        self.assertIn("recent decision", context)
        self.assertLess(len(context), 8_000)

    def test_supervisor_context_truncates_long_bug_and_deduplicates_paths(self):
        state = {
            "messages": [HumanMessage(content="mission objective")],
            "next_agent": "",
            "step_count": 1,
            "action_tape": [],
            "bugs_found": ["bug " + ("detail " * 200)],
            "explored_paths": ["https://app.example/a", "https://app.example/a", "https://app.example/b"],
        }

        context = _build_routing_context(cast(AgentState, cast(object, state)), [])

        self.assertIn("BUGS_FOUND", context)
        self.assertIn("… [truncated]", context)
        self.assertEqual(context.count("https://app.example/a"), 1)
        self.assertIn("https://app.example/b", context)

    def test_pr_human_message_uses_diff_excerpt(self):
        huge_diff = "".join(
            f"diff --git a/file{i}.py b/file{i}.py\n@@ -1 +1 @@\n-old{i}\n+new{i}\n"
            for i in range(2_000)
        )
        pr = PRData(
            owner="o",
            repo="r",
            number=123,
            title="Change UI",
            body="body",
            diff=huge_diff,
            files_changed=[{"filename": f"file{i}.py", "additions": 1, "deletions": 1} for i in range(120)],
            url="https://github.com/o/r/pull/123",
        )
        app = AppMeta(name="App", url="https://app.example", description="desc")
        message = _build_human_message(pr, app)
        self.assertIn("Code Diff Excerpt", message)
        self.assertIn("additional files omitted", message)
        self.assertLess(len(message), len(huge_diff) + 1_000)

    def test_diff_excerpt_respects_small_budget(self):
        diff = "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n" + ("-old\n+new\n" * 5000)
        excerpt = _build_diff_excerpt(diff, budget_chars=4_000)
        self.assertIn("diff context disclosed progressively", excerpt)
        self.assertLess(len(excerpt), 5_000)

    def test_pr_mission_validation_rejects_oversized_prompt(self):
        with self.assertRaises(ValueError):
            _validate_missions(
                {
                    "missions": [
                        {
                            "thread_id": "pr_123_new_user_01",
                            "prompt": "x" * (PR_GENERATED_MISSION_PROMPT_MAX_CHARS + 1),
                        }
                    ]
                },
                123,
            )


if __name__ == "__main__":
    unittest.main()

