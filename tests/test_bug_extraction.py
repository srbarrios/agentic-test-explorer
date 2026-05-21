"""Coverage for ``_extract_bugs`` in ``orchestration.graph_base``.

The Visual Mode counter and final report both depend on this extractor
capturing every distinct finding an agent emits — via the structured
``<bugs_found>`` JSON tag, the ``capture_bug_screenshot`` tool call, or
explicit ``BUG FOUND: ...`` prose prefixes.
"""

import unittest

from langchain_core.messages import AIMessage

from agentic_explorer.orchestration.graph_base import (
    _extract_bugs,
    _extract_bugs_from_text,
    dedupe_bugs,
)


class ExtractBugsFromTextTests(unittest.TestCase):
    def test_parses_multi_entry_json_tag(self):
        text = (
            'Summary text. <bugs_found>['
            '{"title": "Form Validation", "summary": "shows required error on filled field"},'
            '{"title": "Empty Cart", "summary": "no guidance text"}'
            ']</bugs_found>'
        )
        bugs = _extract_bugs_from_text(text)
        self.assertEqual(len(bugs), 2)
        self.assertTrue(any("Form Validation" in b for b in bugs))
        self.assertTrue(any("Empty Cart" in b for b in bugs))

    def test_prose_fallback_catches_bug_found_prefix(self):
        text = (
            "## Findings\n\n"
            "BUG FOUND: Checkout submit button stays disabled after valid input.\n"
            "Some other narrative line.\n"
            "- **BUG**: Logout link not visible inside the burger menu.\n"
        )
        bugs = _extract_bugs_from_text(text)
        self.assertEqual(len(bugs), 2)
        self.assertTrue(any("Checkout submit" in b for b in bugs))
        self.assertTrue(any("Logout link" in b for b in bugs))

    def test_json_tag_and_prose_dedupe(self):
        """When the same bug appears both in the tag and as prose, it
        must count once — agents often restate findings in both places."""
        text = (
            'BUG FOUND: Empty cart lacks messaging on /cart page.\n'
            '<bugs_found>[{"title": "Empty cart lacks messaging", '
            '"summary": "on /cart page"}]</bugs_found>'
        )
        bugs = _extract_bugs_from_text(text)
        self.assertEqual(len(bugs), 1)

    def test_empty_array_returns_no_bugs(self):
        bugs = _extract_bugs_from_text("All clear. <bugs_found>[]</bugs_found>")
        self.assertEqual(bugs, [])

    def test_malformed_json_does_not_crash(self):
        text = '<bugs_found>[{"title": "broken,</bugs_found>'
        # Should swallow the JSON error and return whatever else it finds.
        self.assertEqual(_extract_bugs_from_text(text), [])

    def test_narrative_mention_of_bug_does_not_trigger_fallback(self):
        """The fallback is anchored to BUG: / BUG FOUND: prefixes only —
        sentences like 'this is a known bug pattern' must not be flagged."""
        text = (
            "The login flow is solid. This validation behaviour is a known bug "
            "pattern in single-page apps but is not actually broken here."
        )
        self.assertEqual(_extract_bugs_from_text(text), [])


class ExtractBugsFromMessagesTests(unittest.TestCase):
    def test_screenshot_tool_call_args_yield_bug(self):
        """Currently the ToolMessage result ('Evidence captured!') drops the
        bug_summary; the extractor must read it from the AIMessage tool_call."""
        msg = AIMessage(
            content="Capturing evidence.",
            tool_calls=[{
                "name": "capture_bug_screenshot",
                "args": {"bug_summary": "rapid_click_button_detachment"},
                "id": "call_1",
            }],
        )
        bugs = _extract_bugs([msg])
        self.assertEqual(len(bugs), 1)
        self.assertIn("rapid_click_button_detachment", bugs[0])

    def test_screenshot_and_tag_dedupe(self):
        """An agent that screenshots and then enumerates the same finding
        in the closing summary must not double-count it."""
        screenshot_msg = AIMessage(
            content="capturing.",
            tool_calls=[{
                "name": "capture_bug_screenshot",
                "args": {"bug_summary": "Form Validation Race"},
                "id": "call_1",
            }],
        )
        summary_msg = AIMessage(content=(
            '<bugs_found>[{"title": "Form Validation Race", '
            '"summary": "required error on filled field"}]</bugs_found>'
        ))
        bugs = _extract_bugs([screenshot_msg, summary_msg])
        self.assertEqual(len(bugs), 1)

    def test_multi_issue_summary_captured_in_full(self):
        """Regression: the new_user_agent trace listed 8 issues in prose
        but only one in the tag. The richer tag + fallback combination
        must capture them all."""
        summary_msg = AIMessage(content=(
            "## Findings\n\n"
            '<bugs_found>['
            '{"title": "Form validation bug", "summary": "required error on filled field"},'
            '{"title": "Missing field labels", "summary": "only aria-labels present"},'
            '{"title": "No progress indicator", "summary": "checkout has no Step X of Y"},'
            '{"title": "Empty cart lacks messaging", "summary": "no guidance text"}'
            ']</bugs_found>\n\n'
            "BUG FOUND: No tooltips on primary actions.\n"
            "BUG FOUND: No help section anywhere in the app.\n"
        ))
        bugs = _extract_bugs([summary_msg])
        self.assertEqual(len(bugs), 6)


class DedupeBugsTests(unittest.TestCase):
    def test_removes_cross_node_duplicates(self):
        """Two agents finding the same bug should collapse to one."""
        bugs = [
            "Form Validation Race: required-field error on filled field",
            "Empty Cart Lacks Messaging: no guidance text",
            "form validation race: shows required error on already-filled field",
        ]
        result = dedupe_bugs(bugs)
        self.assertEqual(len(result), 2)
        self.assertIn(bugs[0], result)
        self.assertIn(bugs[1], result)

    def test_preserves_first_seen_order(self):
        bugs = ["Zzz bug: last alphabetically", "Aaa bug: first alphabetically"]
        result = dedupe_bugs(bugs)
        self.assertEqual(result, bugs)

    def test_empty_list(self):
        self.assertEqual(dedupe_bugs([]), [])

    def test_keeps_distinct_bugs(self):
        bugs = [
            "Login redirect fails: users land on 404",
            "Login error message: wrong password shows generic error",
        ]
        result = dedupe_bugs(bugs)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
