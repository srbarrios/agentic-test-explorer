import os
import tempfile
import unittest

from agentic_explorer.tools.browser import engine


class BrowserEngineTests(unittest.TestCase):
    def test_validate_selector_allows_resilient_selectors(self):
        stable_selectors = [
            "[data-test-subj='submitButton']",
            "[aria-label='Search']",
            "button:has-text('Save')",
            "role=button[name='Submit']",
            "text='Apply'",
        ]

        for selector in stable_selectors:
            with self.subTest(selector=selector):
                self.assertIsNone(engine._validate_selector(selector))

    def test_validate_selector_rejects_xpath_and_positional_css(self):
        brittle_selectors = [
            "//button[@type='submit']",
            "/html/body/div/button",
            "button:nth-child(2)",
            "li:nth-of-type(3)",
            "main > div > span[data-test-subj='late']",
        ]

        for selector in brittle_selectors:
            with self.subTest(selector=selector):
                error = engine._validate_selector(selector)
                self.assertIsNotNone(error)
                self.assertIn("BRITTLE_SELECTOR_REJECTED", error)

    def test_ts_escape_handles_quotes_backslashes_and_newlines(self):
        self.assertEqual(engine._ts_escape("a'b\\c\nnext\r"), "a\\'b\\\\c\\nnext\\r")

    def test_tape_entry_to_ts_keeps_failed_actions_as_comments(self):
        entry = {
            "action": "click",
            "params": {"selector": "[data-test-subj='missing']"},
            "ok": False,
            "error": "Timeout\nline two",
        }

        ts = engine._tape_entry_to_ts(entry)

        self.assertIn("FAILED AT RECORD TIME", ts)
        self.assertIn("Timeout", ts)
        self.assertIn("await page.click('[data-test-subj=\\'missing\\']');", ts)

    def test_tape_entry_to_ts_covers_supported_actions(self):
        cases = [
            ({"action": "navigate", "params": {"url": "https://app.example"}, "ok": True}, "page.goto"),
            ({"action": "fill", "params": {"selector": "input", "value": "abc"}, "ok": True}, "page.fill"),
            ({"action": "press", "params": {"selector": "input", "key": "Enter"}, "ok": True}, "page.press"),
            ({"action": "select_option", "params": {"selector": "select", "value": "x"}, "ok": True}, "page.selectOption"),
            ({"action": "hover", "params": {"selector": "button"}, "ok": True}, "page.hover"),
            ({"action": "wait_for", "params": {"selector": "main", "state": "attached"}, "ok": True}, "page.waitForSelector"),
            ({"action": "scroll", "params": {"selector": "section"}, "ok": True}, "scrollIntoViewIfNeeded"),
            ({"action": "scroll", "params": {"y": 250}, "ok": True}, "window.scrollBy"),
            ({"action": "extract_text", "params": {"selector": "h1"}, "ok": True}, "expect(_t.length)"),
            ({"action": "snapshot", "params": {}, "ok": True}, "snapshot"),
            ({"action": "check_page_health", "params": {}, "ok": True}, "check_page_health"),
        ]

        for entry, expected in cases:
            with self.subTest(action=entry["action"], params=entry["params"]):
                self.assertIn(expected, engine._tape_entry_to_ts(entry))

    def test_generate_playwright_spec_writes_replay_from_action_tape(self):
        thread_id = "unit_browser_engine"
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                engine._ACTION_TAPES[thread_id] = [
                    {
                        "ts": 1.0,
                        "action": "navigate",
                        "params": {"url": "https://app.example/home"},
                        "ok": True,
                    },
                    {
                        "ts": 2.0,
                        "action": "click",
                        "params": {"selector": "[data-test-subj='save']"},
                        "ok": True,
                    },
                ]

                spec_path = engine.generate_playwright_spec(
                    thread_id,
                    "Save button fails */ safely",
                    app_url="https://app.example",
                )

                self.assertTrue(os.path.isabs(spec_path))
                self.assertTrue(os.path.exists(spec_path))
                with open(spec_path, encoding="utf-8") as fh:
                    content = fh.read()
                self.assertIn("Auto-generated reproduction", content)
                self.assertIn("Save button fails * / safely", content)
                self.assertIn("await page.goto('https://app.example/home');", content)
                self.assertIn("await page.click('[data-test-subj=\\'save\\']');", content)
                self.assertIn("storageState: 'auth.json'", content)
            finally:
                os.chdir(original_cwd)
                engine._ACTION_TAPES.pop(thread_id, None)


if __name__ == "__main__":
    unittest.main()
