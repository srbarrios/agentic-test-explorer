import unittest

from agentic_explorer.utils.llm_json import (
    extract_json_text,
    extract_yaml_text,
    normalize_llm_text,
    parse_json_from_llm,
)


class LLMJsonTests(unittest.TestCase):
    def test_normalize_llm_text_handles_string_list_blocks_and_fallback(self):
        self.assertEqual(normalize_llm_text("plain"), "plain")
        self.assertEqual(
            normalize_llm_text(["a", {"text": "b"}, {"not_text": "ignored"}, "c"]),
            "abc",
        )
        self.assertEqual(normalize_llm_text(123), "123")

    def test_extract_json_text_prefers_json_fence(self):
        content = "Intro\n```json\n{\"ok\": true}\n```\nOutro"

        self.assertEqual(extract_json_text(content), '{"ok": true}')

    def test_extract_json_text_uses_first_generic_fence(self):
        content = "Before\n```\n{\"value\": 1}\n```\nAfter"

        self.assertEqual(extract_json_text(content), '{"value": 1}')

    def test_parse_json_from_llm_parses_normalized_content(self):
        self.assertEqual(parse_json_from_llm([{"text": "```json\n{\"a\": 2}\n```"}]), {"a": 2})

    def test_extract_yaml_text_prefers_yaml_fence(self):
        content = "Intro\n```yaml\nmissions:\n  - thread_id: pr_1_new_user_01\n```\nOutro"

        self.assertEqual(
            extract_yaml_text(content),
            "missions:\n  - thread_id: pr_1_new_user_01",
        )

    def test_extract_yaml_text_returns_trimmed_plain_text_without_fence(self):
        self.assertEqual(extract_yaml_text("  missions: []\n"), "missions: []")


if __name__ == "__main__":
    unittest.main()
