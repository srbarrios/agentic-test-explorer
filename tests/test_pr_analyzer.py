import json
import tempfile
import unittest
from pathlib import Path

from agentic_explorer.pr_analyzer import (
    _find_github_server,
    _parse_files_text,
    _parse_pr_metadata,
    _validate_missions,
    parse_pr_url,
)


class PRAnalyzerTests(unittest.TestCase):
    def test_parse_pr_url_accepts_nested_url_parts(self):
        owner, repo, number = parse_pr_url("https://github.com/org/repo-name/pull/456/files")

        self.assertEqual(owner, "org")
        self.assertEqual(repo, "repo-name")
        self.assertEqual(number, 456)

    def test_parse_pr_url_rejects_non_pr_url(self):
        with self.assertRaisesRegex(ValueError, "Expected format"):
            parse_pr_url("https://github.com/org/repo/issues/456")

    def test_find_github_server_supports_mcp_servers_transport_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "mcp_servers.json"
            config_path.write_text(
                json.dumps({
                    "mcpServers": {
                        "github": {
                            "transport": "http",
                            "url": "https://api.githubcopilot.com/mcp/",
                        }
                    }
                }),
                encoding="utf-8",
            )

            self.assertEqual(
                _find_github_server(config_path),
                {
                    "github": {
                        "transport": "http",
                        "url": "https://api.githubcopilot.com/mcp/",
                    }
                },
            )

    def test_find_github_server_normalizes_servers_type_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "mcp_servers.json"
            config_path.write_text(
                json.dumps({
                    "servers": {
                        "my-github-server": {
                            "type": "http",
                            "url": "https://example.test/mcp",
                        }
                    }
                }),
                encoding="utf-8",
            )

            self.assertEqual(
                _find_github_server(config_path),
                {"my-github-server": {"transport": "http", "url": "https://example.test/mcp"}},
            )

    def test_parse_pr_metadata_handles_json_labels(self):
        title, body, labels = _parse_pr_metadata(json.dumps({
            "title": "Improve filters",
            "body": "Adds advanced filtering.",
            "labels": [{"name": "ui"}, "qa"],
        }))

        self.assertEqual(title, "Improve filters")
        self.assertEqual(body, "Adds advanced filtering.")
        self.assertEqual(labels, ["ui", "qa"])

    def test_parse_pr_metadata_falls_back_to_first_line_title(self):
        title, body, labels = _parse_pr_metadata("Title line\nLong body")

        self.assertEqual(title, "Title line")
        self.assertEqual(body, "Title line\nLong body")
        self.assertEqual(labels, [])

    def test_parse_files_text_handles_json_and_plain_text(self):
        self.assertEqual(
            _parse_files_text(json.dumps([
                {"filename": "src/app.py", "additions": 10, "deletions": 2},
                {"path": "src/other.py"},
            ])),
            [
                {"filename": "src/app.py", "additions": 10, "deletions": 2},
                {"filename": "src/other.py", "additions": 0, "deletions": 0},
            ],
        )
        self.assertEqual(
            _parse_files_text("- src/app.py\n- src/other.py"),
            [
                {"filename": "src/app.py", "additions": 0, "deletions": 0},
                {"filename": "src/other.py", "additions": 0, "deletions": 0},
            ],
        )

    def test_validate_missions_accepts_allowed_pr_thread_id(self):
        data = {
            "missions": [
                {"thread_id": "pr_123_new_user_01", "prompt": "Verify the updated onboarding flow."},
                {"thread_id": "pr_123_explorer_02", "prompt": "Explore adjacent regressions."},
            ]
        }

        self.assertIs(_validate_missions(data, 123), data)

    def test_validate_missions_rejects_wrong_pr_namespace(self):
        with self.assertRaisesRegex(ValueError, "pr_123"):
            _validate_missions(
                {"missions": [{"thread_id": "pr_999_new_user_01", "prompt": "Verify flow."}]},
                123,
            )

    def test_validate_missions_rejects_deleted_agent_keyword_token(self):
        with self.assertRaisesRegex(ValueError, "deleted agent"):
            _validate_missions(
                {"missions": [{"thread_id": "pr_123_listing_01", "prompt": "Verify listing."}]},
                123,
            )

    def test_validate_missions_rejects_unknown_agent_keyword(self):
        with self.assertRaisesRegex(ValueError, "allowed agent"):
            _validate_missions(
                {"missions": [{"thread_id": "pr_123_unknown_01", "prompt": "Verify flow."}]},
                123,
            )


if __name__ == "__main__":
    unittest.main()
