import json
import tempfile
import unittest
from pathlib import Path

from agentic_explorer import pr_analyzer
from agentic_explorer.config import AppMeta
from agentic_explorer.memory_health import memory_health
from agentic_explorer.pr_analyzer import (
    PRData,
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


class _FakeScenarioLLM:
    def __init__(self):
        self.prompts = []

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        self.prompts.append(messages[-1].content)
        return AIMessage(content='missions:\n  - thread_id: "pr_7_new_user_01"\n    prompt: "Check the cart."\n')


class _BrokenStore:
    async def asearch(self, *args, **kwargs):
        raise RuntimeError("store is down")


def _pr() -> PRData:
    return PRData(owner="o", repo="r", number=7, title="Add cart", body="body", diff="diff", files_changed=[])


async def test_missions_are_still_generated_when_the_bug_history_cannot_be_read(monkeypatch):
    llm = _FakeScenarioLLM()
    monkeypatch.setattr(pr_analyzer, "make_llm", lambda **kwargs: llm)

    generated = await pr_analyzer.generate_missions_from_pr(
        _pr(), AppMeta(name="Shop", url="https://app.test"), store=_BrokenStore(),
    )

    assert generated["missions"][0]["thread_id"] == "pr_7_new_user_01"
    assert memory_health.count == 1  # no longer a silent `pass`
    assert memory_health.errors[0].operation == "PR bug history read"
    assert "historically buggy" not in llm.prompts[0]


async def test_bug_history_is_injected_into_the_prompt_when_the_store_works(monkeypatch):
    from langgraph.store.memory import InMemoryStore

    from agentic_explorer.memory import app_url_hash, catalog_bug

    llm = _FakeScenarioLLM()
    monkeypatch.setattr(pr_analyzer, "make_llm", lambda **kwargs: llm)
    store = InMemoryStore()
    await catalog_bug(store, app_url_hash("https://app.test"), "Cart total is wrong", page="/cart")

    await pr_analyzer.generate_missions_from_pr(_pr(), AppMeta(name="Shop", url="https://app.test"), store=store)

    assert "historically buggy" in llm.prompts[0]
    assert "Cart total is wrong" in llm.prompts[0]
    assert memory_health.count == 0


if __name__ == "__main__":
    unittest.main()
