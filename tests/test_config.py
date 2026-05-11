import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_explorer.config import AppConfig, _interpolate, load_app_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._original_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._original_cwd)

    def test_interpolate_replaces_nested_env_values_and_missing_with_blank(self):
        with patch.dict(os.environ, {"APP_URL": "https://app.example", "TOKEN": "secret"}, clear=False):
            result = _interpolate({
                "app": {"url": "${APP_URL}", "missing": "${DOES_NOT_EXIST}"},
                "headers": ["Bearer ${TOKEN}"],
                "unchanged": 3,
            })

        self.assertEqual(result["app"]["url"], "https://app.example")
        self.assertEqual(result["app"]["missing"], "")
        self.assertEqual(result["headers"], ["Bearer secret"])
        self.assertEqual(result["unchanged"], 3)

    def test_load_app_config_returns_defaults_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"APP_URL": "https://fallback.example"}, clear=False):
            os.chdir(tmp)
            missing = Path(tmp) / "missing.yaml"

            cfg = load_app_config(missing)

        self.assertIsInstance(cfg, AppConfig)
        self.assertEqual(cfg.app.name, "Web Application")
        self.assertEqual(cfg.app.url, "https://fallback.example")
        self.assertEqual(cfg.auth.method, "form")

    def test_load_app_config_parses_nested_yaml_and_env_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "APP_URL": "https://env.example",
                "MCP_SERVERS_CONFIG": "/env/mcp.json",
                "AGENT_SKILLS_ROOT": "/env/skills",
            },
            clear=False,
        ):
            os.chdir(tmp)
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                """
app:
  name: Test App
  url: ${APP_URL}
  description: Product under test
auth:
  method: form
  selectors:
    username: '[data-test-subj="username"]'
    password: '[data-test-subj="password"]'
  post_login_check: '[data-test-subj="home"]'
paths:
  mcp_servers: /project/mcp.json
  skills_root: /project/skills
llm:
  provider: gemini
  gemini_model: gemini-test
  claude_model: claude-test
  gemini_vision_model: gemini-vision-test
  claude_vision_model: claude-vision-test
""".strip(),
                encoding="utf-8",
            )

            cfg = load_app_config(config_path)

        self.assertEqual(cfg.app.name, "Test App")
        self.assertEqual(cfg.app.url, "https://env.example")
        self.assertEqual(cfg.app.description, "Product under test")
        self.assertEqual(cfg.auth.selectors["username"], '[data-test-subj="username"]')
        self.assertEqual(cfg.auth.post_login_check, '[data-test-subj="home"]')
        self.assertEqual(cfg.paths.mcp_servers, "/project/mcp.json")
        self.assertEqual(cfg.paths.skills_root, "/project/skills")
        self.assertEqual(cfg.llm.provider, "gemini")
        self.assertEqual(cfg.llm.gemini_model, "gemini-test")
        self.assertEqual(cfg.llm.claude_model, "claude-test")
        self.assertEqual(cfg.llm.gemini_vision_model, "gemini-vision-test")
        self.assertEqual(cfg.llm.claude_vision_model, "claude-vision-test")

    def test_load_app_config_uses_env_paths_when_yaml_paths_missing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"MCP_SERVERS_CONFIG": "/env/mcp.json", "AGENT_SKILLS_ROOT": "/env/skills"},
            clear=False,
        ):
            os.chdir(tmp)
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("app:\n  name: Env Paths\n", encoding="utf-8")

            cfg = load_app_config(config_path)

        self.assertEqual(cfg.paths.mcp_servers, "/env/mcp.json")
        self.assertEqual(cfg.paths.skills_root, "/env/skills")


if __name__ == "__main__":
    unittest.main()
