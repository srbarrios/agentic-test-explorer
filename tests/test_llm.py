import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_explorer.utils import llm as llm_mod


class GeminiOAuthCredsTest(unittest.TestCase):
    """Verify _make_gemini_llm reads the Gemini CLI oauth_creds.json format."""

    def _write_creds(self, data: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @patch.dict(os.environ, {}, clear=True)
    def test_oauth_creds_without_client_id_loads_successfully(self):
        """The gemini auth login format has access_token/refresh_token but no
        client_id/client_secret.  This must not raise."""
        creds_data = {
            "access_token": "ya29.test-access-token",
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "token_type": "Bearer",
            "id_token": "eyJ-id-token",
            "expiry_date": 1716000000000,
            "refresh_token": "1//test-refresh-token",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            creds_path = Path(tmpdir) / "oauth_creds.json"
            self._write_creds(creds_data, creds_path)

            fake_llm = MagicMock()
            with (
                patch.object(llm_mod, "_OAUTH_CREDS_PATH", creds_path),
                patch(
                    "langchain_google_genai.ChatGoogleGenerativeAI",
                    return_value=fake_llm,
                ) as mock_cls,
            ):
                result = llm_mod._make_gemini_llm(temperature=0, model_name="gemini-test")

        self.assertIs(result, fake_llm)
        call_kwargs = mock_cls.call_args.kwargs
        creds = call_kwargs["credentials"]
        self.assertEqual(creds.token, "ya29.test-access-token")
        self.assertEqual(creds.refresh_token, "1//test-refresh-token")

    @patch.dict(os.environ, {}, clear=True)
    def test_oauth_creds_with_only_access_token(self):
        """Minimal creds file with just an access_token should still work."""
        creds_data = {"access_token": "ya29.minimal-token"}
        with tempfile.TemporaryDirectory() as tmpdir:
            creds_path = Path(tmpdir) / "oauth_creds.json"
            self._write_creds(creds_data, creds_path)

            fake_llm = MagicMock()
            with (
                patch.object(llm_mod, "_OAUTH_CREDS_PATH", creds_path),
                patch(
                    "langchain_google_genai.ChatGoogleGenerativeAI",
                    return_value=fake_llm,
                ) as mock_cls,
            ):
                result = llm_mod._make_gemini_llm(temperature=0.5, model_name="gemini-test")

        self.assertIs(result, fake_llm)
        creds = mock_cls.call_args.kwargs["credentials"]
        self.assertEqual(creds.token, "ya29.minimal-token")
        self.assertIsNone(creds.refresh_token)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_oauth_file_raises(self):
        """When no GOOGLE_API_KEY and no oauth file exist, raise RuntimeError."""
        missing = Path("/tmp/nonexistent_dir_test_llm/oauth_creds.json")
        with patch.object(llm_mod, "_OAUTH_CREDS_PATH", missing):
            with self.assertRaises(RuntimeError) as ctx:
                llm_mod._make_gemini_llm(temperature=0, model_name="gemini-test")
        self.assertIn("GOOGLE_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_api_key_skips_oauth(self):
        """When GOOGLE_API_KEY is set, OAuth creds are not loaded."""
        fake_llm = MagicMock()
        with patch(
            "langchain_google_genai.ChatGoogleGenerativeAI",
            return_value=fake_llm,
        ) as mock_cls:
            result = llm_mod._make_gemini_llm(temperature=0, model_name="gemini-test")

        self.assertIs(result, fake_llm)
        self.assertNotIn("credentials", mock_cls.call_args.kwargs)


class DetectProviderTest(unittest.TestCase):
    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True)
    def test_env_var_gemini(self):
        self.assertEqual(llm_mod._detect_provider(), "gemini")

    @patch.dict(os.environ, {"LLM_PROVIDER": "claude"}, clear=True)
    def test_env_var_claude(self):
        self.assertEqual(llm_mod._detect_provider(), "claude")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True)
    def test_anthropic_key_yields_claude(self):
        self.assertEqual(llm_mod._detect_provider(), "claude")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True)
    @patch.object(llm_mod, "_claude_vertex_config", return_value=None)
    def test_google_key_yields_gemini(self, _):
        self.assertEqual(llm_mod._detect_provider(), "gemini")

    @patch.dict(os.environ, {}, clear=True)
    @patch.object(llm_mod, "_claude_vertex_config", return_value=None)
    def test_oauth_file_yields_gemini(self, _):
        with tempfile.TemporaryDirectory() as tmpdir:
            creds_path = Path(tmpdir) / "oauth_creds.json"
            creds_path.write_text("{}", encoding="utf-8")
            with patch.object(llm_mod, "_OAUTH_CREDS_PATH", creds_path):
                self.assertEqual(llm_mod._detect_provider(), "gemini")

    @patch.dict(os.environ, {}, clear=True)
    @patch.object(llm_mod, "_claude_vertex_config", return_value=None)
    def test_no_credentials_raises(self, _):
        missing = Path("/tmp/nonexistent_dir_test_llm/oauth_creds.json")
        with patch.object(llm_mod, "_OAUTH_CREDS_PATH", missing):
            with self.assertRaises(RuntimeError):
                llm_mod._detect_provider()


class DefaultModelTest(unittest.TestCase):
    @patch.dict(os.environ, {"GEMINI_MODEL": "gemini-custom"}, clear=True)
    def test_gemini_model_env_override(self):
        self.assertEqual(llm_mod._default_gemini_model(), "gemini-custom")

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "key"}, clear=True)
    def test_gemini_api_key_uses_flash_lite(self):
        self.assertEqual(llm_mod._default_gemini_model(), "gemini-3.1-flash-lite")

    @patch.dict(os.environ, {}, clear=True)
    def test_gemini_oauth_uses_flash(self):
        self.assertEqual(llm_mod._default_gemini_model(), "gemini-3.1-flash")

    @patch.dict(os.environ, {"CLAUDE_MODEL": "claude-opus-4"}, clear=True)
    def test_claude_model_env_override(self):
        self.assertEqual(llm_mod._default_claude_model(), "claude-opus-4")

    @patch.dict(os.environ, {}, clear=True)
    def test_claude_default_model(self):
        self.assertEqual(llm_mod._default_claude_model(), "claude-haiku-4-5")


class MakeLlmDispatchTest(unittest.TestCase):
    @patch.object(llm_mod, "_make_gemini_llm", return_value="gemini-llm")
    def test_explicit_gemini_provider(self, mock_fn):
        result = llm_mod.make_llm(temperature=0.1, provider="gemini")
        self.assertEqual(result, "gemini-llm")
        mock_fn.assert_called_once_with(0.1, None)

    @patch.object(llm_mod, "_make_claude_llm", return_value="claude-llm")
    def test_explicit_claude_provider(self, mock_fn):
        result = llm_mod.make_llm(temperature=0.2, model_name="custom", provider="Claude")
        self.assertEqual(result, "claude-llm")
        mock_fn.assert_called_once_with(0.2, "custom")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            llm_mod.make_llm(provider="openai")
        self.assertIn("openai", str(ctx.exception))


class GetModelNameTest(unittest.TestCase):
    def test_reads_model_attr(self):
        obj = MagicMock(spec=[])
        obj.model = "gemini-3.1-flash"
        self.assertEqual(llm_mod.get_model_name(obj), "gemini-3.1-flash")

    def test_falls_back_to_unknown(self):
        obj = MagicMock(spec=[])
        self.assertEqual(llm_mod.get_model_name(obj), "unknown")


if __name__ == "__main__":
    unittest.main()
