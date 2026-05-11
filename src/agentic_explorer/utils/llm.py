from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_OAUTH_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"
_CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


def _load_claude_settings() -> dict:
    if not _CLAUDE_SETTINGS_PATH.is_file():
        return {}
    try:
        with open(_CLAUDE_SETTINGS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _claude_vertex_config() -> Optional[dict]:
    """Return Vertex AI config if ~/.claude/settings.json declares Vertex usage, else None.

    Reads the file directly — does NOT inject env vars into os.environ.
    """
    settings = _load_claude_settings()
    env_section = settings.get("env", {}) or {}
    if env_section.get("CLAUDE_CODE_USE_VERTEX") != "1":
        return None
    project_id = env_section.get("ANTHROPIC_VERTEX_PROJECT_ID") or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
    location = env_section.get("CLOUD_ML_REGION") or os.getenv("CLOUD_ML_REGION", "us-east5")
    if not project_id:
        return None
    return {"project_id": project_id, "location": location}


def _detect_provider() -> str:
    """Auto-detect provider from available credentials.

    Order: LLM_PROVIDER env var > ANTHROPIC_API_KEY > ~/.claude/settings.json Vertex >
           GOOGLE_API_KEY > ~/.gemini/oauth_creds.json
    """
    env_provider = os.getenv("LLM_PROVIDER", "").lower()
    if env_provider in ("gemini", "claude"):
        return env_provider
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    if _claude_vertex_config() is not None:
        return "claude"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if _OAUTH_CREDS_PATH.is_file():
        return "gemini"
    raise RuntimeError(
        "No LLM credentials found. Configure one of:\n"
        "  - ANTHROPIC_API_KEY (Claude direct API)\n"
        "  - ~/.claude/settings.json with CLAUDE_CODE_USE_VERTEX=1 (Claude on Vertex AI)\n"
        "  - GOOGLE_API_KEY (Gemini API key)\n"
        "  - Run 'gemini auth login' for Gemini OAuth\n"
        "Or set LLM_PROVIDER=gemini|claude explicitly."
    )


def _default_gemini_model() -> str:
    if model := os.getenv("GEMINI_MODEL"):
        return model
    # OAuth (Gemini Advanced subscription) → use best model; API key → economical
    return "gemini-2.5-flash" if os.getenv("GOOGLE_API_KEY") else "gemini-2.5-pro"


def _default_claude_model() -> str:
    if model := os.getenv("CLAUDE_MODEL"):
        return model
    # Vertex AI (GCP billing) → Sonnet; direct API key → economical Haiku
    return "claude-sonnet-4-6" if _claude_vertex_config() else "claude-haiku-4-5"


def _make_gemini_llm(temperature: float, model_name: Optional[str]) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if model_name is None:
        model_name = _default_gemini_model()
    kwargs: Dict[str, Any] = {"model": model_name, "temperature": temperature}
    if not os.getenv("GOOGLE_API_KEY"):
        if not _OAUTH_CREDS_PATH.is_file():
            raise RuntimeError(
                f"No GOOGLE_API_KEY and no OAuth credentials at {_OAUTH_CREDS_PATH}. "
                "Set GOOGLE_API_KEY or run 'gemini auth login'."
            )
        from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
        kwargs["credentials"] = Credentials.from_authorized_user_file(str(_OAUTH_CREDS_PATH))
    return ChatGoogleGenerativeAI(**kwargs)


def _make_claude_llm(temperature: float, model_name: Optional[str]) -> Any:
    vertex_cfg = _claude_vertex_config()
    if vertex_cfg:
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex  # type: ignore[import-untyped]
        if model_name is None:
            model_name = _default_claude_model()
        return ChatAnthropicVertex(
            model_name=model_name,
            project=vertex_cfg["project_id"],
            location=vertex_cfg["location"],
            temperature=temperature,
            max_retries=0,
        )
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Claude provider selected but no credentials found. Set ANTHROPIC_API_KEY "
            "or configure ~/.claude/settings.json with CLAUDE_CODE_USE_VERTEX=1."
        )
    from langchain_anthropic import ChatAnthropic  # type: ignore[import-untyped]
    if model_name is None:
        model_name = _default_claude_model()
    return ChatAnthropic(model=model_name, temperature=temperature, api_key=api_key)


def make_llm(
    temperature: float = 0,
    model_name: Optional[str] = None,
    *,
    provider: Optional[str] = None,
) -> Any:
    """Return a LangChain chat model for the configured provider.

    Provider resolution: provider param > LLM_PROVIDER env var > auto-detect.
    Supported: 'gemini' | 'claude'
    """
    resolved = (provider or "").lower() or _detect_provider()
    if resolved == "gemini":
        return _make_gemini_llm(temperature, model_name)
    if resolved == "claude":
        return _make_claude_llm(temperature, model_name)
    raise ValueError(f"Unknown provider '{resolved}'. Use 'gemini' or 'claude'.")


def get_model_name(llm: Any) -> str:
    """Return a human-readable model identifier for any provider returned by make_llm()."""
    for attr in ("model", "model_name", "model_id"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def get_active_provider() -> str:
    """Return the resolved provider name (gemini|claude). Wraps _detect_provider."""
    try:
        return _detect_provider()
    except RuntimeError:
        return os.getenv("LLM_PROVIDER", "unknown").lower() or "unknown"
