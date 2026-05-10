"""Application configuration loader.

Loads the user's ``config.yaml`` (path from ``APP_CONFIG`` env var, default
``./config.yaml``). Substitutes ``${ENV_VAR}`` references against the process
environment so values like ``url: ${APP_URL}`` resolve at load time.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            return os.getenv(name, "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, Mapping):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


@dataclass
class AuthConfig:
    method: str = "form"
    selectors: Dict[str, str] = field(default_factory=dict)
    post_login_check: Optional[str] = None


@dataclass
class AppMeta:
    name: str = "Web Application"
    url: str = ""
    description: str = ""


@dataclass
class PathsConfig:
    mcp_servers: Optional[str] = None
    skills_root: Optional[str] = None


@dataclass
class AppConfig:
    app: AppMeta = field(default_factory=AppMeta)
    auth: AuthConfig = field(default_factory=AuthConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_app_config(path: Optional[str | Path] = None) -> AppConfig:
    """Load the application configuration from a YAML file.

    Resolution order for the path:
      1. Explicit ``path`` argument.
      2. ``APP_CONFIG`` env var.
      3. ``./config.yaml`` relative to the current working directory.

    Returns a default ``AppConfig`` (with ``app.url`` filled from ``APP_URL``)
    if the file does not exist, so the framework can still boot for users who
    configure everything via env vars alone.
    """
    resolved = Path(path or os.getenv("APP_CONFIG", "./config.yaml"))
    if not resolved.is_file():
        return AppConfig(app=AppMeta(url=os.getenv("APP_URL", "")))

    with open(resolved, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = _interpolate(raw)

    app_raw = raw.get("app", {}) or {}
    auth_raw = raw.get("auth", {}) or {}
    paths_raw = raw.get("paths", {}) or {}

    return AppConfig(
        app=AppMeta(
            name=str(app_raw.get("name") or "Web Application"),
            url=str(app_raw.get("url") or os.getenv("APP_URL", "")),
            description=str(app_raw.get("description") or ""),
        ),
        auth=AuthConfig(
            method=str(auth_raw.get("method") or "form"),
            selectors={k: str(v) for k, v in (auth_raw.get("selectors") or {}).items()},
            post_login_check=auth_raw.get("post_login_check"),
        ),
        paths=PathsConfig(
            mcp_servers=paths_raw.get("mcp_servers") or os.getenv("MCP_SERVERS_CONFIG"),
            skills_root=paths_raw.get("skills_root") or os.getenv("AGENT_SKILLS_ROOT"),
        ),
    )
