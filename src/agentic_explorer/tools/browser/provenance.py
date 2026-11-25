"""Provenance of an Action Tape: which app, which session, which starting data.

A generated reproduction spec can fail before it reaches the bug (expired
``auth.json``, missing seed data). Recording where the tape came from lets a reader
tell that kind of failure apart from the bug itself.

The header lives in a sidecar file, ``report_<thread>/tape_meta.json``, and not as
the first line of ``action_tape.jsonl``: every reader of the tape (the in-memory
list, the report, the spec generator) keeps seeing actions only, and tapes
recorded before this existed need no migration.

Nothing secret is stored: ``auth.json`` is only fingerprinted (SHA-256 and mtime).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from agentic_explorer.utils import console

AUTH_STATE_PATH = "auth.json"

_PROVENANCE: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _meta_path(thread_id: str) -> str:
    return os.path.join(f"report_{thread_id}", "tape_meta.json")


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("agentic-test-explorer")
    except Exception:
        return "unknown"


def resolve_app_revision(cli_value: Optional[str], config_value: Optional[str]) -> Optional[str]:
    """Revision of the app under test: CLI flag, then ``APP_REVISION``, then ``config.yaml``.

    Returns ``None`` when nobody provided one; it is never guessed.
    """
    for candidate in (cli_value, os.getenv("APP_REVISION"), config_value):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def auth_fingerprint(path: str = AUTH_STATE_PATH) -> Dict[str, Optional[str]]:
    """SHA-256 and modification time of the session-state file, never its content."""
    try:
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        modified = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat(timespec="seconds")
        return {"path": path, "sha256": digest, "modified": modified}
    except OSError:
        return {"path": path, "sha256": None, "modified": None}


def load_provenance(thread_id: str) -> Optional[Dict[str, Any]]:
    """The header of ``thread_id``, from memory or disk; ``None`` for tapes that have none."""
    if thread_id in _PROVENANCE:
        return _PROVENANCE[thread_id]
    try:
        with open(_meta_path(thread_id), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    _PROVENANCE[thread_id] = meta
    return meta


def write_provenance(
    thread_id: str,
    *,
    app_name: str,
    app_url: str,
    app_revision: Optional[str] = None,
    seed: Sequence[str] = (),
) -> Dict[str, Any]:
    """Write the header once. A resumed mission keeps the header of its first run."""
    existing = load_provenance(thread_id)
    if existing is not None:
        return existing

    meta: Dict[str, Any] = {
        "type": "header",
        "thread_id": thread_id,
        "app_name": app_name,
        "app_url": app_url,
        "app_revision": app_revision,
        "started_at": _now_iso(),
        "tool_version": _tool_version(),
        "auth_state": auth_fingerprint(),
        "seed": list(seed),
    }
    _PROVENANCE[thread_id] = meta
    try:
        os.makedirs(f"report_{thread_id}", exist_ok=True)
        with open(_meta_path(thread_id), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    except OSError as exc:  # pragma: no cover - best-effort, the in-memory copy still serves the spec
        console.warn(f"Failed to persist tape provenance: {exc}")
    return meta
