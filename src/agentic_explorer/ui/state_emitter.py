"""Non-blocking state emitter for the Visual Mode dashboard.

When enabled (via ``enable()``), accumulates swarm state in memory and writes
it atomically to ``.agent_state.json`` on each ``emit()`` call.  Screenshots
are captured as fire-and-forget async tasks to ``.latest_vision.jpg``.

When disabled, every public function is a no-op with a single boolean check.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from playwright.async_api import Page

STATE_FILE = ".agent_state.json"
SCREENSHOT_FILE = ".latest_vision.jpg"

_enabled: bool = False


def _normalize_bug_key(text: str) -> str:
    """Dedupe key derived from the bug title (text before first ``:`` or ``.``).

    Mirrors the graph-side extractor so the Visual Mode counter agrees with
    the framework-merged ``bugs_found`` set across the three shapes agents
    emit (screenshot slug, structured tag entry, prose ``BUG FOUND:`` line).
    """
    title = text.split(":", 1)[0] if ":" in text else text
    title = title.split(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()[:60]


def _is_dup_bug(new_key: str, seen_keys: List[str], *, min_overlap: int = 12) -> bool:
    """Treat two keys as the same bug when one fully prefixes the other."""
    for existing in seen_keys:
        short, long = (new_key, existing) if len(new_key) <= len(existing) else (existing, new_key)
        if len(short) >= min_overlap and long.startswith(short):
            return True
    return False


@dataclass
class VisualState:
    active_node: str = ""
    mission_id: str = ""
    mission_description: str = ""
    mission_type: str = ""
    graph_type: str = ""
    step_count: int = 0
    bugs_count: int = 0
    bugs_found: List[str] = field(default_factory=list)
    explored_paths: List[str] = field(default_factory=list)
    last_thought: str = ""
    last_action: str = ""
    thought_stream: List[Dict[str, Any]] = field(default_factory=list)
    action_tape_recent: List[Dict[str, Any]] = field(default_factory=list)
    app_url: str = ""
    provider: str = ""
    model_name: str = ""
    timestamp: float = 0.0
    completed: bool = False
    total_missions: int = 0
    mission_history: List[Dict[str, Any]] = field(default_factory=list)


_state = VisualState()


def enable() -> None:
    global _enabled
    _enabled = True


def is_enabled() -> bool:
    return _enabled


def update(**kwargs: Any) -> None:
    if not _enabled:
        return
    for k, v in kwargs.items():
        if hasattr(_state, k):
            setattr(_state, k, v)


MAX_THOUGHT_STREAM = 200


def add_bugs(bugs: List[str]) -> None:
    """Append new bug summaries and keep ``bugs_count`` in sync.

    The graph state uses an ``operator.add`` reducer for ``bugs_found``, so
    stream updates deliver only the per-node delta — never the cumulative
    list. Use this helper (instead of ``update(bugs_found=...)``) to merge
    those deltas into the emitter without dropping previously seen bugs.

    Duplicates (same title, case- and punctuation-insensitive) are skipped
    so an agent re-summarizing the same finding across iterations does not
    inflate the Visual Mode counter.
    """
    if not _enabled or not bugs:
        return
    seen = [_normalize_bug_key(b) for b in _state.bugs_found if _normalize_bug_key(b)]
    for bug in bugs:
        key = _normalize_bug_key(bug)
        if not key or _is_dup_bug(key, seen):
            continue
        seen.append(key)
        _state.bugs_found.append(bug)
    _state.bugs_count = len(_state.bugs_found)


def add_paths(paths: List[str]) -> None:
    """Append new explored URL paths, preserving first-seen order and deduplicating."""
    if not _enabled or not paths:
        return
    seen = set(_state.explored_paths)
    for p in paths:
        if p in seen:
            continue
        _state.explored_paths.append(p)
        seen.add(p)


def append_thought(node: str, text: str) -> None:
    if not _enabled or not text.strip():
        return
    trimmed = text[:1000]
    if _state.thought_stream:
        last = _state.thought_stream[-1]
        if last["node"] == node and last["text"] == trimmed:
            return
    _state.thought_stream.append(
        {"node": node, "text": trimmed, "ts": time.time()}
    )
    if len(_state.thought_stream) > MAX_THOUGHT_STREAM:
        _state.thought_stream = _state.thought_stream[-MAX_THOUGHT_STREAM:]
    _state.last_thought = text[:1000]


def emit() -> None:
    if not _enabled:
        return
    _state.timestamp = time.time()
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(_state), f, ensure_ascii=False, default=str)
    os.replace(tmp, STATE_FILE)


async def _capture_screenshot(page: Page) -> None:
    try:
        await page.screenshot(path=SCREENSHOT_FILE, type="jpeg", quality=50)
    except Exception:
        pass


def schedule_screenshot(page: Page) -> None:
    if not _enabled:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_capture_screenshot(page))
    except RuntimeError:
        pass


def start_mission(mission_id: str, description: str, mission_type: str) -> None:
    if not _enabled:
        return
    _state.mission_history.append({
        "id": mission_id,
        "description": description,
        "type": mission_type,
        "status": "running",
        "start_time": time.time(),
        "end_time": None,
        "bugs_count": 0,
        "steps": 0,
    })


def end_mission(mission_id: str) -> None:
    if not _enabled:
        return
    for entry in _state.mission_history:
        if entry["id"] == mission_id and entry["status"] == "running":
            entry["status"] = "completed"
            entry["end_time"] = time.time()
            entry["bugs_count"] = _state.bugs_count
            entry["steps"] = _state.step_count
            break


def mark_completed(total_missions: int = 0) -> None:
    """Mark the session as completed before cleanup."""
    if not _enabled:
        return
    _state.completed = True
    _state.total_missions = total_missions
    _state.active_node = "completed"
    emit()


def cleanup() -> None:
    for f in (STATE_FILE, STATE_FILE + ".tmp", SCREENSHOT_FILE):
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
