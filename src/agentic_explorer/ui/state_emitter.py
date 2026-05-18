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
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

from playwright.async_api import Page

STATE_FILE = ".agent_state.json"
SCREENSHOT_FILE = ".latest_vision.jpg"

_enabled: bool = False


@dataclass
class VisualState:
    active_node: str = ""
    mission_id: str = ""
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


def append_thought(node: str, text: str) -> None:
    if not _enabled or not text.strip():
        return
    _state.thought_stream.append(
        {"node": node, "text": text[:1000], "ts": time.time()}
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
