"""What the agent saw at each step: bounded, optional, referenced from the Action Tape.

After every browser command the engine shows the agent a compact DOM / accessibility
snapshot. The tape records the command and its outcome, but not that view, so a
reviewer could not tell what the agent was looking at when it decided something was
wrong. Each snapshot is saved as ``report_<thread>/snapshots/step_NNNN.txt`` and the
tape entry only carries ``snapshot_ref`` (the relative path), keeping
``action_tape.jsonl`` small.

Bounds: one file is capped at ``MAX_SNAPSHOT_CHARS`` and only the most recent
``AGENT_SNAPSHOT_KEEP`` files are kept (default 200; ``0`` turns the feature off).
A reference to a pruned step simply points at a file that no longer exists.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from agentic_explorer.utils import console

DEFAULT_KEEP = 200
MAX_SNAPSHOT_CHARS = 12_000
_STEP_RE = re.compile(r"step_(\d+)\.txt$")


def snapshot_keep() -> int:
    """How many snapshots to keep per mission (``AGENT_SNAPSHOT_KEEP``); 0 disables them."""
    raw = os.getenv("AGENT_SNAPSHOT_KEEP", "").strip()
    return int(raw) if raw.isdigit() else DEFAULT_KEEP


def _numbers(folder: str) -> List[int]:
    try:
        return sorted(int(m.group(1)) for name in os.listdir(folder) if (m := _STEP_RE.match(name)))
    except OSError:
        return []


def persist_snapshot(thread_id: str, text: str) -> Optional[str]:
    """Save ``text`` as the next step snapshot; return its path relative to the report folder.

    Returns ``None`` when disabled or when the disk fails: losing a snapshot must never
    break a mission.
    """
    keep = snapshot_keep()
    if keep <= 0:
        return None
    folder = os.path.join(f"report_{thread_id}", "snapshots")
    try:
        os.makedirs(folder, exist_ok=True)
        numbers = _numbers(folder)
        number = (numbers[-1] if numbers else 0) + 1
        if len(text) > MAX_SNAPSHOT_CHARS:
            text = text[:MAX_SNAPSHOT_CHARS] + f"\n... [snapshot truncated at {MAX_SNAPSHOT_CHARS} chars]"
        name = f"step_{number:04d}.txt"
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
            fh.write(text)
        for old in (numbers + [number])[:-keep]:
            try:
                os.remove(os.path.join(folder, f"step_{old:04d}.txt"))
            except OSError:
                pass
        return f"snapshots/{name}"
    except OSError as exc:
        console.warn(f"Failed to persist snapshot: {exc}")
        return None
