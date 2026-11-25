"""Replay a recorded Action Tape without any model.

``agent-explorer --replay <thread_id>`` re-executes ``report_<thread_id>/action_tape.jsonl`` in
a fresh browser context (with ``auth.json`` if present) and compares what happens with what was
recorded. No LLM, no node, no cost. It turns "reproducible" into something measured.

Verdicts (and the process exit code):

========================  ====  ==========================================================
``REPRODUCED``            0     every step behaved as recorded, the failing step included
``NOT_REPRODUCED``        1     the recorded symptom did not happen (a failed step now
                                succeeds, or the text it read is different)
``SETUP_FAILURE``         2     the saved session looks expired: not about the bug
``DIVERGED``              3     a step before the symptom behaved differently, so the
                                state the bug needs was never reached
``INCONCLUSIVE``          4     nothing to compare, no tape, or a redacted secret is missing
========================  ====  ==========================================================

A step is a *symptom step* when it failed at record time, or it is the last ``extract_text``
(the value the bug is about). Execution stops at the first step that differs: what follows
would run against a different state.

Replaying runs the recorded actions against the real app, so it has the same side effects the
mission had (it can create data). It never runs by itself.

Values that were redacted from the tape (``tools/browser/redaction.py``) are read from
``REPLAY_SECRET_1``, ``REPLAY_SECRET_2``, ... in order, exactly like the generated spec.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agentic_explorer.tools.browser import engine, session
from agentic_explorer.utils import console

REPRODUCED = "REPRODUCED"
NOT_REPRODUCED = "NOT_REPRODUCED"
SETUP_FAILURE = "SETUP_FAILURE"
DIVERGED = "DIVERGED"
INCONCLUSIVE = "INCONCLUSIVE"
EXIT_CODES = {REPRODUCED: 0, NOT_REPRODUCED: 1, SETUP_FAILURE: 2, DIVERGED: 3, INCONCLUSIVE: 4}

# The same limits as the recording browser context (main.py), so a step fails as fast as it did then.
_ACTION_TIMEOUT_MS = 5000
_NAVIGATION_TIMEOUT_MS = 15000
_NOTE_CHARS = 200


@dataclass
class StepResult:
    index: int
    action: str
    recorded_ok: bool
    replay_ok: Optional[bool]
    matches: Optional[bool]
    note: str = ""


@dataclass
class ReplayReport:
    thread_id: str
    verdict: str
    reason: str = ""
    steps: List[StepResult] = field(default_factory=list)
    replayed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.verdict]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def _clip(text: str) -> str:
    return text if len(text) <= _NOTE_CHARS else text[:_NOTE_CHARS] + "..."


def _compare(entry: Mapping[str, Any], ok: bool, result: Optional[str], error: Optional[str]) -> tuple:
    """(matches, note) for one replayed step."""
    recorded_ok = bool(entry.get("ok"))
    if recorded_ok and not ok:
        return False, _clip(f"the step succeeded when recorded and now fails: {error}")
    if not recorded_ok and ok:
        return False, "the recorded failure did not happen: the step succeeded"
    if not recorded_ok:
        return True, _clip(f"failed again: {error}")
    if entry.get("action") == "extract_text" and _norm(entry.get("result")) != _norm(result):
        return False, _clip(f"text changed: recorded {_norm(entry.get('result'))!r}, now {_norm(result)!r}")
    return True, ""


def _symptom_steps(entries: Sequence[Mapping[str, Any]]) -> set:
    symptoms = {i for i, e in enumerate(entries) if not e.get("ok")}
    reads = [i for i, e in enumerate(entries) if e.get("action") == "extract_text" and e.get("ok")]
    if reads:
        symptoms.add(reads[-1])
    return symptoms


async def replay_tape(thread_id: str, page: Any, entries: Sequence[Mapping[str, Any]]) -> ReplayReport:
    """Re-execute ``entries`` on ``page`` and say whether the recorded behaviour came back."""
    # The recording ended where its session was lost; what comes after was never part of the bug.
    entries = list(entries)
    for i, entry in enumerate(entries):
        if entry.get("auth_required"):
            entries = entries[:i]
            break
    if not entries:
        return ReplayReport(thread_id, INCONCLUSIVE, "The tape is empty: there is nothing to replay.")

    symptoms = _symptom_steps(entries)
    if not symptoms:
        return ReplayReport(
            thread_id, INCONCLUSIVE,
            "Nothing to compare: the recording has no failed step and read no text, so a replay cannot tell a bug from a normal run.",
        )

    secrets: Dict[int, str] = {}
    missing: List[str] = []
    ordinal = 0
    for i, entry in enumerate(entries):
        if (entry.get("params") or {}).get("redacted"):
            ordinal += 1
            name = f"REPLAY_SECRET_{ordinal}"
            if os.getenv(name) is None:
                missing.append(name)
            else:
                secrets[i] = os.environ[name]
    if missing:
        return ReplayReport(
            thread_id, INCONCLUSIVE,
            f"Values typed into secret fields were redacted from the tape. Set {', '.join(missing)} and replay again.",
        )

    steps: List[StepResult] = []
    for i, entry in enumerate(entries):
        action = entry["action"]
        params = dict(entry.get("params") or {})
        if i in secrets:
            params["value"] = secrets[i]
            params.pop("redacted", None)

        result: Optional[str] = None
        error: Optional[str] = None
        try:
            result = await engine._dispatch(page, action, params)
            ok = True
        except Exception as exc:
            ok, error = False, f"{type(exc).__name__}: {exc}"

        recorded_params = entry.get("params") or {}
        if await session.check_page(page, action=action, params=recorded_params, tape=entries[:i]):
            steps.append(StepResult(i, action, bool(entry.get("ok")), ok, None, "landed on a login page"))
            return ReplayReport(
                thread_id, SETUP_FAILURE,
                "The saved session (auth.json) looks expired: the replay landed on a login page the recording never "
                "visited. Run agent-auth and replay again. This says nothing about the bug.",
                steps,
            )

        matches, note = _compare(entry, ok, result, error)
        steps.append(StepResult(i, action, bool(entry.get("ok")), ok, matches, note))
        if not matches:
            if i in symptoms:
                return ReplayReport(thread_id, NOT_REPRODUCED, f"Step {i} ({action}): {note}", steps)
            return ReplayReport(
                thread_id, DIVERGED,
                f"Step {i} ({action}) behaved differently from the recording ({note}) before the symptom, "
                "so the state the bug needs was not reached.",
                steps,
            )

    return ReplayReport(
        thread_id, REPRODUCED,
        f"All {len(steps)} steps behaved as recorded, including {len(symptoms)} symptom step(s).", steps,
    )


async def replay_thread(thread_id: str, page: Any) -> ReplayReport:
    """Replay the tape that ``report_<thread_id>/action_tape.jsonl`` holds."""
    entries = engine.get_action_tape(thread_id)
    if not entries:
        return ReplayReport(thread_id, INCONCLUSIVE, f"No tape found: report_{thread_id}/action_tape.jsonl is missing or empty.")
    return await replay_tape(thread_id, page, entries)


def write_report(report: ReplayReport) -> str:
    """Write ``report_<thread_id>/replay.json`` and return its path."""
    folder = f"report_{report.thread_id}"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "replay.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


async def run_replay_command(thread_id: str, *, headed: bool = False) -> int:
    """The ``--replay`` command: launch a fresh browser, replay, print and save the verdict."""
    from playwright.async_api import async_playwright

    session.arm(expected=os.path.exists("auth.json"), control=None)
    console.warn("Replaying runs the recorded actions against the real app, with the same side effects as the mission.")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not headed)
        try:
            kwargs: Dict[str, Any] = {"no_viewport": True, "ignore_https_errors": True}
            if os.path.exists("auth.json"):
                kwargs["storage_state"] = "auth.json"
            context = await browser.new_context(**kwargs)
            context.set_default_timeout(_ACTION_TIMEOUT_MS)
            context.set_default_navigation_timeout(_NAVIGATION_TIMEOUT_MS)
            report = await replay_thread(thread_id, await context.new_page())
        finally:
            await browser.close()

    path = write_report(report)
    for step in report.steps:
        mark = "ok " if step.matches else ("?? " if step.matches is None else "!! ")
        console.info(f"  {mark}{step.index:>2} {step.action}" + (f"  {step.note}" if step.note else ""))
    (console.success if report.verdict == REPRODUCED else console.warn)(f"{report.verdict}: {report.reason}")
    console.info(f"Saved {path}")
    return report.exit_code
