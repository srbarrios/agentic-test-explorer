"""Replaying a recorded tape without any model, and what the verdicts mean."""

import json

import pytest

from agentic_explorer.cli import parse_args
from agentic_explorer.runtime import replay
from agentic_explorer.tools.browser import engine, redaction, session

SECRET = "hunter2-correct-horse"


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REPLAY_SECRET_1", raising=False)
    monkeypatch.delenv("REPLAY_SECRET_2", raising=False)
    session.reset()
    redaction.forget_all()
    yield tmp_path
    session.reset()
    engine._ACTION_TAPES.clear()


class _Loc:
    def __init__(self, n=0):
        self._n = n

    async def count(self):
        return self._n


class _Page:
    def __init__(self, url="https://app.test/cart", passwords=0):
        self.url = url
        self._passwords = passwords

    def locator(self, selector):
        return _Loc(self._passwords if "password" in selector else 0)


def _entry(action, ok=True, result=None, error=None, **params):
    entry = {"ts": 1.0, "action": action, "params": params, "ok": ok, "page_url": "https://app.test/cart"}
    if result is not None:
        entry["result"] = result
    if error is not None:
        entry["error"] = error
    return entry


class _Script:
    """Stands in for the browser: a scripted outcome per step, recording what was executed."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # each: result string, or an Exception to raise
        self.executed = []

    async def __call__(self, page, action, params):
        self.executed.append((action, dict(params)))
        outcome = self.outcomes[len(self.executed) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _replay(entries, outcomes, monkeypatch, page=None):
    script = _Script(outcomes)
    monkeypatch.setattr(engine, "_dispatch", script)
    report = await replay.replay_tape("t1", page or _Page(), entries)
    return report, script


# --- the verdicts ------------------------------------------------------------------

async def test_the_same_failure_again_is_reproduced(monkeypatch):
    entries = [_entry("navigate", url="https://app.test/cart"), _entry("click", ok=False, error="TimeoutError: waiting", selector="#buy")]
    report, _ = await _replay(entries, ["navigated", TimeoutError("waiting for locator")], monkeypatch)
    assert report.verdict == replay.REPRODUCED and report.exit_code == 0
    assert [s.matches for s in report.steps] == [True, True]


async def test_the_failing_step_now_succeeding_is_not_reproduced(monkeypatch):
    entries = [_entry("navigate", url="u"), _entry("click", ok=False, error="TimeoutError", selector="#buy")]
    report, _ = await _replay(entries, ["navigated", "clicked #buy"], monkeypatch)
    assert report.verdict == replay.NOT_REPRODUCED and report.exit_code == 1
    assert "succeeded" in report.steps[1].note


async def test_a_different_text_at_the_symptom_step_is_not_reproduced(monkeypatch):
    entries = [_entry("navigate", url="u"), _entry("extract_text", result="$120.00", selector="#total")]
    report, _ = await _replay(entries, ["navigated", "$96.00"], monkeypatch)
    assert report.verdict == replay.NOT_REPRODUCED
    assert "$120.00" in report.steps[1].note and "$96.00" in report.steps[1].note


async def test_the_same_text_is_reproduced_ignoring_whitespace(monkeypatch):
    entries = [_entry("navigate", url="u"), _entry("extract_text", result="Total:  $120.00", selector="#total")]
    report, _ = await _replay(entries, ["navigated", " Total: $120.00 "], monkeypatch)
    assert report.verdict == replay.REPRODUCED


async def test_an_earlier_step_breaking_is_a_divergence_not_a_verdict_on_the_bug(monkeypatch):
    entries = [_entry("navigate", url="u"), _entry("click", selector="#add"), _entry("click", ok=False, error="TimeoutError", selector="#buy")]
    report, script = await _replay(entries, ["navigated", TimeoutError("no #add"), "never"], monkeypatch)
    assert report.verdict == replay.DIVERGED and report.exit_code == 3
    assert len(script.executed) == 2  # it stops at the first mismatch; the later state is meaningless


async def test_a_recording_with_nothing_to_compare_is_inconclusive(monkeypatch):
    entries = [_entry("navigate", url="u"), _entry("click", selector="#a")]
    report, _ = await _replay(entries, ["navigated", "clicked"], monkeypatch)
    assert report.verdict == replay.INCONCLUSIVE and report.exit_code == 4
    assert "nothing" in report.reason.lower()


async def test_an_empty_tape_is_inconclusive(monkeypatch):
    report, script = await _replay([], [], monkeypatch)
    assert report.verdict == replay.INCONCLUSIVE and script.executed == []


# --- setup failures ------------------------------------------------------------------------

async def test_landing_on_a_login_page_is_a_setup_failure_not_a_bug(monkeypatch):
    session.arm(expected=True)
    entries = [_entry("navigate", url="https://app.test/cart"), _entry("click", ok=False, error="TimeoutError", selector="#buy")]
    report, script = await _replay(entries, ["navigated", "never"], monkeypatch, page=_Page("https://app.test/login"))
    assert report.verdict == replay.SETUP_FAILURE and report.exit_code == 2
    assert "auth.json" in report.reason and len(script.executed) == 1


async def test_without_a_loaded_session_a_login_page_is_just_a_page(monkeypatch):
    session.arm(expected=False)
    entries = [_entry("navigate", url="https://app.test/login"), _entry("click", ok=False, error="TimeoutError", selector="#x")]
    report, _ = await _replay(entries, ["navigated", TimeoutError("x")], monkeypatch, page=_Page("https://app.test/login"))
    assert report.verdict == replay.REPRODUCED


async def test_the_tape_stops_where_the_recording_lost_its_session(monkeypatch):
    entries = [_entry("navigate", url="u"), _entry("click", ok=False, error="TimeoutError", selector="#buy"),
               {**_entry("click", selector="#x"), "auth_required": True}]
    report, script = await _replay(entries, ["navigated", TimeoutError("x")], monkeypatch)
    assert len(script.executed) == 2 and report.verdict == replay.REPRODUCED


# --- secrets ----------------------------------------------------------------------------------

def _login_entries():
    return [
        _entry("navigate", url="https://app.test/login"),
        {**_entry("fill", selector="#password"), "params": {"selector": "#password", "value": redaction.REDACTED, "redacted": True}},
        _entry("click", ok=False, error="TimeoutError", selector="#go"),
    ]


async def test_redacted_values_come_from_the_environment_and_never_reach_the_report(monkeypatch, clean):
    monkeypatch.setenv("REPLAY_SECRET_1", SECRET)
    report, script = await _replay(_login_entries(), ["navigated", "filled", TimeoutError("x")], monkeypatch)
    assert script.executed[1][1]["value"] == SECRET
    assert report.verdict == replay.REPRODUCED
    assert SECRET not in json.dumps(report.to_dict())


async def test_a_missing_secret_stops_before_touching_the_browser(monkeypatch):
    report, script = await _replay(_login_entries(), [], monkeypatch)
    assert report.verdict == replay.INCONCLUSIVE and script.executed == []
    assert "REPLAY_SECRET_1" in report.reason


# --- the CLI and the artifact -----------------------------------------------------------------------

def test_replay_is_a_complete_command_on_its_own():
    args = parse_args(["--replay", "realrun_login_page"])
    assert args.replay == "realrun_login_page" and args.missions is None


@pytest.mark.parametrize("extra", [["--missions", "m.yaml"], ["--regression"], ["--export-model"], ["--pr-url", "https://x/y/pull/1"]])
def test_replay_cannot_be_mixed_with_running_missions(extra):
    with pytest.raises(SystemExit):
        parse_args(["--replay", "t1", *extra])


def test_the_report_is_written_next_to_the_tape(clean):
    report = replay.ReplayReport(thread_id="t1", verdict=replay.REPRODUCED, reason="same", steps=[])
    path = replay.write_report(report)
    assert path.endswith("report_t1/replay.json") or path.endswith("report_t1\\replay.json")
    assert json.loads((clean / "report_t1" / "replay.json").read_text())["verdict"] == "REPRODUCED"


async def test_replaying_a_thread_without_a_tape_is_inconclusive(clean):
    report = await replay.replay_thread("never-recorded", _Page())
    assert report.verdict == replay.INCONCLUSIVE and "tape" in report.reason.lower()


# --- read-back after fill ----------------------------------------------------------------------------

class _FillPage(_Page):
    def __init__(self, keeps):
        super().__init__()
        self._keeps = keeps

    async def input_value(self, selector):
        return "typed" if self._keeps else ""


async def _fill(page, monkeypatch, value="typed"):
    async def fake_dispatch(_page, action, params):
        return "filled"

    async def fake_snapshot(_page, limit=150):
        return {"url": "u", "title": "t", "source": "dom", "elements": []}

    monkeypatch.setattr(engine, "_dispatch", fake_dispatch)
    monkeypatch.setattr(engine, "extract_dom_snapshot", fake_snapshot)
    tool = engine.get_browser_command_tool(page=page)
    out = await tool.ainvoke({"command_json": json.dumps({"action": "fill", "selector": "#name", "value": value})},
                             config={"configurable": {"thread_id": "t1"}})
    return out, engine.get_action_tape("t1")[0]


async def test_a_fill_that_stuck_is_recorded_as_such(monkeypatch):
    out, entry = await _fill(_FillPage(keeps=True), monkeypatch)
    assert entry["effect"] == {"value_matches": True} and "EFFECT" not in out


async def test_a_fill_that_did_not_stick_is_flagged_to_the_agent_and_the_tape(monkeypatch):
    out, entry = await _fill(_FillPage(keeps=False), monkeypatch)
    assert entry["effect"] == {"value_matches": False}
    assert "EFFECT: the field did not keep the typed value" in out


async def test_the_read_back_never_stores_the_value(monkeypatch):
    _, entry = await _fill(_FillPage(keeps=False), monkeypatch, value="top-secret-string")
    assert entry["effect"] == {"value_matches": False}  # a flag, nothing else
    assert "top-secret-string" not in json.dumps(entry["effect"])


async def test_a_page_that_cannot_read_the_value_back_does_not_break_the_fill(monkeypatch):
    out, entry = await _fill(_Page(), monkeypatch)  # no input_value at all
    assert "effect" not in entry and out.startswith("STATUS: OK")
