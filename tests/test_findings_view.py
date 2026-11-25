"""Visual Mode shows evidence and keeps uncertain findings apart (logic tested without Streamlit)."""

import json

import pytest

from agentic_explorer.ui import state_emitter
from agentic_explorer.ui.findings_view import bug_markdown, split_findings_for_display

BUG = "Total ignores discount: Cart total shows the undiscounted price"
EVIDENCE = [{"bug": BUG, "confidence": "confirmed", "observed": "$120.00", "expected": "$96.00",
             "observed_at": "[data-test=\"total\"] on /cart", "severity": "major"}]


@pytest.fixture(autouse=True)
def emitter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_emitter._enabled = False
    state_emitter._state = state_emitter.VisualState()
    state_emitter.enable()
    yield
    state_emitter.cleanup()


# --- state emitter -----------------------------------------------------------

def test_uncertain_findings_are_tracked_apart_from_the_bug_count():
    state_emitter.add_bugs([BUG])
    state_emitter.add_uncertain(["Coupon field may reject lowercase: maybe"])
    assert state_emitter._state.bugs_count == 1
    assert state_emitter._state.uncertain_findings == ["Coupon field may reject lowercase: maybe"]


def test_uncertain_duplicates_and_confirmed_twins_are_skipped():
    state_emitter.add_bugs([BUG])
    state_emitter.add_uncertain(["Total ignores discount: not sure", "Coupon may reject lowercase: a", "Coupon may reject lowercase: b"])
    assert state_emitter._state.uncertain_findings == ["Coupon may reject lowercase: a"]


def test_evidence_is_accumulated_without_duplicates_and_serialised():
    state_emitter.add_evidence(EVIDENCE)
    state_emitter.add_evidence(EVIDENCE)
    assert len(state_emitter._state.bug_evidence) == 1
    state_emitter.emit()
    data = json.loads(open(state_emitter.STATE_FILE, encoding="utf-8").read())
    assert data["bug_evidence"][0]["expected"] == "$96.00"
    assert data["uncertain_findings"] == []


def test_emitter_helpers_are_noops_when_disabled():
    state_emitter._enabled = False
    state_emitter.add_uncertain(["x: y"])
    state_emitter.add_evidence(EVIDENCE)
    assert state_emitter._state.uncertain_findings == [] and state_emitter._state.bug_evidence == []


# --- rendering ---------------------------------------------------------------

def test_bug_markdown_shows_observed_expected_and_where():
    md = bug_markdown(BUG, EVIDENCE)
    assert BUG in md
    assert "**Observed:** $120.00" in md and "**Expected:** $96.00" in md
    assert "[data-test=\"total\"] on /cart" in md


def test_bug_markdown_says_when_no_observed_value_was_recorded():
    md = bug_markdown("Plain bug here: s", [])
    assert "no observed value recorded" in md and "Expected" not in md


def test_split_for_display_separates_confirmed_from_uncertain():
    state = {"bugs_found": [BUG], "uncertain_findings": ["Maybe broken: x"], "bug_evidence": EVIDENCE}
    view = split_findings_for_display(state)
    assert [b for b, _ in view.confirmed] == [BUG] and view.confirmed[0][1].count("Expected") == 1
    assert [b for b, _ in view.uncertain] == ["Maybe broken: x"]


def test_split_for_display_tolerates_state_files_from_older_runs():
    view = split_findings_for_display({"bugs_found": ["Old bug here: s"]})
    assert len(view.confirmed) == 1 and view.uncertain == []
    assert split_findings_for_display({}).confirmed == []
