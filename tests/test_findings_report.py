"""test_report.md shows observed / expected / where per bug and keeps uncertain findings apart."""

import os

from agentic_explorer.reporting import (
    append_action_tape_summary,
    append_findings_section,
    build_report_prompt,
    render_findings_section,
    write_report,
)

CONFIRMED = [
    "Total ignores discount: Cart total shows the undiscounted price",
    "Empty cart lacks messaging: Cart page shows no guidance text",
]
UNCERTAIN = ["Coupon may reject lowercase codes: SAVE20 applied but save20 seemed to do nothing"]
EVIDENCE = [
    {"bug": CONFIRMED[0], "confidence": "confirmed", "observed": "$120.00", "expected": "$96.00",
     "observed_at": "[data-test=\"total\"] on /cart", "severity": "major", "location": "/cart"},
    {"bug": CONFIRMED[1], "confidence": "confirmed", "severity": "minor", "location": "/cart"},
    {"bug": UNCERTAIN[0], "confidence": "uncertain", "observed": "no change after save20", "severity": "minor"},
]
GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "findings_section.md")


def test_findings_section_matches_golden():
    actual = render_findings_section(CONFIRMED, UNCERTAIN, EVIDENCE)
    if os.environ.get("UPDATE_GOLDEN"):
        open(GOLDEN, "w", encoding="utf-8").write(actual)
    assert actual == open(GOLDEN, encoding="utf-8").read()


def test_missing_observed_value_is_said_explicitly_not_omitted():
    text = render_findings_section(CONFIRMED, [], EVIDENCE)
    assert "no observed value recorded" in text


def test_bug_without_any_evidence_record_still_listed():
    text = render_findings_section(["Orphan bug here: no record"], [], [])
    assert "Orphan bug here: no record" in text and "no observed value recorded" in text


def test_uncertain_are_a_separate_section_and_never_in_the_confirmed_count():
    text = render_findings_section(CONFIRMED, UNCERTAIN, EVIDENCE)
    confirmed_part, uncertain_part = text.split("## Not determined")
    assert "Confirmed bugs (2)" in confirmed_part
    assert "Coupon may reject" in uncertain_part and "Coupon may reject" not in confirmed_part
    assert "(1)" in text.split("## Not determined")[1].splitlines()[0]


def test_no_findings_renders_nothing():
    assert render_findings_section([], [], []) == ""


def test_only_uncertain_has_no_confirmed_section():
    text = render_findings_section([], UNCERTAIN, EVIDENCE)
    assert "Confirmed bugs" not in text and "## Not determined (1)" in text


def test_append_findings_section_writes_to_the_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("report_t1")
    write_report("t1", "LLM text")
    append_findings_section("t1", CONFIRMED, UNCERTAIN, EVIDENCE)
    append_action_tape_summary("t1", tape_len=3, bugs_count=2, uncertain_count=1)
    content = (tmp_path / "report_t1" / "test_report.md").read_text()
    assert content.index("LLM text") < content.index("Confirmed bugs") < content.index("## Action Tape")
    assert "**Bugs captured:** 2" in content
    assert "**Not determined:** 1" in content


def test_tape_summary_is_unchanged_without_uncertain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("report_t1")
    write_report("t1", "x")
    append_action_tape_summary("t1", tape_len=3, bugs_count=2)
    assert "Not determined" not in (tmp_path / "report_t1" / "test_report.md").read_text()


def test_report_prompt_keeps_uncertain_out_of_the_bug_list():
    prompt = build_report_prompt("t1", "transcript", CONFIRMED, UNCERTAIN)
    assert "BUGS CAPTURED (2)" in prompt
    assert "NOT DETERMINED (1)" in prompt and "do not report these as confirmed bugs" in prompt
    assert build_report_prompt("t1", "transcript", CONFIRMED) == build_report_prompt("t1", "transcript", CONFIRMED, ())
    assert "NOT DETERMINED" not in build_report_prompt("t1", "transcript", CONFIRMED)
