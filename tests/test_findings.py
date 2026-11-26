"""Structured evidence in <bugs_found>: observed / expected / where / confidence.

Old blocks (title + summary only) must keep working and keep the same deduplication.
"""

import json
import re

from agentic_explorer.orchestration.findings import (
    CONTRACT_FIELDS,
    extract_findings,
    find_evidence,
    split_findings,
    validate_contract_entry,
)
from agentic_explorer.orchestration.graph_base import (
    BROWSER_AGENT_RULES,
    _extract_bugs_from_text,
    _normalize_bug_key,
    dedupe_bugs,
)

OLD_BLOCK = (
    '<bugs_found>[{"title": "Empty cart lacks messaging", "summary": "Cart page shows no guidance text", '
    '"location": "/cart", "severity": "minor"}]</bugs_found>'
)
NEW_BLOCK = (
    '<bugs_found>[{"title": "Total ignores discount", "summary": "Cart total shows the undiscounted price", '
    '"location": "/cart", "severity": "major", "observed": "$120.00", "observed_at": "[data-test-subj=\'total\'] on /cart", '
    '"expected": "$96.00", "confidence": "confirmed"}]</bugs_found>'
)
UNCERTAIN_BLOCK = (
    '<bugs_found>[{"title": "Coupon field may reject lowercase", "summary": "SAVE20 worked but save20 did not appear to", '
    '"location": "/cart", "severity": "minor", "observed": "no change after save20", "confidence": "uncertain"}]</bugs_found>'
)


# --- parsing -----------------------------------------------------------------

def test_old_block_still_yields_the_same_confirmed_bug():
    found = extract_findings(OLD_BLOCK)
    assert found.confirmed == ["Empty cart lacks messaging: Cart page shows no guidance text"]
    assert found.uncertain == []
    assert _extract_bugs_from_text(OLD_BLOCK) == found.confirmed


def test_new_fields_are_captured_as_evidence():
    found = extract_findings(NEW_BLOCK)
    assert found.confirmed == ["Total ignores discount: Cart total shows the undiscounted price"]
    [rec] = found.evidence
    assert rec["bug"] == found.confirmed[0]
    assert rec["confidence"] == "confirmed"
    assert rec["observed"] == "$120.00" and rec["expected"] == "$96.00"
    assert rec["observed_at"] == "[data-test-subj='total'] on /cart"


def test_uncertain_findings_are_kept_apart_from_confirmed():
    found = extract_findings(UNCERTAIN_BLOCK)
    assert found.confirmed == []
    assert found.uncertain == ["Coupon field may reject lowercase: SAVE20 worked but save20 did not appear to"]
    assert found.evidence[0]["confidence"] == "uncertain"
    # The legacy extractor feeds the confirmed count, so it must not include them.
    assert _extract_bugs_from_text(UNCERTAIN_BLOCK) == []


def test_unknown_confidence_is_treated_as_uncertain_not_rounded_up():
    block = '<bugs_found>[{"title": "Maybe broken link", "summary": "Footer link looked dead", "confidence": "probable"}]</bugs_found>'
    found = extract_findings(block)
    assert found.confirmed == [] and len(found.uncertain) == 1


def test_mixed_block_with_old_and_new_entries():
    block = (
        '<bugs_found>[{"title": "Old style bug here", "summary": "no evidence fields"}, '
        '{"title": "Total ignores discount", "summary": "wrong total", "observed": "$120", "expected": "$96"}, '
        '{"title": "Might be slow", "summary": "spinner lingered", "confidence": "uncertain"}]</bugs_found>'
    )
    found = extract_findings(block)
    assert len(found.confirmed) == 2 and len(found.uncertain) == 1
    assert {r["bug"] for r in found.evidence} == set(found.confirmed + found.uncertain)


def test_invalid_json_degrades_without_raising():
    for bad in ("<bugs_found>[{broken</bugs_found>", "<bugs_found>not json</bugs_found>", "<bugs_found>{}</bugs_found>", ""):
        found = extract_findings(bad)
        assert found.confirmed == [] and found.uncertain == [] and found.evidence == []


def test_non_string_values_and_oversized_values_are_normalised():
    block = json.dumps([{"title": "Count wrong", "summary": "badge off", "observed": 42, "expected": 7, "observed_at": "x" * 900}])
    [rec] = extract_findings(f"<bugs_found>{block}</bugs_found>").evidence
    assert rec["observed"] == "42" and rec["expected"] == "7"
    assert len(rec["observed_at"]) <= 500


def test_prose_fallback_is_still_confirmed_and_deduplicated_with_the_tag():
    text = OLD_BLOCK + "\nBUG FOUND: Empty cart lacks messaging on the /cart page."
    assert extract_findings(text).confirmed == ["Empty cart lacks messaging: Cart page shows no guidance text"]


def test_uncertain_duplicate_of_a_confirmed_bug_is_dropped():
    block = (
        '<bugs_found>[{"title": "Total ignores discount", "summary": "wrong total"}, '
        '{"title": "Total ignores discount", "summary": "wrong total?", "confidence": "uncertain"}]</bugs_found>'
    )
    found = extract_findings(block)
    assert len(found.confirmed) == 1 and found.uncertain == []


# --- deduplication stays intact ---------------------------------------------

def test_dedupe_and_key_normalisation_are_unchanged():
    assert _normalize_bug_key("Login validation race: fields rapidly") == "login validation race"
    assert dedupe_bugs(["Empty cart lacks messaging: a", "empty cart lacks messaging on /cart", "Other bug here: b"]) == [
        "Empty cart lacks messaging: a", "Other bug here: b"]


def test_split_findings_removes_uncertain_that_duplicate_confirmed():
    confirmed, uncertain = split_findings(
        ["Total ignores discount: x"], ["Total ignores discount: maybe", "Coupon may reject lowercase: y", "Coupon may reject lowercase: z"])
    assert confirmed == ["Total ignores discount: x"]
    assert uncertain == ["Coupon may reject lowercase: y"]


def test_find_evidence_matches_by_dedupe_key():
    records = extract_findings(NEW_BLOCK).evidence
    assert find_evidence("Total ignores discount: Cart total shows the undiscounted price", records)["expected"] == "$96.00"
    assert find_evidence("total ignores discount on /cart", records)["expected"] == "$96.00"
    assert find_evidence("Something else entirely: z", records) is None


# --- the prompt contract -----------------------------------------------------

def test_prompt_names_every_contract_field():
    for name in CONTRACT_FIELDS:
        assert name in BROWSER_AGENT_RULES
    assert "uncertain" in BROWSER_AGENT_RULES and "confirmed" in BROWSER_AGENT_RULES


def test_prompt_examples_are_valid_json_and_follow_the_schema():
    # The prose "<bugs_found>...</bugs_found>" in the instructions is not an example; JSON ones start with "[".
    blocks = [b for b in re.findall(r"<bugs_found>(.*?)</bugs_found>", BROWSER_AGENT_RULES, re.DOTALL)
              if b.lstrip().startswith("[")]
    assert blocks, "the prompt must show at least one example block"
    entries = []
    for raw in blocks:
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        entries.extend(parsed)
    assert entries
    for entry in entries:
        assert validate_contract_entry(entry) == [], entry
    assert any(e.get("confidence") == "uncertain" for e in entries), "the prompt must show an uncertain example"
    assert any("expected" in e and "observed" in e for e in entries), "the prompt must show observed/expected"


def test_validator_flags_bad_entries():
    assert validate_contract_entry({"title": "t"}) != []
    assert validate_contract_entry({"title": "t", "summary": "s", "severity": "huge"}) != []
    assert validate_contract_entry({"title": "t", "summary": "s", "confidence": "maybe"}) != []
    assert validate_contract_entry({"title": "t", "summary": "s", "extra": 1}) != []


# --- one agent turn ----------------------------------------------------------

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agentic_explorer.orchestration.graph_base import (  # noqa: E402
    _extract_bugs,
    _extract_turn_findings,
    make_agent_node,
)


def _screenshot_call(summary: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "capture_bug_screenshot", "args": {"bug_summary": summary}, "id": "c1"}])


def test_screenshot_bug_still_counts_as_confirmed():
    found = _extract_turn_findings([_screenshot_call("Total ignores discount on cart")])
    assert found.confirmed == ["Total ignores discount on cart"] and found.uncertain == []
    assert _extract_bugs([_screenshot_call("Total ignores discount on cart")]) == found.confirmed


def test_explicit_uncertain_verdict_wins_over_the_screenshot_that_preceded_it():
    msgs = [_screenshot_call("Coupon field may reject lowercase codes"), AIMessage(content=UNCERTAIN_BLOCK)]
    found = _extract_turn_findings(msgs)
    assert found.confirmed == []
    assert len(found.uncertain) == 1 and found.uncertain[0].startswith("Coupon field may reject lowercase")


class _EchoAgent:
    def __init__(self, *new):
        self.new = list(new)

    async def astream(self, state, config=None):
        yield {"messages": list(state["messages"])}
        yield {"messages": [*state["messages"], *self.new]}


async def test_agent_node_emits_confirmed_uncertain_and_evidence_separately():
    agent = _EchoAgent(AIMessage(content=NEW_BLOCK, id="a1"), AIMessage(content=UNCERTAIN_BLOCK, id="a2"))
    state = {"messages": [HumanMessage(content="mission", id="h0")], "next_agent": "", "action_tape": [],
             "step_count": 0, "bugs_found": [], "explored_paths": []}
    out = await make_agent_node(agent, name="agent", quiet=True)(state)
    assert out["bugs_found"] == ["Total ignores discount: Cart total shows the undiscounted price"]
    assert out["uncertain_findings"] == ["Coupon field may reject lowercase: SAVE20 worked but save20 did not appear to"]
    assert {r["confidence"] for r in out["bug_evidence"]} == {"confirmed", "uncertain"}


# --- one defect, reported twice in the same turn ------------------------------------------
# A real Haiku run: capture_bug_screenshot and generate_reproduction_spec named the bug one way
# and the closing block another, so title matching kept both (bugs=2 for one defect).

SHOT_TITLE = "Login error message mismatch when submitting empty form"
OBSERVED = "Epic sadface: Username is required"
EXPECTED = "Please enter your credentials"


def _shot_and_spec(summary=SHOT_TITLE, observed=OBSERVED, expected=EXPECTED):
    spec_args = {"bug_summary": summary}
    if observed is not None:
        spec_args["observed"] = observed
    if expected is not None:
        spec_args["expected"] = expected
    return AIMessage(content="", tool_calls=[
        {"name": "capture_bug_screenshot", "args": {"bug_summary": summary}, "id": "c1"},
        {"name": "generate_reproduction_spec", "args": spec_args, "id": "c2"},
    ])


def _block(title="Login error message does not match specification", observed=OBSERVED, expected=EXPECTED, confidence="confirmed"):
    entry = {"title": title, "summary": "shows the wrong text", "observed": observed, "observed_at": "h3 on /",
             "expected": expected, "confidence": confidence}
    return AIMessage(content=f"done <bugs_found>{json.dumps([entry])}</bugs_found>")


def test_the_same_observation_under_a_different_title_is_one_bug():
    found = _extract_turn_findings([_shot_and_spec(), _block()])
    assert found.confirmed == ["Login error message does not match specification: shows the wrong text"]
    assert found.uncertain == []
    assert [r["observed"] for r in found.evidence] == [OBSERVED]  # one record, the block's


def test_whitespace_differences_do_not_hide_the_match():
    found = _extract_turn_findings([_shot_and_spec(observed="Epic sadface:  Username is\nrequired"), _block()])
    assert len(found.confirmed) == 1


def test_a_screenshot_bug_without_a_closing_block_is_kept_and_gets_the_evidence_from_the_spec_call():
    found = _extract_turn_findings([_shot_and_spec()])
    assert found.confirmed == [SHOT_TITLE]
    [rec] = found.evidence
    assert rec["bug"] == SHOT_TITLE and rec["observed"] == OBSERVED and rec["expected"] == EXPECTED
    assert rec["confidence"] == "confirmed"


def test_different_observations_are_two_bugs():
    found = _extract_turn_findings([_shot_and_spec(), _block(title="Footer link is dead", observed="404 page")])
    assert len(found.confirmed) == 2


def test_the_same_observed_text_with_a_different_expectation_is_not_merged():
    found = _extract_turn_findings([_shot_and_spec(), _block(title="Unrelated wrong text", expected="Something else entirely")])
    assert len(found.confirmed) == 2


def test_without_an_observed_value_nothing_is_merged():
    found = _extract_turn_findings([_shot_and_spec(observed=None, expected=None), _block(title="Another title entirely")])
    assert len(found.confirmed) == 2


def test_an_uncertain_block_with_the_same_observation_absorbs_the_screenshot_too():
    found = _extract_turn_findings([_shot_and_spec(), _block(confidence="uncertain")])
    assert found.confirmed == [] and len(found.uncertain) == 1


def test_a_missing_expectation_on_one_side_still_matches_on_the_observed_value():
    found = _extract_turn_findings([_shot_and_spec(expected=None), _block()])
    assert len(found.confirmed) == 1


def test_two_distinct_defects_each_keep_their_own_entry():
    second = AIMessage(content="", tool_calls=[
        {"name": "capture_bug_screenshot", "args": {"bug_summary": "Badge count is wrong"}, "id": "c3"},
        {"name": "generate_reproduction_spec", "args": {"bug_summary": "Badge count is wrong", "observed": "5", "expected": "3"}, "id": "c4"},
    ])
    found = _extract_turn_findings([_shot_and_spec(), second, _block()])
    titles = " | ".join(found.confirmed)
    assert len(found.confirmed) == 2 and "Badge count is wrong" in titles and "does not match specification" in titles


def test_the_tool_call_arguments_may_be_missing_or_malformed():
    odd = AIMessage(content="", tool_calls=[{"name": "generate_reproduction_spec", "args": {"bug_summary": 7, "observed": None}, "id": "c9"},
                                           {"name": "generate_reproduction_spec", "args": {}, "id": "c10"}])
    assert _extract_turn_findings([odd, _shot_and_spec(observed=None, expected=None)]).confirmed == [SHOT_TITLE]
