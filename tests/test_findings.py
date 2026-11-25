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
