"""Evidence enriches the bug catalog and regression missions without changing bug identity."""

from langgraph.store.memory import InMemoryStore

from agentic_explorer.memory import (
    _bug_fingerprint,
    catalog_bug,
    generate_regression_missions,
    write_episode_memory,
)

H = "evhash"
BUGS_NS = ("episodes", H, "bugs")
BUG = "Total ignores discount: Cart total shows the undiscounted price"
EVIDENCE = [{"bug": BUG, "confidence": "confirmed", "observed": "$120.00", "expected": "$96.00",
             "observed_at": "[data-test=\"total\"] on /cart"}]


async def test_fingerprint_is_the_same_with_or_without_evidence():
    store = InMemoryStore()
    await catalog_bug(store, H, BUG, page="/cart")
    await catalog_bug(store, H, BUG, page="/cart", evidence={"observed": "$120.00", "expected": "$96.00"})
    items = await store.asearch(BUGS_NS, limit=10)
    assert len(items) == 1  # the same bug, not a new one
    assert items[0].key == _bug_fingerprint(BUG, "/cart")
    assert items[0].value["seen_count"] == 2


async def test_evidence_is_stored_on_the_catalog_entry():
    store = InMemoryStore()
    await catalog_bug(store, H, BUG, page="/cart", evidence={"observed": "$120.00", "expected": "$96.00", "empty": ""})
    [item] = await store.asearch(BUGS_NS, limit=10)
    assert item.value["evidence"] == {"observed": "$120.00", "expected": "$96.00"}  # blanks dropped


async def test_old_entries_without_evidence_keep_working():
    store = InMemoryStore()
    await catalog_bug(store, H, "Login race", page="/login")
    [item] = await store.asearch(BUGS_NS, limit=10)
    assert "evidence" not in item.value


async def test_episode_memory_attaches_the_matching_evidence_to_each_bug():
    store = InMemoryStore()
    await write_episode_memory(
        store, H, "t1", "prompt", [], [BUG, "Empty cart lacks messaging: no text"], ["https://app.test/cart"],
        bug_evidence=EVIDENCE,
    )
    by_summary = {i.value["summary"]: i.value for i in await store.asearch(BUGS_NS, limit=10)}
    assert by_summary[BUG]["evidence"]["expected"] == "$96.00"
    assert "evidence" not in by_summary["Empty cart lacks messaging: no text"]


async def test_regression_mission_mentions_observed_and_expected():
    store = InMemoryStore()
    await catalog_bug(store, H, BUG, page="/cart", evidence={"observed": "$120.00", "expected": "$96.00"})
    [mission] = await generate_regression_missions(store, H)
    assert "$120.00" in mission["prompt"] and "$96.00" in mission["prompt"]


async def test_regression_mission_is_unchanged_without_evidence():
    store = InMemoryStore()
    await catalog_bug(store, H, "Login race", page="/login")
    [mission] = await generate_regression_missions(store, H)
    assert "Previously discovered bugs in this area: Login race." in mission["prompt"]
    assert "observed" not in mission["prompt"]
