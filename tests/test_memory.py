"""Characterization tests for the episodic bug catalog in ``memory.py``."""

from langgraph.store.memory import InMemoryStore

from agentic_explorer.memory import (
    _bug_fingerprint,
    app_url_hash,
    catalog_bug,
    write_episode_memory,
)

H = "testhash"
BUGS_NS = ("episodes", H, "bugs")


def test_fingerprint_ignores_case_and_whitespace_but_not_page():
    a = _bug_fingerprint("Empty   cart lacks MESSAGING", "/cart")
    b = _bug_fingerprint("empty cart lacks messaging", "/cart")
    c = _bug_fingerprint("empty cart lacks messaging", "/checkout")
    assert a == b
    assert a != c
    assert len(a) == 12


def test_app_url_hash_is_stable_and_distinguishes_hosts():
    assert app_url_hash("https://one.test") == app_url_hash("https://one.test")
    assert app_url_hash("https://one.test") != app_url_hash("https://two.test")


async def test_catalog_bug_increments_instead_of_duplicating():
    store = InMemoryStore()

    await catalog_bug(store, H, "Login race", page="/login", session_id="s1")
    await catalog_bug(store, H, "Login race", page="/login", session_id="s1")
    await catalog_bug(store, H, "Login race", page="/login", session_id="s2")

    items = await store.asearch(BUGS_NS)
    assert len(items) == 1
    data = items[0].value
    assert data["seen_count"] == 3
    assert data["sessions"] == ["s1", "s2"]  # no duplicate session ids
    assert data["status"] == "open"


async def test_catalog_bug_keeps_distinct_bugs_apart():
    store = InMemoryStore()

    await catalog_bug(store, H, "Login race", page="/login")
    await catalog_bug(store, H, "Empty cart lacks messaging", page="/cart")

    assert len(await store.asearch(BUGS_NS)) == 2


async def test_write_episode_memory_attributes_bug_to_last_explored_page():
    store = InMemoryStore()

    await write_episode_memory(
        store, H, "thread_1", "mission prompt", [],
        bugs_found=["Cart total wrong"],
        explored_paths=["https://app.test/home", "https://app.test/cart?x=1"],
    )

    items = await store.asearch(BUGS_NS)
    assert len(items) == 1
    assert items[0].value["page"] == "/cart"
    assert items[0].value["sessions"] == ["thread_1"]
