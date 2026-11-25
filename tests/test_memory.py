"""Characterization tests for the episodic bug catalog in ``memory.py``."""

import pytest
from langgraph.store.memory import InMemoryStore

from agentic_explorer.memory import (
    _bug_fingerprint,
    app_url_hash,
    catalog_bug,
    get_agent_prompt_supplement,
    get_recall_tool,
    get_routing_rules_supplement,
    update_procedural_memory,
    write_episode_memory,
    write_session_summary,
)
from agentic_explorer.memory_health import memory_health

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


# ---------------------------------------------------------------- recall without an embedding index

@pytest.fixture(autouse=True)
def _fresh_memory_health():
    memory_health.reset()
    yield
    memory_health.reset()


async def _seed_two_bugs(store):
    await catalog_bug(store, H, "Login form rejects a valid password", page="/login")
    await catalog_bug(store, H, "Cart total is wrong after removing an item", page="/cart")


async def test_recall_filters_by_keyword_when_the_store_has_no_index():
    # Without an index, store.asearch(query=...) ignores the query and returns everything;
    # the recall tool used to rely on an exception that never comes.
    store = InMemoryStore()
    await _seed_two_bugs(store)

    out = await get_recall_tool(store, H).ainvoke({"query": "login"})

    assert "Login form rejects a valid password" in out
    assert "Cart total" not in out


async def test_recall_matches_any_word_of_a_multi_word_query():
    store = InMemoryStore()
    await _seed_two_bugs(store)

    out = await get_recall_tool(store, H).ainvoke({"query": "the cart page"})

    assert "Cart total is wrong" in out
    assert "Login form" not in out


async def test_recall_says_so_when_nothing_matches():
    store = InMemoryStore()
    await _seed_two_bugs(store)

    out = await get_recall_tool(store, H).ainvoke({"query": "payments"})

    assert out.startswith("No past findings for 'payments'")


class _SpyIndexedStore(InMemoryStore):
    """An InMemoryStore with an embedding index that records the queries it receives."""

    def __init__(self):
        super().__init__(index={"dims": 2, "embed": lambda texts: [[1.0, 0.0] for _ in texts]})
        self.queries = []

    async def asearch(self, namespace_prefix, /, *, query=None, **kwargs):
        self.queries.append(query)
        return await super().asearch(namespace_prefix, query=query, **kwargs)


async def test_recall_uses_semantic_search_when_the_store_has_an_index():
    store = _SpyIndexedStore()
    await _seed_two_bugs(store)

    await get_recall_tool(store, H).ainvoke({"query": "login"})

    assert "login" in store.queries


# ---------------------------------------------------------------- procedural memory

class _FakeOptimizer:
    """Stands in for the prompt optimizer: records calls, optionally fails for some prompts."""

    def __init__(self, fail_when_prompt_contains=None):
        self.calls = []
        self.fail_when = fail_when_prompt_contains

    async def optimize(self, prompt, trajectory, *, context):
        self.calls.append({"context": context, "prompt": prompt, "trajectory": trajectory})
        if self.fail_when and self.fail_when in prompt:
            raise RuntimeError("optimizer exploded")
        return f"OPTIMIZED[{context}]"


async def _seed_session(store, thread_id="test_new_user_01"):
    await write_session_summary(
        store, H, thread_id, "explore the cart", [{"ok": True}, {"ok": False}],
        ["Cart total wrong"], ["https://app.test/cart"],
    )


async def test_learned_prompt_reaches_the_persona_that_ran_the_mission():
    # Sessions are keyed by thread_id ("test_new_user_01") but agents read their
    # supplement by persona name ("new_user_agent"); the loop never closed.
    store = InMemoryStore()
    await _seed_session(store, "test_new_user_01")

    await update_procedural_memory(store, H, None, optimizer=_FakeOptimizer())

    supplement = await get_agent_prompt_supplement(store, H, "new_user_agent")
    assert supplement == "LEARNED FROM PAST SESSIONS:\nOPTIMIZED[batch review]"


async def test_each_persona_learns_only_from_its_own_missions():
    store = InMemoryStore()
    await _seed_session(store, "test_new_user_01")
    await _seed_session(store, "pr_7_explorer_01")

    optimizer = _FakeOptimizer()
    await update_procedural_memory(store, H, None, optimizer=optimizer)

    assert await get_agent_prompt_supplement(store, H, "new_user_agent")
    assert await get_agent_prompt_supplement(store, H, "explorer_agent")
    assert await get_agent_prompt_supplement(store, H, "power_user_agent") == ""
    prompts = [c["prompt"] for c in optimizer.calls if c["context"] == "batch review"]
    assert any("new_user_agent" in p for p in prompts) and any("explorer_agent" in p for p in prompts)


async def test_missions_that_match_no_persona_create_no_agent_prompt_but_still_feed_routing():
    store = InMemoryStore()
    await _seed_session(store, "smoke_checkout_01")

    await update_procedural_memory(store, H, None, optimizer=_FakeOptimizer())

    assert await store.asearch(("procedures", H, "agent_prompts")) == []
    assert await get_routing_rules_supplement(store, H) == "LEARNED ROUTING RULES:\nOPTIMIZED[routing review]"


async def test_versions_increase_on_every_update():
    store = InMemoryStore()
    await _seed_session(store, "test_new_user_01")
    optimizer = _FakeOptimizer()

    await update_procedural_memory(store, H, None, optimizer=optimizer)
    await update_procedural_memory(store, H, None, optimizer=optimizer)

    prompt = await store.aget(("procedures", H, "agent_prompts"), "new_user_agent")
    rules = await store.aget(("procedures", H, "routing_rules"), "current")
    assert prompt.value["version"] == 2
    assert rules.value["version"] == 2


async def test_the_optimizer_sees_the_session_trajectory():
    store = InMemoryStore()
    await _seed_session(store, "test_new_user_01")
    optimizer = _FakeOptimizer()

    await update_procedural_memory(store, H, None, optimizer=optimizer)

    trajectory = optimizer.calls[0]["trajectory"]
    assert trajectory[0] == {"role": "user", "content": "Mission: explore the cart"}
    assert "2 actions, 1 bugs found" in trajectory[1]["content"]


async def test_without_sessions_nothing_is_optimized():
    optimizer = _FakeOptimizer()

    await update_procedural_memory(InMemoryStore(), H, None, optimizer=optimizer)

    assert optimizer.calls == []


async def test_an_optimizer_failure_is_counted_and_the_other_personas_still_learn():
    store = InMemoryStore()
    await _seed_session(store, "test_new_user_01")
    await _seed_session(store, "test_power_user_01")

    await update_procedural_memory(store, H, None, optimizer=_FakeOptimizer(fail_when_prompt_contains="new_user_agent"))

    assert memory_health.count == 1  # the failed persona is no longer a silent `pass`
    assert "new_user_agent" in memory_health.errors[0].operation
    assert await get_agent_prompt_supplement(store, H, "new_user_agent") == ""
    assert await get_agent_prompt_supplement(store, H, "power_user_agent") != ""
    assert await get_routing_rules_supplement(store, H) != ""


async def test_a_routing_failure_is_counted_too():
    store = InMemoryStore()
    await _seed_session(store, "test_new_user_01")

    await update_procedural_memory(store, H, None, optimizer=_FakeOptimizer(fail_when_prompt_contains="routing supervisor"))

    assert memory_health.count == 1
    assert "routing" in memory_health.errors[0].operation
    assert await get_agent_prompt_supplement(store, H, "new_user_agent") != ""
