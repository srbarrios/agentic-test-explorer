"""The unified ``build_graph(tier=...)`` must behave exactly like the two legacy builders.

``fixtures/legacy_graph_prompts.json`` was captured from commit 66e1fae, before
the unification, by running ``standard_graph.build_graph`` and
``advanced_graph.build_advanced_graph`` with fake ``create_agent`` and
supervisor factories. It records, for three app/memory variants and both tiers:
the agent order, every agent's tool names and system prompt, and the
descriptions given to the supervisor.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore

from agentic_explorer.config import AppMeta
from agentic_explorer.memory import app_url_hash
from agentic_explorer.orchestration import advanced_graph, standard_graph
from agentic_explorer.orchestration import graph as graph_module
from agentic_explorer.orchestration.graph import build_graph

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "legacy_graph_prompts.json").read_text(encoding="utf-8"))

FULL = AppMeta(name="Shop", url="https://app.test", description="An e-commerce site")
APPS = {
    "full": (FULL, False),
    "bare": (AppMeta(name="", url="https://x.test", description=""), False),
    "full+memory": (FULL, True),
}
# Raw Playwright-toolkit tools must be filtered out; the others pass through.
BASE_TOOLS = [SimpleNamespace(name=n) for n in ("click_element", "navigate_browser", "fetch_agent_skill", "mcp_docs_search")]


class _StubLLM:
    def with_structured_output(self, **_kwargs):
        return self


async def _seeded_store(app: AppMeta) -> InMemoryStore:
    store = InMemoryStore()
    url_hash = app_url_hash(app.url)
    await store.aput(("procedures", url_hash, "agent_prompts"), "new_user_agent", {"learned_additions": ["cart badge lags"]})
    await store.aput(("procedures", url_hash, "agent_prompts"), "explorer_agent", {"optimized_prompt": "Prefer /settings"})
    await store.aput(("procedures", url_hash, "routing_rules"), "current", {"rules": ["send cart bugs to power_user_agent"]})
    return store


@pytest.fixture
def capture(monkeypatch):
    """Replace create_agent / make_llm / the supervisor factory and record their inputs."""
    recorded = {"agents": [], "supervisor": {}}

    def fake_create_agent(llm, tools, system_prompt):
        recorded["agents"].append({"tools": [t.name for t in tools], "system_prompt": system_prompt.content})
        return SimpleNamespace()

    def fake_supervisor(llm, agent_names, app_url, max_steps, agent_descriptions=None, app_url_hash=""):
        recorded["supervisor"] = {
            "agent_names": list(agent_names), "descriptions": agent_descriptions,
            "max_steps": max_steps, "app_url": app_url,
        }

        async def node(state, *, store=None):
            return {}

        return node

    monkeypatch.setattr(graph_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(graph_module, "make_llm", lambda temperature=0: _StubLLM())
    monkeypatch.setattr(graph_module, "make_supervisor_node", fake_supervisor)
    return recorded


def _assert_matches_golden(recorded, golden):
    names = recorded["supervisor"]["agent_names"]
    assert names == golden["agent_names"]
    assert recorded["supervisor"]["max_steps"] == golden["max_steps"]
    assert recorded["supervisor"]["descriptions"] == golden["descriptions"]
    built = {name: agent for name, agent in zip(names, recorded["agents"])}
    assert built == golden["agents"]


@pytest.mark.parametrize("variant", list(APPS))
@pytest.mark.parametrize("tier", ["standard", "advanced"])
async def test_unified_builder_matches_the_legacy_output(capture, variant, tier):
    app, with_memory = APPS[variant]
    store = await _seeded_store(app) if with_memory else None

    compiled = await build_graph(BASE_TOOLS, object(), None, app, tier=tier, max_steps=7, quiet=True, store=store)

    _assert_matches_golden(capture, GOLDEN[variant][tier])
    nodes = set(compiled.get_graph().nodes)
    assert {"Supervisor", "Summarizer", *GOLDEN[variant][tier]["agent_names"]} <= nodes


@pytest.mark.parametrize("variant", list(APPS))
async def test_legacy_standard_wrapper_matches_and_warns(capture, variant):
    app, with_memory = APPS[variant]
    store = await _seeded_store(app) if with_memory else None

    with pytest.warns(DeprecationWarning, match="build_graph"):
        await standard_graph.build_graph(BASE_TOOLS, object(), None, app, max_steps=7, quiet=True, store=store)

    _assert_matches_golden(capture, GOLDEN[variant]["standard"])


@pytest.mark.parametrize("variant", list(APPS))
async def test_legacy_advanced_wrapper_matches_and_warns(capture, variant):
    app, with_memory = APPS[variant]
    store = await _seeded_store(app) if with_memory else None

    with pytest.warns(DeprecationWarning, match="build_advanced_graph"):
        await advanced_graph.build_advanced_graph(BASE_TOOLS, object(), None, app, max_steps=7, quiet=True, store=store)

    _assert_matches_golden(capture, GOLDEN[variant]["advanced"])


async def test_unknown_tier_is_rejected(capture):
    with pytest.raises(ValueError, match="tier must be 'standard' or 'advanced'"):
        await build_graph(BASE_TOOLS, object(), None, FULL, tier="legacy")
