"""Tests for the persona registry and for how a mission picks its graph.

Before this registry the persona names, their tier and their routing keywords
were repeated in ``main.py``, both graph builders, ``pr_analyzer.py`` and
``ui/swarm_diagram.py``.
"""

import warnings

import pytest

from agentic_explorer import pr_analyzer
from agentic_explorer.orchestration.graph_base import BROWSER_AGENT_RULES
from agentic_explorer.orchestration.personas import (
    ADVANCED_KEYWORDS,
    PERSONAS,
    personas_for_tier,
    resolve_tier,
    tier_for_thread_id,
)
from agentic_explorer.ui import swarm_diagram

STANDARD = ["new_user_agent", "power_user_agent", "adversarial_user_agent"]
ADVANCED = [
    "accessibility_user_agent", "data_heavy_user_agent", "impatient_user_agent",
    "returning_user_agent", "explorer_agent",
]


# ---------------------------------------------------------------- registry

def test_registry_order_and_tiers():
    assert [p.name for p in personas_for_tier("standard")] == STANDARD
    assert [p.name for p in personas_for_tier("advanced")] == ADVANCED
    assert list(PERSONAS) == STANDARD + ADVANCED


def test_registry_keys_match_persona_names():
    assert all(key == persona.name for key, persona in PERSONAS.items())


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError, match="tier must be 'standard' or 'advanced'"):
        personas_for_tier("legacy")


def test_every_persona_prompt_carries_the_browser_rules():
    for persona in PERSONAS.values():
        prompt = persona.system_prompt(app_name="Shop", app_url="https://app.test", app_description="")
        assert BROWSER_AGENT_RULES in prompt, persona.name


# ---------------------------------------------------------------- keywords (legacy compatibility)

def test_advanced_keywords_are_the_legacy_tuple():
    assert ADVANCED_KEYWORDS == (
        "accessibility", "a11y", "data_heavy", "data-heavy", "impatient",
        "returning", "explorer", "chaos", "autonomous",
    )


LEGACY_PR_ALLOWED = (
    "new_user", "power_user", "adversarial_user", "accessibility_user", "accessibility", "a11y",
    "data_heavy_user", "data_heavy", "data-heavy", "impatient_user", "impatient",
    "returning_user", "returning", "explorer", "chaos", "autonomous",
)


@pytest.mark.parametrize("legacy_keyword", LEGACY_PR_ALLOWED)
def test_pr_allowed_keywords_still_accept_every_legacy_keyword(legacy_keyword):
    thread_id = f"pr_9_{legacy_keyword}_01"
    assert any(kw in thread_id for kw in pr_analyzer._ALLOWED_PR_AGENT_KEYWORDS)


def test_pr_allowed_keywords_still_reject_unknown_agents():
    assert not any(kw in "pr_9_checkout_01" for kw in pr_analyzer._ALLOWED_PR_AGENT_KEYWORDS)


def test_pr_allowed_keywords_are_derived_from_the_registry():
    registry_keywords = {kw for p in PERSONAS.values() for kw in p.keywords}
    assert set(pr_analyzer._ALLOWED_PR_AGENT_KEYWORDS) == registry_keywords


def test_pr_system_prompt_describes_every_persona():
    for name in PERSONAS:
        assert name in pr_analyzer._SYSTEM_PROMPT, name


# ---------------------------------------------------------------- swarm diagram

def test_swarm_diagram_agents_come_from_the_registry():
    assert swarm_diagram.STANDARD_AGENTS == tuple(STANDARD)
    assert swarm_diagram.ADVANCED_AGENTS == tuple(ADVANCED)


def test_swarm_diagram_labels_come_from_the_registry():
    for persona in PERSONAS.values():
        assert swarm_diagram._LABELS[persona.name] == persona.label
    assert swarm_diagram._LABELS["Supervisor"] == "Supervisor"


# ---------------------------------------------------------------- tier_for_thread_id

@pytest.mark.parametrize("thread_id, tier", [
    ("test_new_user_01", "standard"),
    ("pr_123_power_user_02", "standard"),
    ("test_adversarial_02", "standard"),
    ("test_accessibility_01", "advanced"),
    ("test_A11Y_keyboard", "advanced"),
    ("test_data-heavy_02", "advanced"),
    ("explorer_chaos_agent_01", "advanced"),
    ("test_autonomous_run", "advanced"),
])
def test_tier_for_thread_id_by_keyword(thread_id, tier):
    assert tier_for_thread_id(thread_id) == tier


# ---------------------------------------------------------------- resolve_tier

def test_explicit_graph_field_wins_without_warning():
    mission = {"thread_id": "test_new_user_explorer_notes", "prompt": "p", "graph": "standard"}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_tier(mission) == "standard"


@pytest.mark.parametrize("graph", ["standard", "advanced"])
def test_explicit_graph_field_is_returned_as_is(graph):
    assert resolve_tier({"thread_id": "anything", "prompt": "p", "graph": graph}) == graph


def test_invalid_graph_field_raises_with_the_mission_id():
    with pytest.raises(ValueError, match="test_x"):
        resolve_tier({"thread_id": "test_x", "prompt": "p", "graph": "turbo"})


def test_missing_graph_field_falls_back_to_keywords_and_warns(capsys):
    with pytest.warns(DeprecationWarning, match="test_explorer_01.*graph: advanced"):
        assert resolve_tier({"thread_id": "test_explorer_01", "prompt": "p"}) == "advanced"

    assert "test_explorer_01" in capsys.readouterr().out  # also visible on the console


def test_fallback_documents_the_substring_footgun():
    with pytest.warns(DeprecationWarning):
        assert resolve_tier({"thread_id": "test_new_user_explorer_notes", "prompt": "p"}) == "advanced"


def test_thread_id_is_coerced_to_str():
    with pytest.warns(DeprecationWarning):
        assert resolve_tier({"thread_id": 42, "prompt": "p"}) == "standard"


# ---------------------------------------------------------------- PR generated missions get an explicit graph

@pytest.mark.parametrize("thread_id, graph", [
    ("pr_7_new_user_01", "standard"),
    ("pr_7_explorer_01", "advanced"),
    ("pr_7_impatient_user_02", "advanced"),
])
def test_validated_pr_missions_carry_an_explicit_graph(thread_id, graph):
    data = {"missions": [{"thread_id": thread_id, "prompt": "check cart"}]}

    validated = pr_analyzer._validate_missions(data, 7)

    assert validated["missions"][0]["graph"] == graph
