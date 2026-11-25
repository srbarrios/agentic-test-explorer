"""Single registry of QA personas and of how a mission chooses its graph.

Persona names, tiers, routing keywords and diagram labels used to be repeated in
``main.py``, both graph builders, ``pr_analyzer.py`` and ``ui/swarm_diagram.py``.
Adding a persona is now one entry in ``PERSONAS``.

This module stays light on purpose (no Playwright/LangChain imports at load
time) because the Streamlit dashboard imports it too.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from agentic_explorer.utils import console

Tier = Literal["standard", "advanced"]
TIERS: Tuple[str, ...] = ("standard", "advanced")


@dataclass(frozen=True)
class Persona:
    name: str          # registry key, graph node name and mission persona
    label: str         # short label for the Mermaid swarm diagram
    role: str          # "You are {role}." in the system prompt
    focus: str         # what this persona tests
    description: str   # one line shown to the supervisor when routing
    tier: Tier
    keywords: Tuple[str, ...]  # thread_id substrings that identify this persona
    style: Literal["persona", "explorer"] = "persona"

    def system_prompt(
        self, *, app_name: str, app_url: str, app_description: str, supplement: str = "",
    ) -> str:
        """Build the agent's system prompt for the app under test.

        ``supplement`` is learned memory for this persona: it goes inside the
        focus for regular personas and after the rules for the explorer.
        """
        # Imported here so that importing the registry does not load Playwright.
        from agentic_explorer.orchestration.graph_base import BROWSER_AGENT_RULES, make_browser_agent_prompt

        name = app_name or "the application"
        description = (app_description or "").strip()

        if self.style == "explorer":
            domain_line = f"Domain context: {description}\n\n" if description else ""
            prompt = (
                f"You are {self.role}. Application: '{name}' at {app_url}. {domain_line}"
                f"{self.focus} {BROWSER_AGENT_RULES}"
            )
            return f"{prompt}\n\n{supplement}" if supplement else prompt

        app_context = (
            f" The application under test is '{name}', accessible at {app_url}."
            + (f" Domain context: {description}" if description else "")
        )
        focus = f"{self.focus}\n\n{supplement}" if supplement else self.focus
        return make_browser_agent_prompt(self.role, app_context, focus)


_PERSONAS = (
    Persona(
        name="new_user_agent", label="New User", tier="standard", keywords=("new_user",),
        role="the New User / First-Timer Persona",
        focus="Test onboarding, discoverability, default states, and empty states. Catch assumptions about prior knowledge.",
        description="Tests onboarding flows, discoverability, default states, and empty states.",
    ),
    Persona(
        name="power_user_agent", label="Power User", tier="standard", keywords=("power_user",),
        role="the Power User Persona",
        focus="Use keyboard shortcuts, bulk operations, advanced filters, and edge workflows; chain operations in unexpected ways.",
        description="Uses keyboard shortcuts, bulk operations, advanced filters, edge-case workflows.",
    ),
    Persona(
        name="adversarial_user_agent", label="Adversarial", tier="standard", keywords=("adversarial_user",),
        role="the Adversarial User (Chaos Monkey) Persona",
        focus="Try to break flows with invalid inputs, injection-like strings, rapid clicks, back-button abuse, concurrent sessions, and boundary values.",
        description="Deliberately tries to break things — invalid inputs, SQL injection attempts, rapid clicks, back-button abuse.",
    ),
    Persona(
        name="accessibility_user_agent", label="Accessibility", tier="advanced", keywords=("accessibility", "a11y"),
        role="the Accessibility User Persona",
        focus="Test keyboard-only navigation, focus order, semantic structure, labels, high-contrast/zoom behavior, and WCAG-oriented outcomes.",
        description="Validates WCAG-oriented behavior, screen reader navigation, focus order, and keyboard-only interaction.",
    ),
    Persona(
        name="data_heavy_user_agent", label="Data Heavy", tier="advanced", keywords=("data_heavy", "data-heavy"),
        role="the Data-Heavy User Persona",
        focus="Use large files, many records, long strings, nested data, large result sets, and boundary inputs to expose performance cliffs and pagination bugs.",
        description="Stresses large files, large record sets, long strings, and performance-sensitive workflows.",
    ),
    Persona(
        name="impatient_user_agent", label="Impatient", tier="advanced", keywords=("impatient",),
        role="the Impatient User Persona",
        focus="Cancel mid-flight, refresh during submissions, click repeatedly, navigate away during async work, and rapidly change filters to expose races.",
        description="Cancels operations, refreshes mid-flight, clicks repeatedly, and exposes race conditions.",
    ),
    Persona(
        name="returning_user_agent", label="Returning", tier="advanced", keywords=("returning",),
        role="the Returning User Persona",
        focus="Simulate stale sessions, cached pages, outdated bookmarks, expired tokens, and journeys resumed after time away.",
        description="Tests stale sessions, cached pages, outdated bookmarks, and resumed user journeys.",
    ),
    Persona(
        name="explorer_agent", label="Explorer", tier="advanced", keywords=("explorer", "chaos", "autonomous"),
        style="explorer",
        role="the Autonomous Explorer Agent chasing unreproducible incidents",
        focus=(
            "Explore chaotically across distinct areas: filters, dropdowns, toggles, cross-links, "
            "long lists, forms, settings, boundary inputs, rapid clicks, refreshes, and combined filters. "
            "After significant interactions call check_page_health. Vary entry points and avoid repetitive paths."
        ),
        description="Autonomous chaos exploration across features, integrations, edge cases, and regression sweeps.",
    ),
)

# Insertion order matters: it is the order of the agents in the graph and of
# their descriptions in the supervisor prompt.
PERSONAS: Dict[str, Persona] = {p.name: p for p in _PERSONAS}

# Thread-id substrings that route a mission to the advanced graph (legacy fallback).
ADVANCED_KEYWORDS: Tuple[str, ...] = tuple(
    kw for p in PERSONAS.values() if p.tier == "advanced" for kw in p.keywords
)
# Every substring that identifies a known persona in a thread id.
ALL_KEYWORDS: Tuple[str, ...] = tuple(kw for p in PERSONAS.values() for kw in p.keywords)


def personas_for_tier(tier: str) -> List[Persona]:
    if tier not in TIERS:
        raise ValueError(f"tier must be 'standard' or 'advanced', got {tier!r}")
    return [p for p in PERSONAS.values() if p.tier == tier]


def describe_personas(personas: Sequence[Persona]) -> str:
    """The agent list given to the supervisor: one ``- name: description`` line each."""
    return "\n" + "\n".join(f"- {p.name}: {p.description}" for p in personas) + "\n"


def persona_for_thread_id(thread_id: str) -> Optional[Persona]:
    """The persona a mission's thread id names, by its routing keywords.

    The first persona in registry order wins when several match. ``None`` for ids
    that follow no persona naming convention (e.g. ``smoke_checkout_01``).
    """
    lowered = thread_id.lower()
    return next((p for p in PERSONAS.values() if any(kw in lowered for kw in p.keywords)), None)


def tier_for_thread_id(thread_id: str) -> Tier:
    """Legacy routing: a thread id containing an advanced keyword runs on the advanced graph."""
    return "advanced" if any(kw in thread_id.lower() for kw in ADVANCED_KEYWORDS) else "standard"


def resolve_tier(mission: Mapping[str, Any]) -> Tier:
    """The graph a mission runs on: its ``graph:`` field, else inferred from its thread id.

    The thread-id inference is deprecated: it matches substrings, so a standard
    mission whose id merely contains "explorer" is sent to the advanced graph.
    """
    thread_id = str(mission["thread_id"])
    graph = mission.get("graph")
    if graph is not None:
        if graph not in TIERS:
            raise ValueError(
                f"Mission '{thread_id}' has invalid graph {graph!r}; expected 'standard' or 'advanced'."
            )
        return graph

    tier = tier_for_thread_id(thread_id)
    message = (
        f"Mission '{thread_id}' has no 'graph:' field; inferred '{tier}' from its thread_id. "
        f"This fallback is deprecated — add `graph: {tier}` to the mission."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    console.warn(message)
    return tier
