"""Advanced testing graph: deep persona and autonomous chaos exploration.

The advanced graph hosts high-intensity behavioral personas and the open-ended
``explorer_agent`` for missions that need focused stress, accessibility,
statefulness, or autonomous chaos coverage.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph
from langchain.agents import create_agent
from playwright.async_api import Page

from agentic_explorer.config import AppMeta
from agentic_explorer.tools.common.custom_tools import get_screenshot_tool
from agentic_explorer.tools.browser.engine import (
    get_browser_command_tool,
    get_dom_snapshot_tool,
    get_code_generator_tool,
)
from agentic_explorer.orchestration.graph_base import (
    AgentState,
    filter_base_tools,
    make_agent_node,
    make_supervisor_node,
    make_llm,
    compile_swarm,
    make_browser_agent_prompt,
    BROWSER_AGENT_RULES,
)


# ---------------------------------------------------------
# Advanced Swarm Graph Builder
# ---------------------------------------------------------

def build_advanced_graph(base_tools: list, active_page: Page, checkpointer, app: AppMeta, max_steps: int = 30, quiet: bool = False):
    """Build the advanced persona and autonomous exploration graph.

    Args:
        base_tools: MCP, Skills, and raw Playwright tools (browser tools are filtered out).
        active_page: Live Playwright page used by the deterministic engine.
        checkpointer: LangGraph checkpoint backend (SQLite saver).
        app: App metadata (name, url, description) injected into agent prompts.
        max_steps: Supervisor reset threshold.
    """
    llm = make_llm(temperature=0)

    app_url = app.url
    app_name = app.name or "the application"
    app_description = (app.description or "").strip()
    app_context = (
        f" The application under test is '{app_name}', accessible at {app_url}."
        + (f" Domain context: {app_description}" if app_description else "")
    )

    bug_screenshot = get_screenshot_tool(page=active_page)
    execute_browser_command = get_browser_command_tool(page=active_page)
    get_dom_snapshot = get_dom_snapshot_tool(page=active_page)
    generate_reproduction_spec = get_code_generator_tool(app_url=app_url)

    advanced_tools = filter_base_tools(base_tools) + [
        execute_browser_command,
        get_dom_snapshot,
        bug_screenshot,
        generate_reproduction_spec,
    ]

    domain_line = f"Domain context: {app_description}\n\n" if app_description else ""

    accessibility_user_agent = create_agent(llm, tools=advanced_tools, system_prompt=SystemMessage(content=make_browser_agent_prompt(
        "the Accessibility User Persona",
        app_context,
        "Test keyboard-only navigation, focus order, semantic structure, labels, high-contrast/zoom behavior, and WCAG-oriented outcomes.",
    )))

    data_heavy_user_agent = create_agent(llm, tools=advanced_tools, system_prompt=SystemMessage(content=make_browser_agent_prompt(
        "the Data-Heavy User Persona",
        app_context,
        "Use large files, many records, long strings, nested data, large result sets, and boundary inputs to expose performance cliffs and pagination bugs.",
    )))

    impatient_user_agent = create_agent(llm, tools=advanced_tools, system_prompt=SystemMessage(content=make_browser_agent_prompt(
        "the Impatient User Persona",
        app_context,
        "Cancel mid-flight, refresh during submissions, click repeatedly, navigate away during async work, and rapidly change filters to expose races.",
    )))

    returning_user_agent = create_agent(llm, tools=advanced_tools, system_prompt=SystemMessage(content=make_browser_agent_prompt(
        "the Returning User Persona",
        app_context,
        "Simulate stale sessions, cached pages, outdated bookmarks, expired tokens, and journeys resumed after time away.",
    )))

    explorer_prompt = SystemMessage(content=(
        "You are the Autonomous Explorer Agent chasing unreproducible incidents. "
        f"Application: '{app_name}' at {app_url}. {domain_line}"
        "Explore chaotically across distinct areas: filters, dropdowns, toggles, cross-links, "
        "long lists, forms, settings, boundary inputs, rapid clicks, refreshes, and combined filters. "
        "After significant interactions call check_page_health. Vary entry points and avoid repetitive paths. "
        + BROWSER_AGENT_RULES
    ))

    explorer_agent = create_agent(llm, tools=advanced_tools, system_prompt=explorer_prompt)

    agent_registry = {
        "accessibility_user_agent": accessibility_user_agent,
        "data_heavy_user_agent": data_heavy_user_agent,
        "impatient_user_agent": impatient_user_agent,
        "returning_user_agent": returning_user_agent,
        "explorer_agent": explorer_agent,
    }

    agent_descriptions = """
- accessibility_user_agent: Validates WCAG-oriented behavior, screen reader navigation, focus order, and keyboard-only interaction.
- data_heavy_user_agent: Stresses large files, large record sets, long strings, and performance-sensitive workflows.
- impatient_user_agent: Cancels operations, refreshes mid-flight, clicks repeatedly, and exposes race conditions.
- returning_user_agent: Tests stale sessions, cached pages, outdated bookmarks, and resumed user journeys.
- explorer_agent: Autonomous chaos exploration across features, integrations, edge cases, and regression sweeps.
"""

    workflow = StateGraph(AgentState)  # type: ignore[arg-type]
    workflow.add_node("Supervisor", make_supervisor_node(llm, tuple(agent_registry), app_url, max_steps, agent_descriptions))  # type: ignore[arg-type]
    for agent_name, agent in agent_registry.items():
        workflow.add_node(agent_name, make_agent_node(agent, name=agent_name, quiet=quiet))  # type: ignore[arg-type]

    return compile_swarm(workflow, agent_registry, checkpointer)
