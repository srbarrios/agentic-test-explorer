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
from agentic_explorer.memory import app_url_hash, get_recall_tool, get_memory_management_tool, get_agent_prompt_supplement, get_routing_rules_supplement
from agentic_explorer.orchestration.graph_base import (
    AgentState,
    filter_base_tools,
    make_agent_node,
    make_supervisor_node,
    make_llm,
    compile_swarm,
    make_browser_agent_prompt,
)

# ---------------------------------------------------------
# Swarm Graph Builder
# ---------------------------------------------------------

async def build_graph(base_tools: list, active_page: Page, checkpointer, app: AppMeta, max_steps: int = 30, quiet: bool = False, store=None):
    """Build the standard QA swarm of three behavioral persona agents.

    Args:
        base_tools: MCP, Skills, and raw Playwright tools (browser tools are filtered out).
        active_page: Live Playwright page used by the deterministic engine.
        checkpointer: LangGraph checkpoint backend (SQLite saver).
        app: App metadata (name, url, description) injected into agent prompts.
        max_steps: Supervisor reset threshold.
        store: Optional LangGraph Store for cross-session memory.
    """
    llm = make_llm(temperature=0)

    app_url = app.url
    app_name = app.name or "the application"
    app_description = (app.description or "").strip()
    app_context = (
        f" The application under test is '{app_name}', accessible at {app_url}."
        + (f" Domain context: {app_description}" if app_description else "")
    )

    url_hash = app_url_hash(app_url)

    # Initialize page-aware tools
    bug_screenshot = get_screenshot_tool(page=active_page)
    execute_browser_command = get_browser_command_tool(page=active_page)
    get_dom_snapshot = get_dom_snapshot_tool(page=active_page)
    generate_reproduction_spec = get_code_generator_tool(app_url=app_url)

    # Strip raw PlayWrightBrowserToolkit tools — agents emit JSON intents only.
    non_browser_base = filter_base_tools(base_tools)

    dom_tools = non_browser_base + [
        execute_browser_command,
        get_dom_snapshot,
        bug_screenshot,
        generate_reproduction_spec,
    ]
    if store:
        dom_tools.append(get_recall_tool(store, url_hash))
        dom_tools.append(get_memory_management_tool(store, url_hash))

    agent_focuses = {
        "new_user_agent": (
            "the New User / First-Timer Persona",
            "Test onboarding, discoverability, default states, and empty states. Catch assumptions about prior knowledge.",
        ),
        "power_user_agent": (
            "the Power User Persona",
            "Use keyboard shortcuts, bulk operations, advanced filters, and edge workflows; chain operations in unexpected ways.",
        ),
        "adversarial_user_agent": (
            "the Adversarial User (Chaos Monkey) Persona",
            "Try to break flows with invalid inputs, injection-like strings, rapid clicks, back-button abuse, concurrent sessions, and boundary values.",
        ),
    }

    agent_registry = {}
    for agent_name, (role, focus) in agent_focuses.items():
        if store:
            supplement = await get_agent_prompt_supplement(store, url_hash, agent_name)
            if supplement:
                focus = f"{focus}\n\n{supplement}"
        agent_registry[agent_name] = create_agent(llm, tools=dom_tools, system_prompt=SystemMessage(content=make_browser_agent_prompt(
            role, app_context, focus,
        )))

    agent_descriptions = """
- new_user_agent: Tests onboarding flows, discoverability, default states, and empty states.
- power_user_agent: Uses keyboard shortcuts, bulk operations, advanced filters, edge-case workflows.
- adversarial_user_agent: Deliberately tries to break things — invalid inputs, SQL injection attempts, rapid clicks, back-button abuse.
"""

    if store:
        routing_supplement = await get_routing_rules_supplement(store, url_hash)
        if routing_supplement:
            agent_descriptions = f"{agent_descriptions}\n{routing_supplement}"

    workflow = StateGraph(AgentState)  # type: ignore[arg-type]
    workflow.add_node("Supervisor", make_supervisor_node(llm, tuple(agent_registry), app_url, max_steps, agent_descriptions, app_url_hash=url_hash))  # type: ignore[arg-type]
    for agent_name, agent in agent_registry.items():
        workflow.add_node(agent_name, make_agent_node(agent, name=agent_name, quiet=quiet, app_url_hash=url_hash))  # type: ignore[arg-type]

    return compile_swarm(workflow, agent_registry, checkpointer, store=store)
