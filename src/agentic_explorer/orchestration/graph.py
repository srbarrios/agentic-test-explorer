"""The supervisor/worker swarm: one builder for both the standard and the advanced graph."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph
from playwright.async_api import Page

from agentic_explorer.config import AppMeta, RuntimeConfig
from agentic_explorer.memory import (
    app_url_hash,
    get_agent_prompt_supplement,
    get_memory_management_tool,
    get_recall_tool,
    get_routing_rules_supplement,
)
from agentic_explorer.orchestration.graph_base import (
    AgentState,
    compile_swarm,
    filter_base_tools,
    make_agent_node,
    make_llm,
    make_supervisor_node,
)
from agentic_explorer.orchestration.personas import Tier, describe_personas, personas_for_tier
from agentic_explorer.orchestration.resilience import agent_node_options, supervisor_node_options
from agentic_explorer.tools.browser.engine import (
    get_browser_command_tool,
    get_code_generator_tool,
    get_dom_snapshot_tool,
)
from agentic_explorer.tools.common.custom_tools import get_screenshot_tool


async def build_graph(
    base_tools: list,
    active_page: Page,
    checkpointer,
    app: AppMeta,
    *,
    tier: Tier,
    max_steps: int = 30,
    quiet: bool = False,
    store=None,
    runtime: RuntimeConfig | None = None,
):
    """Build the swarm for ``tier``: the personas of that tier, a supervisor and a summarizer.

    Args:
        base_tools: MCP and Skills tools (raw Playwright-toolkit tools are filtered out).
        active_page: Live Playwright page used by the deterministic engine.
        checkpointer: LangGraph checkpoint backend (SQLite saver).
        app: App metadata (name, url, description) injected into agent prompts.
        tier: ``"standard"`` (3 behavioural personas) or ``"advanced"`` (4 deep personas
            plus the autonomous explorer).
        max_steps: Supervisor reset threshold.
        store: Optional LangGraph Store for cross-session memory.
        runtime: Node timeouts and retries (``config.yaml`` ``runtime:``); defaults apply if omitted.
    """
    runtime = runtime or RuntimeConfig()
    personas = personas_for_tier(tier)
    llm = make_llm(temperature=0)

    app_url = app.url
    url_hash = app_url_hash(app_url)

    # Page-aware tools. Raw PlayWrightBrowserToolkit tools are stripped: agents
    # emit JSON intents through execute_browser_command instead.
    tools = filter_base_tools(base_tools) + [
        get_browser_command_tool(page=active_page),
        get_dom_snapshot_tool(page=active_page),
        get_screenshot_tool(page=active_page),
        get_code_generator_tool(app_url=app_url),
    ]
    if store:
        tools.append(get_recall_tool(store, url_hash))
        tools.append(get_memory_management_tool(store, url_hash))

    agent_registry = {}
    for persona in personas:
        supplement = await get_agent_prompt_supplement(store, url_hash, persona.name) if store else ""
        system_prompt = persona.system_prompt(
            app_name=app.name, app_url=app_url, app_description=app.description, supplement=supplement,
        )
        agent_registry[persona.name] = create_agent(llm, tools=tools, system_prompt=SystemMessage(content=system_prompt))

    agent_descriptions = describe_personas(personas)
    if store:
        routing_supplement = await get_routing_rules_supplement(store, url_hash)
        if routing_supplement:
            agent_descriptions = f"{agent_descriptions}\n{routing_supplement}"

    workflow = StateGraph(AgentState)  # type: ignore[arg-type]
    workflow.add_node(  # type: ignore[arg-type]
        "Supervisor",
        make_supervisor_node(llm, tuple(agent_registry), app_url, max_steps, agent_descriptions, app_url_hash=url_hash),
        **supervisor_node_options(runtime),
    )
    for name, agent in agent_registry.items():
        workflow.add_node(  # type: ignore[arg-type]
            name,
            make_agent_node(agent, name=name, quiet=quiet, app_url_hash=url_hash),
            **agent_node_options(runtime),
        )

    return compile_swarm(workflow, agent_registry, checkpointer, store=store)
