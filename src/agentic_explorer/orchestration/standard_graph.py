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
)

# ---------------------------------------------------------
# Swarm Graph Builder
# ---------------------------------------------------------

def build_graph(base_tools: list, active_page: Page, checkpointer, app: AppMeta, max_steps: int = 30, quiet: bool = False):
    """Build the standard QA swarm of three behavioral persona agents.

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
    global_qa_rule = (
        " ARCHITECTURE — RECORD & TRANSLATE: You are the *brain*. You do NOT touch the browser directly."
        " To interact with the application you MUST emit strict JSON commands to 'execute_browser_command'."
        " Use 'get_dom_snapshot' to inspect the page before choosing a selector."
        " Use 'check_page_health' (action: 'check_page_health') to detect spinners and error banners."
        " Every command is appended to an immutable Action Tape. Supported actions:"
        " navigate, click, fill, press, select_option, hover, wait_for, scroll, extract_text,"
        " snapshot, check_page_health."
        " Example: execute_browser_command({\"action\":\"click\",\"selector\":\"[data-test-subj='submitButton']\"})."
        " BEFORE planning, if MCP documentation tools or installed Skills are available,"
        " consult them to look up expected behaviors for the area under test. Do not guess."
        " IMPORTANT: If you discover any UI error, missing element, tool failure, or visual anomaly,"
        " you MUST (1) invoke 'capture_bug_screenshot' to save visual evidence, then"
        " (2) invoke 'generate_reproduction_spec' so the Action Tape is translated into a"
        " runnable Playwright .spec.ts that the developer can execute locally."
        " ——— SELECTOR POLICY (STRICTLY ENFORCED) ———"
        " You MUST use ONLY resilient, stable selectors. Priority order:"
        " 1. data-test-subj attributes   → [data-test-subj='myButton']"
        " 2. ARIA labels / roles         → [aria-label='Search'], role='dialog'"
        " 3. Semantic HTML / visible text → button:has-text('Save'), text='Apply'"
        " FORBIDDEN: XPath expressions (//div, /html/body/div[2]/span), positional CSS like"
        " 'div:nth-child(3) > span'. Call get_dom_snapshot first and look for data-test-subj"
        " or aria-label. NEVER invent a selector — brittle selectors cause flaky scripts."
    )

    new_user_agent = create_agent(llm, tools=dom_tools, system_prompt=SystemMessage(content=(
        "You are the New User / First-Timer Persona."
        + app_context +
        " You know nothing about the app. Test onboarding flows, discoverability, default states, and empty states. "
        "Catch assumptions developers make about prior knowledge. "
        "Drive the UI exclusively through 'execute_browser_command' JSON intents."
        + global_qa_rule
    )))

    power_user_agent = create_agent(llm, tools=dom_tools, system_prompt=SystemMessage(content=(
        "You are the Power User Persona."
        + app_context +
        " You use keyboard shortcuts, bulk operations, advanced filters, edge-case workflows. "
        "Push features to their limits and chain operations in unexpected ways. "
        "Drive the UI exclusively through 'execute_browser_command' JSON intents."
        + global_qa_rule
    )))

    adversarial_user_agent = create_agent(llm, tools=dom_tools, system_prompt=SystemMessage(content=(
        "You are the Adversarial User (Chaos Monkey) Persona."
        + app_context +
        " You deliberately try to break things — invalid inputs, SQL injection attempts, rapid clicks, back-button abuse, "
        "concurrent sessions, boundary values. Focused on robustness and security. "
        "Drive the UI exclusively through 'execute_browser_command' JSON intents."
        + global_qa_rule
    )))

    agent_registry = {
        "new_user_agent": new_user_agent,
        "power_user_agent": power_user_agent,
        "adversarial_user_agent": adversarial_user_agent,
    }

    agent_descriptions = """
- new_user_agent: Tests onboarding flows, discoverability, default states, and empty states.
- power_user_agent: Uses keyboard shortcuts, bulk operations, advanced filters, edge-case workflows.
- adversarial_user_agent: Deliberately tries to break things — invalid inputs, SQL injection attempts, rapid clicks, back-button abuse.
"""

    workflow = StateGraph(AgentState)  # type: ignore[arg-type]
    workflow.add_node("Supervisor", make_supervisor_node(llm, tuple(agent_registry), app_url, max_steps, agent_descriptions))  # type: ignore[arg-type]
    for agent_name, agent in agent_registry.items():
        workflow.add_node(agent_name, make_agent_node(agent, name=agent_name, quiet=quiet))  # type: ignore[arg-type]

    return compile_swarm(workflow, agent_registry, checkpointer)
