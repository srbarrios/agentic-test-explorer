import os
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langchain.agents import create_agent
from playwright.async_api import Page

from agentic_explorer.config import AppMeta
from agentic_explorer.tools.common.custom_tools import get_visual_validation_tool, get_screenshot_tool
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
    compile_swarm,
)

# ---------------------------------------------------------
# Swarm Graph Builder
# ---------------------------------------------------------

def build_graph(base_tools: list, active_page: Page, checkpointer, app: AppMeta, max_steps: int = 30):
    """Build the standard QA swarm of 5 UI-pattern specialist agents.

    Args:
        base_tools: MCP, Skills, and raw Playwright tools (browser tools are filtered out).
        active_page: Live Playwright page used by the deterministic engine.
        checkpointer: LangGraph checkpoint backend (SQLite saver).
        app: App metadata (name, url, description) injected into agent prompts.
        max_steps: Supervisor reset threshold.
    """
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    app_url = app.url
    app_name = app.name or "the application"
    app_description = (app.description or "").strip()
    app_context = (
        f" The application under test is '{app_name}', accessible at {app_url}."
        + (f" Domain context: {app_description}" if app_description else "")
    )

    # Initialize page-aware tools
    vision_validation = get_visual_validation_tool(page=active_page)
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
    visual_tools = dom_tools + [vision_validation]

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

    listing_agent = create_agent(llm, tools=dom_tools, system_prompt=SystemMessage(content=(
        "You are the Listing & Search QA Specialist."
        + app_context +
        " Your focus is list views, data tables, search/filter bars, faceted navigation, "
        "infinite scroll, pagination, and row-detail flyouts/expanders. "
        "Verify result counts, sort behavior, search highlighting, empty states, and that "
        "row interactions (clicks, expanders, context menus) reveal correctly populated detail panels. "
        "Drive the UI exclusively through 'execute_browser_command' JSON intents."
        + global_qa_rule
    )))

    graph_agent = create_agent(llm, tools=visual_tools, system_prompt=SystemMessage(content=(
        "You are the Graph & Timeline QA Specialist."
        + app_context +
        " Your focus is node-link graphs, hierarchical trees, waterfalls, timelines, and any complex "
        "SVG/Canvas visualization where rendering correctness matters as much as data correctness. "
        "You MUST invoke 'analyze_visual_state' to validate that nodes, edges, lanes, and connectors "
        "render without overlap, broken segments, or layout regressions."
        + global_qa_rule
    )))

    chart_agent = create_agent(llm, tools=visual_tools, system_prompt=SystemMessage(content=(
        "You are the Chart & Dashboard QA Specialist."
        + app_context +
        " Your focus is time-series charts, bar/line/area visualizations, KPI tiles, gauge widgets, "
        "and dashboards composed of multiple panels. Exercise time-range pickers, legend toggles, "
        "tooltips, drill-downs, and panel resizing. Use 'analyze_visual_state' to verify axis labels, "
        "data lines, and tile layouts render cleanly."
        + global_qa_rule
    )))

    map_agent = create_agent(llm, tools=visual_tools, system_prompt=SystemMessage(content=(
        "You are the Map & Status-Grid QA Specialist."
        + app_context +
        " Your focus is geographic maps, geospatial overlays, status grids/heatmaps, and any "
        "spatially-arranged visualization. Validate marker placement, cluster collapsing, panning/zooming, "
        "tooltip popups, and that grids render without overlapping or missing tiles. "
        "Use 'analyze_visual_state' to verify spatial correctness."
        + global_qa_rule
    )))

    form_agent = create_agent(llm, tools=dom_tools, system_prompt=SystemMessage(content=(
        "You are the Form & Wizard QA Specialist."
        + app_context +
        " Your focus is forms, multi-step wizards, configuration screens, and validation flows. "
        "Exercise required-field validation, format constraints, slider/numeric inputs, dependent "
        "field reveals, and submission outcomes (success toasts, inline errors, redirect targets). "
        "Interact only by emitting JSON intents to 'execute_browser_command'."
        + global_qa_rule
    )))

    agent_registry = {
        "listing_agent": listing_agent,
        "graph_agent": graph_agent,
        "chart_agent": chart_agent,
        "map_agent": map_agent,
        "form_agent": form_agent,
    }

    workflow = StateGraph(AgentState)  # type: ignore[arg-type]
    workflow.add_node("Supervisor", make_supervisor_node(llm, tuple(agent_registry), app_url, max_steps))  # type: ignore[arg-type]
    for name, agent in agent_registry.items():
        workflow.add_node(name, make_agent_node(agent))  # type: ignore[arg-type]

    return compile_swarm(workflow, agent_registry, checkpointer)
