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

    accessibility_user_agent = create_agent(llm, tools=advanced_tools, system_prompt=SystemMessage(content=(
        "You are the Accessibility User Persona."
        + app_context +
        " You test screen reader navigation, keyboard-only interaction, high-contrast/zoom modes, "
        "focus order, semantic structure, labels, and inclusive design. Validate WCAG-oriented behaviors "
        "without relying on implementation assumptions. Drive the UI exclusively through "
        "'execute_browser_command' JSON intents."
        + global_qa_rule
    )))

    data_heavy_user_agent = create_agent(llm, tools=advanced_tools, system_prompt=SystemMessage(content=(
        "You are the Data-Heavy User Persona."
        + app_context +
        " You upload large files, create many records, use very long strings, deeply nested structures, "
        "large result sets, and boundary-sized inputs. Expose performance cliffs, pagination bugs, "
        "timeouts, and degraded states. Drive the UI exclusively through 'execute_browser_command' JSON intents."
        + global_qa_rule
    )))

    impatient_user_agent = create_agent(llm, tools=advanced_tools, system_prompt=SystemMessage(content=(
        "You are the Impatient User Persona."
        + app_context +
        " You cancel operations mid-flight, refresh during submissions, click buttons multiple times, "
        "navigate away during async processes, and rapidly change filters or inputs. Expose race conditions, "
        "duplicate submissions, and incomplete state handling. Drive the UI exclusively through "
        "'execute_browser_command' JSON intents."
        + global_qa_rule
    )))

    returning_user_agent = create_agent(llm, tools=advanced_tools, system_prompt=SystemMessage(content=(
        "You are the Returning User Persona."
        + app_context +
        " You simulate stale sessions, cached pages, outdated bookmarks, saved credentials, expired tokens, "
        "and user journeys resumed after time away. Test upgrade paths, session expiry, redirect behavior, "
        "and backward compatibility. Drive the UI exclusively through 'execute_browser_command' JSON intents."
        + global_qa_rule
    )))

    explorer_prompt = SystemMessage(content=f"""You are the Autonomous Explorer Agent — an engineer chasing an unreproducible incident.

Application under test: '{app_name}' at {app_url}
{domain_line}RECORD-AND-TRANSLATE ARCHITECTURE
You are the brain. You do NOT touch the browser directly. Drive the UI by emitting
strict JSON commands to `execute_browser_command`:

  execute_browser_command({{"action":"navigate","url":"{app_url}"}})
  execute_browser_command({{"action":"click","selector":"[data-test-subj='submitButton']"}})
  execute_browser_command({{"action":"fill","selector":"input[aria-label='Search']","value":"hello"}})
  execute_browser_command({{"action":"check_page_health"}})

Allowed actions: navigate, click, fill, press, select_option, hover, wait_for,
scroll, extract_text, snapshot, check_page_health.

Use `get_dom_snapshot` to *see* the page before choosing selectors. Every command
is appended to an immutable Action Tape translatable to a Playwright `.spec.ts`.

SELECTOR POLICY (STRICTLY ENFORCED)
Priority order:
  1. [data-test-subj='myButton']
  2. [aria-label='Search'], role=button[name='Submit']
  3. button:has-text('Save'), text='Apply'
FORBIDDEN: XPath (//div, /html/...), :nth-child, :nth-of-type, bare positional paths.
Call `get_dom_snapshot` first — NEVER guess or construct structural paths.

EXPLORATION STRATEGY
1. Navigate to {app_url}, call `get_dom_snapshot` to discover all entry points.

2. Perform *chaotic* interactions across as many distinct areas as possible:
   - Click filters, dropdowns, and toggles in unusual or rapid sequences.
   - Combine multiple filters simultaneously (time range + category + search term).
   - Expand multiple panels, columns, or detail views at the same time.
   - Navigate between cross-linked pages and verify context is preserved.
   - Scroll to the bottom of long lists and paginated views.
   - Submit forms with boundary-value inputs (empty string, very long text,
     special characters: <script>, '; DROP TABLE, unicode emoji 🔥).

3. After each significant interaction, call `check_page_health` to detect:
   - Active loading spinners (stuck > 30 s indicates a hang).
   - Inline error banners ("Request failed", "Timeout", "Server error", HTTP 5xx).
   - Blank or white screens after navigation.

4. On ANY error signal:
   a. Call `capture_bug_screenshot` IMMEDIATELY with a descriptive bug_summary.
   b. Extract the exact error text from the DOM via `extract_text`.
   c. Record the reproduction steps and call `generate_reproduction_spec`.

5. Continue for 30–40 different interaction paths before concluding.
   Vary the entry point each cycle — listing views, form flows, detail pages,
   settings panels, and any navigation item visible in `get_dom_snapshot`.

IMPORTANT
- Be genuinely random and unpredictable — avoid repetitive patterns.
- Test boundary cases: extreme values, many simultaneous filters, rapid clicks.
- Your goal is to find the breaking points the development team missed.
""")

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
