"""Advanced testing graph: autonomous chaos exploration.

The advanced graph hosts open-ended exploration agents that don't follow a
prescriptive script.  The ``explorer_agent`` wanders the app looking for
crashes, timeouts, and visual regressions.

The single-agent supervisor is preserved as an extension point so additional
specialized agents (custom fuzzers, integrity auditors, etc.) can be added
without restructuring the graph.
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

def build_advanced_graph(base_tools: list, active_page: Page, checkpointer, app: AppMeta, max_steps: int = 30):
    """Build the advanced (autonomous exploration) graph.

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

    bug_screenshot = get_screenshot_tool(page=active_page)
    execute_browser_command = get_browser_command_tool(page=active_page)
    get_dom_snapshot = get_dom_snapshot_tool(page=active_page)
    generate_reproduction_spec = get_code_generator_tool(app_url=app_url)

    explorer_tools = filter_base_tools(base_tools) + [
        execute_browser_command,
        get_dom_snapshot,
        bug_screenshot,
        generate_reproduction_spec,
    ]

    domain_line = f"Domain context: {app_description}\n\n" if app_description else ""

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

5. Continue for 10–15 different interaction paths before concluding.
   Vary the entry point each cycle — listing views, form flows, detail pages,
   settings panels, and any navigation item visible in `get_dom_snapshot`.

IMPORTANT
- Be genuinely random and unpredictable — avoid repetitive patterns.
- Test boundary cases: extreme values, many simultaneous filters, rapid clicks.
- Your goal is to find the breaking points the development team missed.
""")

    explorer_agent = create_agent(llm, tools=explorer_tools, system_prompt=explorer_prompt)

    agent_registry = {
        "explorer_agent": explorer_agent,
    }

    workflow = StateGraph(AgentState)  # type: ignore[arg-type]
    workflow.add_node("Supervisor", make_supervisor_node(llm, tuple(agent_registry), app_url, max_steps))  # type: ignore[arg-type]
    for name, agent in agent_registry.items():
        workflow.add_node(name, make_agent_node(agent))  # type: ignore[arg-type]

    return compile_swarm(workflow, agent_registry, checkpointer)
