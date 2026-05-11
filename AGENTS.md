# AGENTS.md

## Project Overview
This repository is a product-agnostic Agentic Test Explorer Proof-of-Concept (PoC). It performs
automated, autonomous exploratory QA against any web application configured by the user. It runs
on an async Python runtime utilizing a LangGraph supervisor-worker swarm architecture, Playwright
for browser automation, and Claude (default) or Gemini models. Application-specific details
(URL, credentials, auth selectors, MCP servers, Skills) are supplied through `config.yaml`,
`.env`, and `mcp_servers.json` — none are baked into the codebase.

The framework also supports **PR-driven test generation**: given a GitHub Pull Request URL, it
extracts the code diff (preferring the **GitHub MCP server** at
`https://api.githubcopilot.com/mcp/`, falling back to the `gh` CLI) and uses an LLM to
auto-generate targeted mission YAML covering the UI areas impacted by the changes.

## Architecture & Swarm Pattern
The system employs a Supervisor-Worker pattern implemented with LangGraph, where a Supervisor
node routes tasks with structured enum outputs to specialized agents and loops workers back to
itself until a `FINISH` state is reached.

### Record-and-Translate Browser Engine
A core architectural principle is the **brain/hands separation** implemented in
`src/agentic_explorer/tools/browser/engine.py`:

* **Brain** — LangGraph agents emit strict JSON intents (e.g.,
  `{"action":"click","selector":"[data-test-subj='submitButton']"}`).
* **Hands** — The deterministic engine parses, validates, and executes each command with
  Playwright.
* **Action Tape** — Every command is appended to a per-thread, immutable
  `action_tape.jsonl` log stored in `report_<thread_id>/`.
* **Reproduction** — When a bug is found, `generate_reproduction_spec` translates the Action
  Tape into a runnable `reproduction_*.spec.ts` Playwright test file.

Supported JSON actions: `navigate`, `click`, `fill`, `press`, `select_option`, `hover`,
`wait_for`, `scroll`, `extract_text`, `snapshot`.

### Agent Types
`main.py` automatically compiles either a standard or advanced graph based on `thread_id`
keywords (`explorer`, `chaos`, `autonomous` → advanced; everything else → standard).

### PR-Driven Scenario Generation
`src/agentic_explorer/pr_analyzer.py` provides a pipeline that sits **before** mission
execution:
1. `parse_pr_url()` — extracts `(owner, repo, number)` from a GitHub PR URL.
2. `fetch_pr_data()` — **MCP-first with gh fallback**:
   - Looks for a `"github"` entry in `mcp_servers.json` (supports both `mcpServers`/`transport`
     and `servers`/`type` formats).
   - If found, connects via `MultiServerMCPClient` and calls `get_pull_request`,
     `get_pull_request_files`, and `get_pull_request_diff` MCP tools concurrently.
   - If the MCP server is not configured, unreachable, or missing required tools, falls back
     to concurrent `gh` CLI subprocess calls.
   - Truncates diffs exceeding 100K chars.
3. `generate_missions_from_pr()` — assembles the PR data and app context into a structured
   prompt, calls the configured LLM (Claude or Gemini), parses the YAML response, validates
   the mission structure, and retries on parse failures or transient API errors.

The generated missions follow the standard YAML format and are routed to agents via the same
`thread_id`-keyword mechanism. Thread IDs use the convention `pr_{number}_{agenttype}_{nn}`.

* **Standard QA Agents** (`src/agentic_explorer/orchestration/standard_graph.py`): Thirteen agents available for routing by the supervisor:
  * **5 UI-pattern specialists**: Tools are split by modality (DOM-only vs. visual validation).
    * `listing_agent` (DOM tools) — lists, tables, search/filter bars, pagination, row flyouts.
    * `graph_agent` (DOM + visual) — node-link graphs, timelines, waterfalls, complex SVG/Canvas.
    * `chart_agent` (DOM + visual) — time-series, bar/line/area charts, KPI tiles, dashboards.
    * `map_agent` (DOM + visual) — geographic maps, status grids, geospatial overlays.
    * `form_agent` (DOM tools) — forms, wizards, validation flows, multi-step configuration.
  * **8 Generic Exploration Personas**: These personas apply behavioral strategies to the application.
    * `new_user_agent` — tests onboarding flows, discoverability, default states, and empty states.
    * `power_user_agent` — uses keyboard shortcuts, bulk operations, advanced filters, edge-case workflows.
    * `adversarial_user_agent` — deliberately tries to break things (invalid inputs, back-button abuse).
    * `impatient_user_agent` — cancels operations mid-flight, refreshes during submissions, clicks buttons multiple times.
    * `accessibility_user_agent` — validates WCAG compliance, screen reader navigation, keyboard-only interaction.
    * `constrained_user_agent` — tests degraded-experience paths and responsive design for slow networks and small viewports.
    * `data_heavy_user_agent` — uploads large files, creates thousands of records, uses long strings.
    * `returning_user_agent` — scenarios for returning users with stale sessions, cached pages, outdated bookmarks.
* **Advanced Testing Agents** (`src/agentic_explorer/orchestration/advanced_graph.py`):
    * `explorer_agent` — autonomous chaos exploration; uses the full Record-and-Translate
      engine. The advanced graph's single-agent supervisor is preserved as an extension point
      for additional specialized agents.

### State Management
* **Persistent Memory**: State is persisted via an SQLite checkpointer (`agent_memory.sqlite`),
  keyed by the `thread_id`.
* **Mission Isolation**: Each mission has a unique `thread_id` to isolate its memory; reusing a
  thread ID resumes the prior context.
* **`AgentState` / `AdvancedAgentState`**: Both state TypedDicts carry `messages`,
  `next_agent`, `step_count`, and `action_tape` (an append-only list of recorded browser
  commands).

## Configuration Surface
The framework is configured through three user-supplied files (templates ship as
`*.example`):

* **`.env`** — `APP_URL`, `APP_USERNAME`, `APP_PASSWORD` are required.
  **LLM auth** — the framework supports Claude (default) and Gemini. Claude auth: set
  `ANTHROPIC_API_KEY` (direct API), or the framework reads `~/.claude/settings.json`
  for Vertex AI config automatically. Gemini auth: set `GOOGLE_API_KEY` (API key), or
  leave it unset and the framework loads `~/.gemini/oauth_creds.json` (produced by
  `gemini auth login`). Set `LLM_PROVIDER` to force a specific provider.
  Optional: `APP_CONFIG`, `MCP_SERVERS_CONFIG`, `AGENT_SKILLS_ROOT`,
  `AGENT_SKILL_SCRIPT_TIMEOUT`, `CLAUDE_MODEL`, `CLAUDE_VISION_MODEL`,
  `CLAUDE_REPORT_MODEL`, `CLAUDE_SCENARIO_MODEL`, `GEMINI_MODEL`, `GEMINI_VISION_MODEL`,
  `GEMINI_REPORT_MODEL`, `GEMINI_SCENARIO_MODEL`.
* **`config.yaml`** (loaded by `src/agentic_explorer/config.py`) — `app.{name,url,description}`,
  `auth.{method,selectors,post_login_check}`, `paths.{mcp_servers,skills_root}`,
  `llm.{provider,claude_model,gemini_model,claude_vision_model,gemini_vision_model}`.
  Supports `${ENV_VAR}` interpolation.
* **`mcp_servers.json`** — Claude-Desktop-compatible MCP server map. Optional; agents run
  without MCP tools if missing or empty. A `"github"` entry enables MCP-based PR data
  fetching (preferred over `gh` CLI):
  ```json
  {"mcpServers": {"github": {"transport": "http", "url": "https://api.githubcopilot.com/mcp/"}}}
  ```

## Custom Tools & Integrations
* **Record-and-Translate Browser Tools** (`src/agentic_explorer/tools/browser/engine.py`):
    * `execute_browser_command` — dispatches a JSON intent to Playwright, records to the
      Action Tape, and returns the resulting DOM snapshot.
    * `get_dom_snapshot` — read-only Accessibility Tree / JS DOM digest; does **not** write to
      the tape.
    * `generate_reproduction_spec` — translates the Action Tape into a `reproduction_*.spec.ts`
      Playwright script.
    * Raw `PlayWrightBrowserToolkit` tools (`click_element`, `navigate_browser`, etc.) are
      **filtered out** from agents; they exist only for the monkey-patch self-healing
      mechanism.
* **MCP (Model Context Protocol) Tools** (`get_mcp_tools` in
  `src/agentic_explorer/tools/common/custom_tools.py`): Loads tools from any MCP servers
  defined in `mcp_servers.json`. The framework ships zero hardcoded servers — bring your own.
* **Agent Skills** (`fetch_agent_skill`, `run_agent_skill_script`): Discover and execute
  skills installed under `AGENT_SKILLS_ROOT`, following the
  [agentskills.io](https://agentskills.io/specification) progressive-disclosure model.
* **Vision & Screenshots**: The screenshot tool captures full-page bug evidence and is
  thread-aware via `RunnableConfig`. The `analyze_visual_state` tool uses the configured
  AI vision model (Claude or Gemini) to validate UI rendering.

## Running the System

### Initial Setup
1. **Install dependencies**: `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`).
   Re-run this after every `git pull` to pick up new or updated packages. Key packages:
   `langchain`, `langchain-anthropic`, `langchain-google-genai`, `langchain-google-vertexai`,
   `langgraph`, `playwright`, `python-dotenv`, `pyyaml`, `pillow`, `langchain-mcp-adapters`,
   `langgraph-checkpoint-sqlite`, `aiosqlite`, `httpx`.
2. **Install browser**: Run `playwright install chromium`.
3. **Configure**: Copy `.env.example` → `.env`, `config.yaml.example` → `config.yaml`, and
   (optionally) `mcp_servers.json.example` → `mcp_servers.json`. Fill in your app's URL,
   credentials, and login selectors.
4. **Authenticate**: Run `agent-auth` for one-time login (saves `auth.json`).

### Developer Workflows
Execute missions defined in YAML format (see `missions/README.md`):
* **Standard Run**: `agent-explorer --missions missions/smoke.yaml`
* **Explicit Provider**: `agent-explorer --missions missions/smoke.yaml --provider claude`
* **Headed Mode** (Debugging): `agent-explorer --missions missions/smoke.yaml --headed`
* **Clear Memory**: `agent-explorer --missions missions/smoke.yaml --clear-memory`
* **Custom Step Limit**: `agent-explorer --missions missions/smoke.yaml --max-steps 50`
  (default: 30; supervisor resets to the app homepage on limit and tries a new strategy)

PR-driven test generation (prefers GitHub MCP server; falls back to [`gh` CLI](https://cli.github.com/)):
* **Generate Only**: `agent-explorer --pr-url https://github.com/org/repo/pull/123`
  — writes `missions/pr_123.yaml`
* **Generate + Execute**: `agent-explorer --pr-url https://github.com/org/repo/pull/123 --execute --headed`
* **Custom Output Dir**: `agent-explorer --pr-url <url> --output-dir ./pr-missions`
* **Combined**: `agent-explorer --missions missions/smoke.yaml --pr-url <url> --execute`
  — runs both hand-written and auto-generated missions

## Output Artifacts
Every mission generates artifacts localized in a `report_<thread_id>/` directory:
* `traces.log`: Full message history with tool calls and responses.
* `test_report.md`: An LLM-generated summary detailing actions, issues, Action Tape stats,
  and status.
* `action_tape.jsonl`: Immutable, line-delimited JSON log of every deterministic browser
  command executed (used for reproduction).
* `reproduction_*.spec.ts`: Auto-generated Playwright TypeScript test files. Run with
  `npx playwright test reproduction_*.spec.ts --headed`.
* `screenshots/`: Full-page screenshots captured when bugs are detected.

## Conventions When Changing Code
* **Global QA Rule**: Strictly enforced in
  `src/agentic_explorer/orchestration/standard_graph.py` and `advanced_graph.py` — agents
  must consult MCP/Skills first when available, never guess, and capture screenshot evidence
  + call `generate_reproduction_spec` immediately on failures.
* **Selector Policy (Engine-Enforced)**: `execute_browser_command` rejects brittle selectors
  at runtime. Priority order:
    1. `data-test-subj` attributes → `[data-test-subj='myButton']`
    2. ARIA labels / roles → `[aria-label='Search']`, `role='dialog'`
    3. Semantic HTML / visible text → `button:has-text('Save')`, `text='Apply'`
    * **Forbidden**: XPath (`//div`), positional CSS (`:nth-child(3)`,
      `div > span:nth-of-type(2)`). Always call `get_dom_snapshot` first.
* **Agent Modification**: If adding a new agent, you must update the system prompt, tool
  bundle, node wrapper function, supervisor enum, and conditional routing table.
* **Mission Modifications**: If adding a new advanced mission category, update both the
  mission `thread_id` naming and the `ADVANCED_KEYWORDS` tuple in `main.py`.
* **PR Analyzer Modifications**: The PR scenario generation prompt is in
  `src/agentic_explorer/pr_analyzer.py` (`_SYSTEM_PROMPT`). When adding a new agent type,
  also add its description to this prompt so the LLM can route PR changes to it. Generated
  thread IDs must follow `pr_{number}_{agenttype}_{nn}` and respect `ADVANCED_KEYWORDS`.
* **LLM Configuration**: The system supports Claude (default) and Gemini providers. The
  provider is auto-detected from credentials or set explicitly via `LLM_PROVIDER` env var,
  `--provider` CLI flag, or `config.yaml > llm.provider`. Smart model defaults are chosen
  per auth method (see `.env.example`). Override models via `CLAUDE_MODEL`,
  `CLAUDE_VISION_MODEL`, `CLAUDE_REPORT_MODEL`, `CLAUDE_SCENARIO_MODEL` (Claude) or
  `GEMINI_MODEL`, `GEMINI_VISION_MODEL`, `GEMINI_REPORT_MODEL`, `GEMINI_SCENARIO_MODEL`
  (Gemini). The LLM is instantiated through `make_llm()` in
  `src/agentic_explorer/utils/llm.py` — never construct `ChatAnthropic`,
  `ChatAnthropicVertex`, or `ChatGoogleGenerativeAI` directly anywhere in the codebase.
  The function accepts optional `model_name` and `provider` parameters for callers that
  need a specific model or provider override.
* **Tool Outputs**: Keep tool outputs as machine-consumable strings or JSON, as many agent
  prompts expect parseable responses.
* **App-Specific Details**: NEVER hardcode application URLs, credentials, selectors, MCP
  server URLs, or skill names in source. Read them from `config.yaml` / env / user files.
