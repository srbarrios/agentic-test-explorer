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

### Agent Types and Graph Selection
`src/agentic_explorer/orchestration/personas.py` is the **single registry** of personas
(`PERSONAS`): each entry carries the persona's name, diagram label, role, focus, one-line
supervisor description, tier (`standard` | `advanced`) and the `thread_id` keywords that
identify it. `orchestration/graph.py::build_graph(..., tier=...)` builds the supervisor/worker
swarm for one tier; `main.py` builds both and runs each mission on the graph named by its
`graph:` field (`resolve_tier`). A mission without `graph:` falls back to the **deprecated**
`thread_id` keyword inference (`accessibility`, `a11y`, `data_heavy`, `data-heavy`,
`impatient`, `returning`, `explorer`, `chaos`, `autonomous` → advanced; everything else →
standard), which matches plain substrings and prints a warning. PR-generated missions always
set `graph:` explicitly. `standard_graph.py` and `advanced_graph.py` remain only as deprecated
thin wrappers.

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

The generated missions follow the standard YAML format; `_validate_missions` sets an explicit
`graph:` on each one from its persona. Thread IDs use the convention
`pr_{number}_{agenttype}_{nn}`.

* **Standard QA Agents** (tier `standard`): Three agents available for routing by the supervisor:
  * `new_user_agent` — tests onboarding flows, discoverability, default states, and empty states.
  * `power_user_agent` — uses keyboard shortcuts, bulk operations, advanced filters, edge-case workflows.
  * `adversarial_user_agent` — deliberately tries to break things (invalid inputs, back-button abuse).
* **Advanced Testing Agents** (tier `advanced`):
  * `accessibility_user_agent` — validates WCAG compliance, screen reader navigation, keyboard-only interaction.
  * `data_heavy_user_agent` — uploads large files, creates thousands of records, uses long strings.
  * `impatient_user_agent` — cancels operations mid-flight, refreshes during submissions, clicks buttons multiple times.
  * `returning_user_agent` — scenarios for returning users with stale sessions, cached pages, outdated bookmarks.
  * `explorer_agent` — autonomous chaos exploration; uses the full Record-and-Translate engine.

### State Management
* **Persistent Memory**: State is persisted via an SQLite checkpointer (`agent_memory.sqlite`),
  keyed by the `thread_id`. A companion `AsyncSqliteStore` (sharing the same SQLite file,
  optionally configured with an embedding index for semantic search) provides cross-session
  memory. LLM-driven memory operations (procedural reflection, agent observations) use the
  **Langmem SDK**, which only the adapters in `memory_backends.py` import (lazily).
* **Mission Isolation**: Each mission has a unique `thread_id` to isolate its checkpoints;
  reusing a thread ID resumes the prior context. Use `--clear-checkpoints` to reset
  checkpoints while preserving learned memory, `--clear-learned` for the inverse, or
  `--clear-all` for both.
* **Graceful stop and resume** (`runtime/drain.py`): The first Ctrl-C (or SIGTERM) asks
  LangGraph to drain: the step in progress finishes, the checkpoint is saved and the process
  exits with status 130. Running the same command again resumes the interrupted mission
  without repeating the finished step; `traces.log` is appended to and marked
  `=== RESUMED ===`. A second Ctrl-C aborts immediately.
* **Node resilience** (`orchestration/resilience.py`, `config.yaml` `runtime:`): every agent
  and the supervisor get an *idle* timeout (`node_idle_timeout_s`, renewed by any LLM or tool
  activity inside the node, so long healthy turns are never cut) and a retry for nodes that
  timed out. An agent that still fails becomes a `[NODE FAILED]` message and control returns
  to the supervisor via the Summarizer. Transient API errors (429/503/529) are deliberately
  **not** handled there: they are re-raised so the mission-level `with_retry` can wait and
  resume from the checkpoint.
* **`AgentState`**: A unified state `TypedDict` that carries `messages`,
  `next_agent`, `step_count`, `action_tape` (bounded to 50 entries in state; full log
  persisted to JSONL on disk), `bugs_found`, `uncertain_findings`, `bug_evidence` and
  `explored_paths`. `messages` uses the `add_messages` reducer (so `RemoveMessage` really
  deletes history); the other lists use `operator.add` and arrive as per-node **deltas**.
  `bugs_found` holds **confirmed** findings only (`"title: summary"` strings); findings the
  agent marked `uncertain` go to `uncertain_findings` and are never counted as bugs;
  `bug_evidence` holds the observed / expected / where record of each finding. The
  `<bugs_found>` parser, the dedup key and the contract validator live in
  `orchestration/findings.py`.
* **Message Summarization**: A `Summarizer` node between agents and the Supervisor
  compresses old messages to prevent unbounded state growth. Keeps the first HumanMessage
  (mission prompt) and the 20 most recent messages; the messages in between are replaced by
  one deterministic compact summary (no LLM call) placed right after the mission prompt.
  Agent nodes return only the messages they produced, never the replayed history. Agents are
  given a text-only copy of the history (`_sanitize_messages_for_model` drops system and tool
  messages): the summarizer can cut an AI tool-call / tool-result pair in half, which
  Anthropic rejects.
* **Cross-Session Memory** (`src/agentic_explorer/memory.py`): Four levels of memory backed
  by the LangGraph Store, with LLM-driven operations behind the `PromptOptimizer` and
  observation-tool adapters of `memory_backends.py` (currently **Langmem**):
  - **Semantic** — page knowledge, selector reliability tracking, application quirks
    (written automatically on every agent turn from action tape entries), plus
    Langmem-managed agent observations (recorded proactively via `record_observation` tool).
  - **Episodic** — session summaries, deduplicated bug catalog. Written after each mission
    completes. Agents query past findings via the `recall_past_findings` tool, which uses
    semantic vector search when an embedding index is configured (falls back to keyword
    matching otherwise).
  - **Procedural** — per-agent prompt supplements and supervisor routing rules. Optimized
    after each batch through a `PromptOptimizer` (Langmem's `create_prompt_optimizer` by
    default); read at graph construction time to evolve agent prompts and routing strategy.
    Each mission is attributed to the persona its `thread_id` names
    (`persona_for_thread_id`), because agents read their supplement by persona name; ids
    that follow no persona convention (e.g. `smoke_checkout_01`) feed only the routing rules.
  - **Prioritization** — risk-scored page ranking (bug density, selector flakiness, quirk
    count). Injected into supervisor routing context automatically.

  Namespace layout:
  ```
  ("app", "{url_hash}", "pages"|"selectors"|"quirks"|"agent_observations")
  ("episodes", "{url_hash}", "sessions"|"bugs")
  ("procedures", "{url_hash}", "agent_prompts"|"routing_rules")
  ```

  **Memory is best-effort, never silent** (`memory_health.py`): a broken store must not abort
  a mission, but every failed memory operation is logged with its traceback to the
  `agentic_explorer.memory` logger, counted, and warned about once per kind on the console.
  `MissionResult.memory_errors` and a `Memory errors` line in `test_report.md` show how many
  failed during a mission, so a run that learned nothing is visible.

## Configuration Surface
The framework is configured through three user-supplied files (templates ship as
`*.example`):

* **`.env`** — `APP_URL`, `APP_USERNAME`, `APP_PASSWORD` are required. Optional:
  `APP_REVISION` (revision of the app under test) and `AGENT_SNAPSHOT_KEEP` (per-step snapshots
  to keep, default 200, `0` = off). `config.yaml` also takes `app.revision` and `app.seed`.
  **LLM auth** — the framework supports Claude (default) and Gemini. Claude auth: set
  `ANTHROPIC_API_KEY` (direct API), or the framework reads `~/.claude/settings.json`
  for Vertex AI config automatically. Gemini auth: set `GOOGLE_API_KEY` (API key), or
  leave it unset and the framework loads `~/.gemini/oauth_creds.json` (produced by
  `gemini auth login`). Set `LLM_PROVIDER` to force a specific provider.
  Optional: `APP_CONFIG`, `MCP_SERVERS_CONFIG`, `AGENT_SKILLS_ROOT`,
  `AGENT_SKILL_SCRIPT_TIMEOUT`, `ANTHROPIC_VERTEX_REGION`, `CLAUDE_MODEL`, `CLAUDE_VISION_MODEL`,
  `CLAUDE_REPORT_MODEL`, `CLAUDE_SCENARIO_MODEL`, `GEMINI_MODEL`, `GEMINI_VISION_MODEL`,
  `GEMINI_REPORT_MODEL`, `GEMINI_SCENARIO_MODEL`, `SCENARIO_MODEL` (provider-agnostic).
* **`config.yaml`** (loaded by `src/agentic_explorer/config.py`) — `app.{name,url,description}`,
  `auth.{method,selectors,post_login_check}`, `paths.{mcp_servers,skills_root}`,
  `llm.{provider,claude_model,gemini_model,claude_vision_model,gemini_vision_model,
  embedding_model,embedding_dims}`,
  `runtime.{node_idle_timeout_s,node_run_timeout_s,node_retries}` (node timeouts; `0`
  disables a timeout). Supports `${ENV_VAR}` interpolation. Embedding config
  enables semantic search in long-term memory (e.g. `embedding_model: google-genai:models/embedding-001`,
  `embedding_dims: 768`). Also overridable via `EMBEDDING_MODEL` and `EMBEDDING_DIMS` env vars.
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
      Playwright script. Optional `observed` / `expected` arguments turn the last successful
      `extract_text` step into a real `toContain` assertion. The spec header carries the tape
      provenance and a setup check throws `SETUP_FAILURE` before the `BUG STEPS` marker. The spec
      uses the recording's action/navigation timeouts (`ACTION_TIMEOUT_MS`,
      `NAVIGATION_TIMEOUT_MS`) and warns when it asserts no symptom (a pass is not a fix).
    * Raw `PlayWrightBrowserToolkit` tools (`click_element`, `navigate_browser`, etc.) are
      **filtered out** from agents; they exist only for the monkey-patch self-healing
      mechanism.
* **MCP (Model Context Protocol) Tools** (`get_mcp_tools` in
  `src/agentic_explorer/tools/common/custom_tools.py`): Loads tools from any MCP servers
  defined in `mcp_servers.json`. The framework ships zero hardcoded servers — bring your own.
* **Agent Skills** (`fetch_agent_skill`, `run_agent_skill_script`): Discover and execute
  skills installed under `AGENT_SKILLS_ROOT`, following the
  [agentskills.io](https://agentskills.io/specification) progressive-disclosure model.
  `fetch_agent_skill` returns `SKILL.md` and a `references/` manifest first; pass
  `include_references=true` with a specific `reference_path` only when detailed reference
  material is required.
* **Screenshots & Reproductions**: The screenshot tool captures full-page bug evidence and
  is thread-aware via `RunnableConfig`. The browser engine translates Action Tape entries
  into reproducible Playwright specs.
* **Memory Recall** (`recall_past_findings` in `src/agentic_explorer/memory.py`): Agents
  can call this tool to query the bug catalog, session history, and quirks. Uses semantic
  vector search when the store has an embedding index configured; falls back to keyword
  matching otherwise. Added to agent tool bundles automatically when a store is available.
* **Observation Recording** (`record_observation`, built by
  `memory_backends.make_observation_tool` on Langmem's `create_manage_memory_tool`):
  Agents can proactively record high-level observations
  about the application (unexpected behaviors, UX issues, successful strategies). These
  observations are stored in the `agent_observations` namespace and surfaced in the
  supervisor's `MEMORY_CONTEXT` in subsequent runs.
* **Visual Mode** (`src/agentic_explorer/ui/`): Optional Streamlit-based real-time dashboard
  for monitoring swarm execution. Uses a one-way "spectator" pattern with zero performance
  overhead. The main process writes state to `.agent_state.json` (atomic via `os.replace`)
  and async screenshots to `.latest_vision.jpg` (fire-and-forget). The dashboard polls
  these files every ~1s. Enabled via `--visual` flag; requires `pip install -e ".[visual]"`.
  Implementation: `state_emitter.py` (non-blocking state bridge), `swarm_diagram.py`
  (Mermaid generator), `dashboard.py` (Streamlit app).

## Running the System

### Initial Setup
1. **Install dependencies**: `pip install -e .` (or `uv pip install -e .`).
   Re-run this after every `git pull` to pick up new or updated packages. Key packages:
   `langchain`, `langchain-anthropic`, `langchain-google-genai`, `langchain-google-vertexai`,
   `langgraph`, `langmem`, `playwright`, `python-dotenv`, `pyyaml`, `pillow`,
   `langchain-mcp-adapters`, `langgraph-checkpoint-sqlite`, `aiosqlite`, `httpx`.
   Use Python 3.11–3.12 (3.14 cannot build `pyarrow`, pulled in by
   `langchain-google-vertexai`). For a reproducible install use
   `uv pip install -r requirements.txt && uv pip install -e . --no-deps`; regenerate that
   lock with `uv pip compile pyproject.toml -o requirements.txt --python-version 3.11`
   whenever `pyproject.toml` changes (CI verifies it stays installable). `mcp` is capped
   at `<2` because `langchain-mcp-adapters` 0.2.x breaks with `mcp` 2.x.
   Developer extras: `uv pip install -e ".[dev]"` (pytest).
2. **Install browser**: Run `playwright install chromium`.
3. **Configure**: Copy `.env.example` → `.env`, `config.yaml.example` → `config.yaml`, and
   (optionally) `mcp_servers.json.example` → `mcp_servers.json`. Fill in your app's URL,
   credentials, and login selectors.
4. **Authenticate**: Run `agent-auth` for one-time login (saves `auth.json`).

### Developer Workflows
Execute missions defined in YAML format (see `missions/README.md`):
* **Standard Run**: `agent-explorer --missions missions/new_user_agent.yaml`
* **Explicit Provider**: `agent-explorer --missions missions/power_user_agent.yaml --provider claude`
* **Headed Mode** (Debugging): `agent-explorer --missions missions/accessibility_user_agent.yaml --headed`
* **Visual Mode** (Real-Time Dashboard): `agent-explorer --missions missions/explorer_agent.yaml --visual --headed`
  — launches Streamlit dashboard at `http://localhost:8501` showing live screenshots, swarm
  state diagram, thought stream, and action tape. Requires `pip install -e ".[visual]"`.
  Zero performance overhead when not used.
* **Clear All Memory**: `agent-explorer --missions missions/new_user_agent.yaml --clear-all`
* **Clear Checkpoints Only**: `agent-explorer --missions missions/new_user_agent.yaml --clear-checkpoints`
  (preserves learned memory: pages, bugs, procedures)
* **Clear Learned Only**: `agent-explorer --missions missions/new_user_agent.yaml --clear-learned`
  (preserves checkpoints for resume)
* **Custom Step Limit**: `agent-explorer --missions missions/new_user_agent.yaml --max-steps 50`
  (default: 30; supervisor resets to the app homepage on limit and tries a new strategy)
* **Stop and Resume**: press Ctrl-C once to stop at a safe point (exit status 130) and run
  the same command again to resume the interrupted mission; press it twice to abort now.

PR-driven test generation (prefers GitHub MCP server; falls back to [`gh` CLI](https://cli.github.com/)):
* **Generate Only**: `agent-explorer --pr-url https://github.com/org/repo/pull/123`
  — writes `missions/pr_123.yaml`; injects historical bug data when available
* **Generate + Execute**: `agent-explorer --pr-url https://github.com/org/repo/pull/123 --execute --headed`
* **Custom Output Dir**: `agent-explorer --pr-url <url> --output-dir ./pr-missions`
* **Combined**: `agent-explorer --missions missions/new_user_agent.yaml --pr-url <url> --execute`
  — runs both hand-written and auto-generated missions

Regression testing and model export:
* **Regression**: `agent-explorer --regression --headed`
  — auto-generates missions from the bug catalog targeting known open bugs
* **Export Model**: `agent-explorer --export-model`
  — exports discovered app structure as `app_model.json` (no browser needed)

### Tests
`uv pip install -e ".[dev]"` then `pytest -q`. The suite needs no LLM, browser or network:
agents run on fake chat models, and the streaming, resilience and stop/resume tests use
real compiled LangGraph graphs (others use small fakes for speed). Nothing in the suite
talks to a real LLM, so model behaviour itself is not covered. CI runs it on Python 3.11
and 3.12 and also checks that `requirements.txt` stays installable.

## Output Artifacts
Every mission generates artifacts localized in a `report_<thread_id>/` directory:
* `traces.log`: Full message history with tool calls and responses (each message once;
  appended, with a `=== RESUMED ===` marker, when a stopped mission is resumed).
* `test_report.md`: An LLM-generated summary detailing actions, issues, Action Tape stats,
  and status, plus a `Memory errors` count when a memory operation failed. Code (not the
  LLM) appends the findings: confirmed bugs with Observed / Expected / Where and a separate
  **Not determined** section.
* `action_tape.jsonl`: Immutable, line-delimited JSON log of every deterministic browser
  command executed (used for reproduction). Entries may carry `snapshot_ref`. Values typed
  into secret fields (`tools/browser/redaction.py`) are recorded as `<redacted>`; the spec
  reads them from `REPLAY_SECRET_n` environment variables.
* `tape_meta.json`: Provenance written once per mission (app name/URL, app revision from
  `--app-revision` > `APP_REVISION` > `app.revision`, session start, SHA-256 of `auth.json` —
  never its content —, tool version, `app.seed` preconditions). A sidecar on purpose, so the
  tape itself only holds actions.
* Replay: `agent-explorer --replay <thread_id>` (`runtime/replay.py`) re-executes the tape in a
  fresh browser with no LLM and exits 0 `REPRODUCED`, 1 `NOT_REPRODUCED`, 2 `SETUP_FAILURE`,
  3 `DIVERGED`, 4 `INCONCLUSIVE`; it writes `replay.json`. Redacted values come from
  `REPLAY_SECRET_n`. Replaying has the mission's side effects. After a `fill` the tape entry
  carries `effect.value_matches` (never the value).
* Session loss: with `auth.json` loaded, `tools/browser/session.py` turns a redirect to a
  login page into `STATUS: AUTH_REQUIRED`, marks the tape entry `auth_required`, and drains the
  run (exit 130). Re-run `agent-auth`, then the same command to resume. `tape_meta.json`
  records each later `auth.json` fingerprint in `auth_history`.
* `snapshots/step_NNNN.txt`: The exact DOM snapshot the agent saw after each command, capped
  per file; the last `AGENT_SNAPSHOT_KEEP` (default 200, `0` = off) are kept.
* `reproduction_*.spec.ts`: Auto-generated Playwright TypeScript test files. Run with
  `npx playwright test reproduction_*.spec.ts --headed`.
* `screenshots/`: Full-page screenshots captured when bugs are detected.

## Conventions When Changing Code
* **Context Disclosure Rule**: Keep prompts compact enough for regular Opus-class context
  windows. Do not replay full histories unless absolutely necessary. The supervisor routes
  from a compact mission brief, recent progress, recent Action Tape entries, bugs, and
  explored URLs. Report generation preserves the start and latest outcome of long transcripts
  while omitting the middle. PR scenario generation sends a budgeted diff excerpt rather than
  the full diff. Tunables: `PR_PROMPT_DIFF_BUDGET_CHARS`, `PR_PROMPT_BODY_BUDGET_CHARS`,
  `PR_PROMPT_FILE_LIST_LIMIT`, and `PR_GENERATED_MISSION_PROMPT_MAX_CHARS`.
* **Global QA Rule**: Strictly enforced in
  `src/agentic_explorer/orchestration/graph_base.py` via compact shared prompt helpers — agents
  must consult MCP/Skills first when available, never guess, and capture screenshot evidence
  + call `generate_reproduction_spec` immediately on failures.
* **Selector Policy (Engine-Enforced)**: `execute_browser_command` rejects brittle selectors
  at runtime. Priority order:
    1. `data-test-subj` attributes → `[data-test-subj='myButton']`
    2. ARIA labels / roles → `[aria-label='Search']`, `role='dialog'`
    3. Semantic HTML / visible text → `button:has-text('Save')`, `text='Apply'`
    * **Forbidden**: XPath (`//div`), positional CSS (`:nth-child(3)`,
      `div > span:nth-of-type(2)`). Always call `get_dom_snapshot` first.
* **Graph Builders Are Async**: `orchestration.graph.build_graph(..., tier=...)` is an
  `async` function (it reads procedural memory from the store at construction time). Always
  `await` it. `standard_graph.build_graph` and `advanced_graph.build_advanced_graph` are
  deprecated wrappers; do not call them from new code.
* **Memory Module**: All cross-session memory logic lives in `src/agentic_explorer/memory.py`.
  Do not scatter store reads/writes across other files — add new memory functions there and
  import them. Only `memory_backends.py` may import `langmem` (it is unmaintained since
  Oct 2025 and pulls a LangGraph import that LangGraph 2.0 removes), and only lazily. A
  best-effort memory operation must go through `guarded(...)` from `memory_health.py`:
  never write `except Exception: pass`. Do not rely on `store.asearch(query=...)` to filter
  unless `store.index_config` is set: without an embedding index the query is ignored.
  Store namespace conventions: `("app", hash, ...)` for semantic,
  `("episodes", hash, ...)` for episodic, `("procedures", hash, ...)` for procedural.
* **Agent Modification**: To add a persona, add one `Persona` entry to `PERSONAS` in
  `orchestration/personas.py` (role, focus, supervisor description, tier, thread-id
  keywords). Graph nodes, the Mermaid diagram labels and the PR keyword list derive from it.
  Also add its description to `pr_analyzer._SYSTEM_PROMPT` and a `missions/<name>.yaml`
  template with `graph:` set; tests fail if either is missing. The tool bundle is shared by
  all personas. A persona automatically receives procedural supplements stored under its
  name.
* **Mission Modifications**: A mission picks its graph with `graph: standard|advanced`.
  Do not depend on `thread_id` keywords: that inference is deprecated and matches plain
  substrings (`test_new_user_explorer_notes` would run on the advanced graph).
* **PR Analyzer Modifications**: The PR scenario generation prompt is in
  `src/agentic_explorer/pr_analyzer.py` (`_SYSTEM_PROMPT`). When adding a new agent type,
  also add its description to this prompt so the LLM can route PR changes to it. Generated
  thread IDs must follow `pr_{number}_{agenttype}_{nn}` and contain a persona keyword
  (`ALL_KEYWORDS` in `personas.py`).
* **Testing Streaming Changes**: A fake graph that you wrap in the expected stream shape
  cannot prove LangGraph really emits that shape. Any change to how the stream is consumed
  needs at least one test on a real compiled graph (see
  `test_runs_end_to_end_on_a_real_langgraph_v2_stream` in `tests/test_runner.py`).
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
* **Visual Mode Integration**: When adding new state fields to the swarm, emit them via
  `state_emitter.update()` from the appropriate integration point (supervisor for step/bug
  counts, agent nodes for thoughts, browser engine for actions). Always check
  `state_emitter.is_enabled()` before emitting. See existing integrations in
  `runtime/runner.py`, `graph_base.py`, and `engine.py` for patterns.
