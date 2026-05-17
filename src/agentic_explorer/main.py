import asyncio
import os
import argparse
import random
import warnings
import yaml
from typing import Any

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# Suppress library warnings before any imports that trigger them.
warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r".*allowed_objects.*",
)

from agentic_explorer.utils import console  # noqa: E402

from langchain_core.messages import (  # noqa: E402
    HumanMessage, AIMessage, AIMessageChunk, SystemMessage,
)

from playwright.async_api import async_playwright

from agentic_explorer.tools.browser.engine import get_action_tape
from agentic_explorer.config import load_app_config, load_environment
from agentic_explorer.utils.llm import make_llm, get_model_name, get_active_provider

from agentic_explorer.tools.common.custom_tools import (
    get_mcp_tools,
    fetch_agent_skill,
    run_agent_skill_script,
)
from agentic_explorer.orchestration.standard_graph import build_graph
from agentic_explorer.orchestration.advanced_graph import build_advanced_graph

load_environment()

# Mission-type detection: thread_ids matching these substrings are routed to the
# advanced graph instead of the standard 3-persona swarm.
ADVANCED_KEYWORDS = (
    "accessibility",
    "a11y",
    "data_heavy",
    "data-heavy",
    "impatient",
    "returning",
    "explorer",
    "chaos",
    "autonomous",
)

REPORT_TRANSCRIPT_MAX_CHARS = 35_000
REPORT_TRANSCRIPT_HEAD_CHARS = 6_000


def _clip_for_prompt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 24:
        return text[:max(0, max_chars)]
    return text[: max_chars - 24].rstrip() + " … [truncated]"


def _bound_transcript_for_report(transcript: str) -> str:
    """Keep report context within a predictable prompt budget.

    Reports need the mission start plus the latest outcomes most. Preserve both
    ends and disclose how much was omitted instead of sending unbounded history.
    """
    if len(transcript) <= REPORT_TRANSCRIPT_MAX_CHARS:
        return transcript
    tail_chars = REPORT_TRANSCRIPT_MAX_CHARS - REPORT_TRANSCRIPT_HEAD_CHARS - 160
    omitted = len(transcript) - REPORT_TRANSCRIPT_HEAD_CHARS - tail_chars
    return (
        transcript[:REPORT_TRANSCRIPT_HEAD_CHARS].rstrip()
        + f"\n\n... [omitted {omitted:,} chars of middle transcript for context budget] ...\n\n"
        + transcript[-tail_chars:].lstrip()
    )


def _get_async_sqlite_saver() -> Any:
    """Import AsyncSqliteSaver while suppressing a known upstream pending warning."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver


def _log_message_summary(msg) -> None:
    """Print a compact one-line summary of a LangChain message to the console."""
    content = msg.content
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and "text" in b
        )
    text = str(content).strip().replace("\n", " ")
    if len(text) > 120:
        text = text[:117] + "..."

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        names = ", ".join(tc["name"] for tc in tool_calls)
        console.info(f"{msg.type}: [{names}] {text[:80]}")
    elif text:
        console.info(f"{msg.type}: {text}")


def _strip_extras(msg) -> None:
    """Remove the noisy 'extras' block from list-content messages in place."""
    if isinstance(msg.content, list):
        for block in msg.content:
            if isinstance(block, dict) and "extras" in block:
                del block["extras"]


def _message_text(msg) -> str:
    """Flatten BaseMessage.content (list-of-blocks or str) to plain text."""
    content = msg.content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return str(content)


def _summarize_tool_args(tool_call: dict, max_chars: int = 80) -> str:
    """Return a compact one-line summary of a tool_call args dict."""
    import json
    args = tool_call.get("args") or {}
    if not isinstance(args, dict):
        text = str(args)
    elif "action" in args:
        # execute_browser_command JSON-intent shape — surface action+key fields cleanly.
        parts = [f"action={args['action']}"]
        for key in ("selector", "url", "value", "key"):
            if args.get(key):
                val = str(args[key])
                if len(val) > 40:
                    val = val[:37] + "..."
                parts.append(f"{key}={val!r}")
        text = " ".join(parts)
    else:
        try:
            text = json.dumps(args, ensure_ascii=False, default=str)
        except Exception:
            text = str(args)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(x in msg for x in ("503", "UNAVAILABLE", "429", "RATE_LIMIT", "RESOURCE_EXHAUSTED", "QUOTA"))


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(x in msg for x in ("429", "RATE_LIMIT", "RESOURCE_EXHAUSTED", "QUOTA"))


async def run_missions():
    parser = argparse.ArgumentParser(description="Run Agentic Exploratory Tests")
    parser.add_argument("--missions", type=str, default=None, help="Path to the YAML missions file")
    parser.add_argument("--pr-url", type=str, default=None, help="GitHub PR URL to analyze and generate test missions from")
    parser.add_argument("--execute", action="store_true", help="Execute generated PR missions immediately (default: generate only)")
    parser.add_argument("--output-dir", type=str, default="missions", help="Directory for generated mission files (default: missions/)")
    parser.add_argument("--headed", action="store_true", help="Run browser with visible UI")
    parser.add_argument("--clear-memory", action="store_true", help="[Deprecated] Alias for --clear-all. Delete the entire SQLite memory database")
    parser.add_argument("--clear-checkpoints", action="store_true", help="Clear only LangGraph checkpoints (mission state). Preserves learned memory (pages, bugs, procedures)")
    parser.add_argument("--clear-learned", action="store_true", help="Clear only learned memory (semantic, episodic, procedural). Preserves checkpoints for resume")
    parser.add_argument("--clear-all", action="store_true", help="Delete the entire SQLite memory database (checkpoints + learned memory)")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum LangGraph execution steps per mission before resetting to homepage (default: 30)")
    parser.add_argument(
        "--provider", type=str, default=None, choices=["gemini", "claude"],
        help="LLM provider to use — overrides LLM_PROVIDER env var and config.yaml",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress real-time ReAct THOUGHT/ACTION/OBSERV console output (traces.log still captures everything)",
    )
    parser.add_argument(
        "--regression", action="store_true",
        help="Auto-generate regression missions from the bug catalog (no --missions needed)",
    )
    parser.add_argument(
        "--export-model", action="store_true",
        help="Export discovered application model from memory store as JSON",
    )
    args = parser.parse_args()

    # Load config.yaml early so its llm section can seed env var defaults.
    cfg = load_app_config()

    # Apply config.yaml llm values as env-var defaults (env vars & --provider win).
    _llm_defaults = {
        "LLM_PROVIDER": cfg.llm.provider,
        "CLAUDE_MODEL": cfg.llm.claude_model,
        "GEMINI_MODEL": cfg.llm.gemini_model,
        "CLAUDE_VISION_MODEL": cfg.llm.claude_vision_model,
        "GEMINI_VISION_MODEL": cfg.llm.gemini_vision_model,
    }
    for key, val in _llm_defaults.items():
        if val:
            os.environ.setdefault(key, val)

    # --provider overrides both env var and config.yaml.
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    try:
        probe_llm = make_llm(temperature=0)  # early credential probe — raises RuntimeError with clear message
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    console.model_info(get_active_provider(), get_model_name(probe_llm))

    if not args.missions and not args.pr_url and not args.regression and not args.export_model:
        parser.error("At least one of --missions, --pr-url, --regression, or --export-model is required")
    if not cfg.app.url:
        raise ValueError(
            "App URL is not configured. Set APP_URL in .env or app.url in config.yaml."
        )

    missions = []

    if args.missions:
        with open(args.missions, 'r', encoding="utf-8") as missions_file:
            config_data = yaml.safe_load(missions_file)
            missions.extend(config_data.get("missions", []))

    if args.pr_url:
        from agentic_explorer.pr_analyzer import (
            parse_pr_url, fetch_pr_data, generate_missions_from_pr,
        )
        owner, repo, pr_number = parse_pr_url(args.pr_url)
        console.step(f"Fetching PR #{pr_number} from {owner}/{repo}...")
        pr_data = await fetch_pr_data(owner, repo, pr_number, mcp_config_path=cfg.paths.mcp_servers)
        console.success(f"PR: {pr_data.title} ({len(pr_data.files_changed)} files changed)")
        console.step("Generating targeted test scenarios with LLM...")
        _pr_store = None
        if os.path.exists("agent_memory.sqlite"):
            from langgraph.store.sqlite import AsyncSqliteStore as _PRStore
            _pr_store_ctx = _PRStore.from_conn_string("agent_memory.sqlite")
            _pr_store = await _pr_store_ctx.__aenter__()
        try:
            generated = await generate_missions_from_pr(pr_data, cfg.app, store=_pr_store)
        finally:
            if _pr_store is not None:
                await _pr_store_ctx.__aexit__(None, None, None)
        pr_missions = generated.get("missions", [])

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"pr_{pr_number}.yaml")
        with open(output_path, 'w', encoding="utf-8") as f:
            yaml.dump(generated, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        console.success(f"Generated {len(pr_missions)} missions -> {output_path}")

        if args.execute:
            missions.extend(pr_missions)
        elif not missions:
            console.info("Use --execute to run generated missions, or --missions to run a file.")
            return

    _clear_all = args.clear_all or args.clear_memory
    if _clear_all:
        console.section("Clearing all memory")
        for mem_file in ["agent_memory.sqlite", "agent_memory.sqlite-wal", "agent_memory.sqlite-shm"]:
            if os.path.exists(mem_file):
                os.remove(mem_file)
                console.info(f"Deleted {mem_file}")
    elif args.clear_checkpoints or args.clear_learned:
        import sqlite3
        db_path = "agent_memory.sqlite"
        if os.path.exists(db_path):
            console.section("Selective memory clearing")
            conn = sqlite3.connect(db_path)
            try:
                if args.clear_checkpoints:
                    for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
                        try:
                            conn.execute(f"DELETE FROM {table}")
                            console.info(f"Cleared table: {table}")
                        except sqlite3.OperationalError:
                            pass
                    conn.commit()
                    console.success("Checkpoints cleared (learned memory preserved)")
                if args.clear_learned:
                    try:
                        conn.execute("DELETE FROM store")
                        conn.commit()
                        console.success("Learned memory cleared (checkpoints preserved)")
                    except sqlite3.OperationalError:
                        console.warn("No store table found")
            finally:
                conn.close()
        else:
            console.info("No memory database to clear")

    # Handle --export-model (standalone operation, no browser needed)
    if args.export_model:
        import json
        from langgraph.store.sqlite import AsyncSqliteStore as _ExportStore
        from agentic_explorer.memory import app_url_hash as _exp_hash, export_app_model

        if not os.path.exists("agent_memory.sqlite"):
            console.warn("No memory database found. Run missions first to build the app model.")
            return
        async with _ExportStore.from_conn_string("agent_memory.sqlite") as _store:
            _hash = _exp_hash(cfg.app.url)
            model = await export_app_model(_store, _hash)
            output_path = "app_model.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(model, f, indent=2, default=str)
            console.success(f"Application model exported to {output_path}")
            page_count = len(model.get("pages", []))
            bug_count = len(model.get("bugs", []))
            console.info(f"  Pages: {page_count}, Bugs: {bug_count}, Selectors: {len(model.get('selectors', []))}")
        if not missions and not args.regression:
            return

    # Handle --regression (generate missions from bug catalog)
    if args.regression:
        from langgraph.store.sqlite import AsyncSqliteStore as _RegStore
        from agentic_explorer.memory import app_url_hash as _reg_hash, generate_regression_missions

        if not os.path.exists("agent_memory.sqlite"):
            console.warn("No memory database found. Run missions first to build the bug catalog.")
            return
        async with _RegStore.from_conn_string("agent_memory.sqlite") as _store:
            _hash = _reg_hash(cfg.app.url)
            regression_missions = await generate_regression_missions(_store, _hash)
            if regression_missions:
                console.success(f"Generated {len(regression_missions)} regression missions from bug catalog")
                missions.extend(regression_missions)
            else:
                console.warn("No open bugs found in catalog. Nothing to regress.")

    if not missions:
        console.warn("No missions found. Exiting.")
        return

    mission_cooldown = int(os.getenv("MISSION_COOLDOWN", "5"))

    console.section("Setup")
    console.step("Loading MCP server tools...")
    doc_tools = await get_mcp_tools(cfg.paths.mcp_servers)
    skill_tools = [fetch_agent_skill, run_agent_skill_script]

    skills_root = cfg.paths.skills_root or os.getenv("AGENT_SKILLS_ROOT", "./agent-skills")
    if not os.path.isdir(skills_root):
        console.info(
            f"Skills directory '{skills_root}' not found — skills disabled. "
            "Set AGENT_SKILLS_ROOT or paths.skills_root in config.yaml."
        )

    console.step("Initializing browser, database, and memory store...")
    async_sqlite_saver = _get_async_sqlite_saver()
    from langgraph.store.sqlite import AsyncSqliteStore

    _emb_model = os.getenv("EMBEDDING_MODEL") or cfg.llm.embedding_model
    _emb_dims = int(os.getenv("EMBEDDING_DIMS", "0")) or cfg.llm.embedding_dims or 0
    _store_kwargs: dict = {}
    if _emb_model and _emb_dims:
        _store_kwargs["index"] = {"dims": _emb_dims, "embed": _emb_model}
        console.info(f"Embedding index: {_emb_model} ({_emb_dims}d)")

    async with (
        async_playwright() as playwright_instance,
        async_sqlite_saver.from_conn_string("agent_memory.sqlite") as memory_saver,
        AsyncSqliteStore.from_conn_string("agent_memory.sqlite", **_store_kwargs) as memory_store,
    ):
        browser = await playwright_instance.chromium.launch(headless=not args.headed, args=["--start-maximized"])

        if not os.path.exists("auth.json"):
            context = await browser.new_context(no_viewport=True, ignore_https_errors=True)
        else:
            context = await browser.new_context(storage_state="auth.json", no_viewport=True, ignore_https_errors=True)

        context.set_default_timeout(5000)
        context.set_default_navigation_timeout(15000)

        active_page = await context.new_page()

        # base_tools: MCP docs + Agent Skills only.
        # Raw PlayWrightBrowserToolkit tools are intentionally excluded — agents emit
        # JSON intents via execute_browser_command, not Playwright calls directly.
        base_tools = doc_tools + skill_tools

        console.step("Compiling LangGraph swarms...")
        standard_app = await build_graph(base_tools, active_page, memory_saver, cfg.app, max_steps=args.max_steps, quiet=args.quiet, store=memory_store)
        advanced_app = await build_advanced_graph(base_tools, active_page, memory_saver, cfg.app, max_steps=args.max_steps, quiet=args.quiet, store=memory_store)
        console.success("Ready")

        for mission in missions:
            thread_id = str(mission["thread_id"])
            prompt = mission["prompt"]

            is_advanced = any(kw in thread_id.lower() for kw in ADVANCED_KEYWORDS)
            mission_type = "ADVANCED" if is_advanced else "STANDARD"
            app = advanced_app if is_advanced else standard_app

            console.mission_start(thread_id, mission_type)

            os.makedirs(f"report_{thread_id}", exist_ok=True)
            with open(f"report_{thread_id}/traces.log", "w", encoding="utf-8") as f:
                f.write(f"=== TRACES: {thread_id} ===\n")

            run_config = {"configurable": {"thread_id": thread_id}}
            existing_state = await app.aget_state(run_config)

            initial_state = None
            if not existing_state.values:
                initial_state = {
                    "messages": [HumanMessage(content=prompt)],
                    "next_agent": "",
                    "step_count": 0,
                    "action_tape": [],
                    "bugs_found": [],
                    "explored_paths": [],
                }

            max_retries = 5
            base_delay = 2

            for attempt in range(max_retries):
                try:
                    seen_message_ids: set[str] = set()
                    trace_path = f"report_{thread_id}/traces.log"
                    current_node: str | None = None

                    async for mode, payload in app.astream(
                        initial_state,
                        config=run_config,
                        stream_mode=["updates", "messages"],
                    ):
                        if mode == "updates":
                            # payload: {node_name: state_update}
                            for node_name, state_update in payload.items():
                                if current_node is not None:
                                    console.state_update_end()
                                current_node = node_name
                                console.state_update(node_name)

                                if "messages" in state_update and state_update["messages"]:
                                    messages = state_update["messages"] if isinstance(state_update["messages"], list) else [state_update["messages"]]
                                    with open(trace_path, "a", encoding="utf-8") as trace_file:
                                        trace_file.write(f"\nSTATE UPDATE FROM: {node_name}\n")
                                        for msg in messages:
                                            _strip_extras(msg)
                                            msg_id = getattr(msg, "id", None)
                                            if msg_id and msg_id in seen_message_ids:
                                                continue
                                            if msg_id:
                                                seen_message_ids.add(msg_id)
                                            trace_file.write(msg.pretty_repr() + "\n")

                        elif mode == "messages":
                            msg, meta = payload
                            node_name = meta.get("langgraph_node") or current_node or "?"

                            # Skip mid-stream chunks — wait for the final assembled message.
                            if isinstance(msg, AIMessageChunk):
                                continue
                            # Skip system prompt messages — they are static boilerplate.
                            if isinstance(msg, SystemMessage):
                                continue

                            msg_id = getattr(msg, "id", None)
                            if msg_id and msg_id in seen_message_ids:
                                continue
                            if msg_id:
                                seen_message_ids.add(msg_id)

                            _strip_extras(msg)

                            # Console: ReAct lines (compact when --quiet).
                            _max = 120 if args.quiet else 500
                            if isinstance(msg, AIMessage):
                                text = _message_text(msg)
                                if text.strip():
                                    console.react_thought(node_name, text, max_chars=_max)
                            elif isinstance(msg, HumanMessage):
                                text = _message_text(msg)
                                if text.strip():
                                    console.info(f"HUMAN: {text[:120]}")

                            # Trace file: full pretty_repr for every message (regardless of --quiet).
                            with open(trace_path, "a", encoding="utf-8") as trace_file:
                                trace_file.write(f"\n[{node_name}] {msg.__class__.__name__}\n")
                                trace_file.write(msg.pretty_repr() + "\n")

                    if current_node is not None:
                        console.state_update_end()
                    break
                except Exception as e:
                    if _is_transient_error(e):
                        if attempt < max_retries - 1:
                            if _is_rate_limit(e):
                                delay = 30 + (attempt * 15) + random.randint(0, 10)
                                console.warn(f"Rate limit hit: {str(e)[:120]}")
                            else:
                                delay = base_delay * (2 ** attempt)
                            console.retry(attempt + 1, max_retries, delay)
                            await asyncio.sleep(delay)
                            initial_state = None  # resume from checkpoint
                        else:
                            console.fail(f"Failed after {max_retries} attempts.")
                            raise
                    else:
                        raise

            console.step(f"Generating report for {thread_id}...")
            final_state = await app.aget_state(run_config)
            mission_history = final_state.values.get("messages", [])
            bugs_found = final_state.values.get("bugs_found", [])

            transcript_lines = []
            for msg in mission_history:
                text_content = (
                    "".join([b.get("text", "") for b in msg.content if isinstance(b, dict) and "text" in b])
                    if isinstance(msg.content, list)
                    else str(msg.content)
                )
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    text_content += f" [Action: {', '.join(tc['name'] for tc in msg.tool_calls)}]"
                if text_content.strip():
                    transcript_lines.append(f"{msg.type.upper()}: {_clip_for_prompt(text_content, 2_000)}")

            clean_transcript = _bound_transcript_for_report("\n".join(transcript_lines))

            _active_provider = os.getenv("LLM_PROVIDER", "").lower()
            report_model = (
                os.getenv("CLAUDE_REPORT_MODEL") if _active_provider == "claude"
                else os.getenv("GEMINI_REPORT_MODEL")
            ) or None
            report_llm = make_llm(temperature=0, model_name=report_model)

            bugs_section = ""
            if bugs_found:
                bugs_section = (
                    f"\n\nBUGS CAPTURED ({len(bugs_found)}):\n"
                    + "\n".join(f"  - {b[:200]}" for b in bugs_found)
                )

            report_instruction = HumanMessage(content=(
                f"You are the Lead QA Engineer. Review the following agent test transcript for mission '{thread_id}'.\n\n"
                f"--- TEST TRANSCRIPT ---\n{clean_transcript}{bugs_section}\n-----------------------\n\n"
                "Write a concise report formatted in Markdown. Include ONLY these sections:\n"
                "- **Mission ID & Objective**\n"
                "- **Actions Taken** (Brief summary)\n"
                "- **Issues Found** (Any UI errors, visual anomalies, or tool failures)\n"
                "- **Final Status** (PASS or FAIL based on whether the objective was achieved)\n\n"
                "Output ONLY plain Markdown text."
            ))

            for attempt in range(max_retries):
                try:
                    report_response = await report_llm.ainvoke([report_instruction])
                    break
                except Exception as e:
                    if _is_transient_error(e):
                        if attempt < max_retries - 1:
                            if _is_rate_limit(e):
                                delay = 30 + (attempt * 15) + random.randint(0, 10)
                                console.warn(f"Rate limit hit: {str(e)[:120]}")
                            else:
                                delay = base_delay * (2 ** attempt)
                            console.retry(attempt + 1, max_retries, delay)
                            await asyncio.sleep(delay)
                        else:
                            console.fail(f"Report generation failed after {max_retries} attempts.")
                            raise
                    else:
                        raise

            clean_report_text = report_response.content
            if isinstance(clean_report_text, list):
                clean_report_text = "".join([
                    block.get("text", "") for block in clean_report_text if isinstance(block, dict)
                ])

            with open(f"report_{thread_id}/test_report.md", "w", encoding="utf-8") as report_file:
                report_file.write(f"\n{clean_report_text}\n\n---\n")

            tape = get_action_tape(thread_id)

            # Write episodic memory (session summary + bug catalog)
            try:
                from agentic_explorer.memory import app_url_hash, write_episode_memory
                url_hash = app_url_hash(cfg.app.url)
                await write_episode_memory(
                    memory_store, url_hash, thread_id,
                    prompt, tape, bugs_found,
                    final_state.values.get("explored_paths", []),
                )
            except Exception:
                pass

            console.mission_done(thread_id, len(tape), len(bugs_found))
            with open(f"report_{thread_id}/test_report.md", "a", encoding="utf-8") as report_file:
                report_file.write(
                    f"\n## Action Tape\n\n"
                    f"- **Recorded steps:** {len(tape)}\n"
                    f"- **Bugs captured:** {len(bugs_found)}\n"
                    f"- **Tape log:** `action_tape.jsonl`\n"
                    f"- **Reproductions:** any `reproduction_*.spec.ts` in this folder can be run with `npx playwright test`.\n"
                )

            if mission_cooldown > 0 and mission != missions[-1]:
                console.info(f"Cooling down {mission_cooldown}s before next mission...")
                await asyncio.sleep(mission_cooldown)

        # Post-batch: update procedural memory via LLM reflection
        try:
            from agentic_explorer.memory import app_url_hash as _url_hash_fn, update_procedural_memory
            _batch_url_hash = _url_hash_fn(cfg.app.url)
            _reflection_llm = make_llm(temperature=0)
            await update_procedural_memory(memory_store, _batch_url_hash, _reflection_llm)
        except Exception:
            pass

        console.final_summary(len(missions))
        await browser.close()

def main():
    asyncio.run(run_missions())

if __name__ == "__main__":
    main()
