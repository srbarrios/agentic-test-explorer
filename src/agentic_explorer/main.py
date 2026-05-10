import asyncio
import os
import argparse
import yaml
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.async_api import async_playwright

from langchain_core.globals import set_verbose

from agentic_explorer.tools.browser.engine import get_action_tape
from agentic_explorer.config import load_app_config

set_verbose(True)

from agentic_explorer.tools.common.custom_tools import (
    get_mcp_tools,
    fetch_agent_skill,
    run_agent_skill_script,
)
from agentic_explorer.orchestration.standard_graph import build_graph
from agentic_explorer.orchestration.advanced_graph import build_advanced_graph

load_dotenv()

# Mission-type detection: thread_ids matching these substrings are routed to the
# advanced (autonomous explorer) graph instead of the standard 5-agent UI swarm.
ADVANCED_KEYWORDS = ("explorer", "chaos", "autonomous")


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(x in msg for x in ("503", "UNAVAILABLE", "429", "RATE_LIMIT", "RESOURCE_EXHAUSTED", "QUOTA"))


async def run_missions():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Please set your GOOGLE_API_KEY environment variable.")

    parser = argparse.ArgumentParser(description="Run Agentic Exploratory Tests")
    parser.add_argument("--missions", type=str, default=None, help="Path to the YAML missions file")
    parser.add_argument("--pr-url", type=str, default=None, help="GitHub PR URL to analyze and generate test missions from")
    parser.add_argument("--execute", action="store_true", help="Execute generated PR missions immediately (default: generate only)")
    parser.add_argument("--output-dir", type=str, default="missions", help="Directory for generated mission files (default: missions/)")
    parser.add_argument("--headed", action="store_true", help="Run browser with visible UI")
    parser.add_argument("--clear-memory", action="store_true", help="Delete the previous SQLite memory database")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum LangGraph execution steps per mission before resetting to homepage (default: 30)")
    args = parser.parse_args()

    if not args.missions and not args.pr_url:
        parser.error("At least one of --missions or --pr-url is required")

    cfg = load_app_config()
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
        print(f"\nFetching PR #{pr_number} from {owner}/{repo}...")
        pr_data = await fetch_pr_data(owner, repo, pr_number, mcp_config_path=cfg.paths.mcp_servers)
        print(f"PR: {pr_data.title} ({len(pr_data.files_changed)} files changed)")
        print("Generating targeted test scenarios with LLM...")
        generated = await generate_missions_from_pr(pr_data, cfg.app)
        pr_missions = generated.get("missions", [])

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"pr_{pr_number}.yaml")
        with open(output_path, 'w', encoding="utf-8") as f:
            yaml.dump(generated, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"Generated {len(pr_missions)} missions -> {output_path}")

        if args.execute:
            missions.extend(pr_missions)
        elif not missions:
            print("\nUse --execute to run generated missions, or --missions to run a file.")
            return

    if not missions:
        print("No missions found. Exiting.")
        return

    if args.clear_memory:
        print("\nCleaning up previous memory files...")
        for mem_file in ["agent_memory.sqlite", "agent_memory.sqlite-wal", "agent_memory.sqlite-shm"]:
            if os.path.exists(mem_file):
                os.remove(mem_file)
                print(f"  - Deleted {mem_file}")

    print("Loading MCP server tools (if configured)...")
    doc_tools = await get_mcp_tools(cfg.paths.mcp_servers)
    skill_tools = [fetch_agent_skill, run_agent_skill_script]

    skills_root = cfg.paths.skills_root or os.getenv("AGENT_SKILLS_ROOT", "./agent-skills")
    if not os.path.isdir(skills_root):
        print(
            f"  Skills directory '{skills_root}' not found. "
            "Set AGENT_SKILLS_ROOT (or paths.skills_root in config.yaml) to a directory "
            "containing skills following the https://agentskills.io/specification layout."
        )

    print("Initializing Authenticated Browser and Persistent Database...")
    async with async_playwright() as playwright_instance, AsyncSqliteSaver.from_conn_string("agent_memory.sqlite") as memory_saver:
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

        print("Compiling LangGraph Swarms...")
        standard_app = build_graph(base_tools, active_page, memory_saver, cfg.app, max_steps=args.max_steps)
        advanced_app = build_advanced_graph(base_tools, active_page, memory_saver, cfg.app, max_steps=args.max_steps)

        for mission in missions:
            thread_id = str(mission["thread_id"])
            prompt = mission["prompt"]

            is_advanced = any(kw in thread_id.lower() for kw in ADVANCED_KEYWORDS)
            mission_type = "ADVANCED" if is_advanced else "STANDARD"
            app = advanced_app if is_advanced else standard_app

            print(f"\n{'='*60}\nSTARTING MISSION [{mission_type}]: {thread_id}\n{'='*60}")

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
                    async for output in app.astream(initial_state, config=run_config, stream_mode="updates"):
                        for node_name, state_update in output.items():
                            header = f"\n{'='*40}\nSTATE UPDATE FROM: {node_name}\n{'='*40}\n"
                            print(header)

                            if "messages" in state_update and state_update["messages"]:
                                messages = state_update["messages"] if isinstance(state_update["messages"], list) else [state_update["messages"]]

                                with open(f"report_{thread_id}/traces.log", "a", encoding="utf-8") as trace_file:
                                    trace_file.write(header)
                                    for msg in messages:
                                        if isinstance(msg.content, list):
                                            for block in msg.content:
                                                if isinstance(block, dict) and "extras" in block:
                                                    del block["extras"]
                                        msg.pretty_print()
                                        trace_file.write(msg.pretty_repr() + "\n")
                    break
                except Exception as e:
                    if _is_transient_error(e):
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"\nTransient error (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                            await asyncio.sleep(delay)
                            initial_state = None  # resume from checkpoint
                        else:
                            print(f"\nFailed after {max_retries} attempts.")
                            raise
                    else:
                        raise

            # Generate report
            print(f"\nGenerating Mission Report for {thread_id}...")
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
                    transcript_lines.append(f"{msg.type.upper()}: {text_content.strip()}")

            clean_transcript = "\n".join(transcript_lines)

            report_model = os.getenv("GEMINI_REPORT_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
            report_llm = ChatGoogleGenerativeAI(model=report_model, temperature=0)

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
                            delay = base_delay * (2 ** attempt)
                            print(f"\nReport generation transient error. Retrying in {delay}s...")
                            await asyncio.sleep(delay)
                        else:
                            print(f"\nReport generation failed after {max_retries} attempts.")
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
            print(f"Action Tape: {len(tape)} deterministic steps recorded (see report_{thread_id}/action_tape.jsonl)")
            with open(f"report_{thread_id}/test_report.md", "a", encoding="utf-8") as report_file:
                report_file.write(
                    f"\n## Action Tape\n\n"
                    f"- **Recorded steps:** {len(tape)}\n"
                    f"- **Bugs captured:** {len(bugs_found)}\n"
                    f"- **Tape log:** `action_tape.jsonl`\n"
                    f"- **Reproductions:** any `reproduction_*.spec.ts` in this folder can be run with `npx playwright test`.\n"
                )

        print("\nAll missions completed!")
        await browser.close()

def main():
    asyncio.run(run_missions())

if __name__ == "__main__":
    main()
