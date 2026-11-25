import asyncio
import os
import warnings
from typing import Any, Optional

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl

# Suppress library warnings before any imports that trigger them.
warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r".*allowed_objects.*",
)

from agentic_explorer.utils import console  # noqa: E402

from playwright.async_api import async_playwright  # noqa: E402

from agentic_explorer.cli import parse_args  # noqa: E402
from agentic_explorer.config import AppConfig, load_app_config, load_environment  # noqa: E402
from agentic_explorer.memory_admin import DEFAULT_DB  # noqa: E402
from agentic_explorer.memory_health import guarded  # noqa: E402
from agentic_explorer.orchestration.graph import build_graph  # noqa: E402
from agentic_explorer.orchestration.personas import resolve_tier  # noqa: E402
from agentic_explorer.runtime.drain import install_drain_handlers  # noqa: E402
from agentic_explorer.runtime.missions import resolve_missions  # noqa: E402
from agentic_explorer.runtime.runner import run_missions_batch  # noqa: E402
from agentic_explorer.tools.browser import session  # noqa: E402
from agentic_explorer.tools.browser.provenance import resolve_app_revision  # noqa: E402
from agentic_explorer.tools.common.custom_tools import (  # noqa: E402
    fetch_agent_skill,
    get_mcp_tools,
    run_agent_skill_script,
)
from agentic_explorer.utils.llm import get_active_provider, get_model_name, make_llm  # noqa: E402

load_environment()

def _get_async_sqlite_saver() -> Any:
    """Import AsyncSqliteSaver while suppressing a known upstream pending warning."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver


def _configure_llm(cfg: AppConfig, provider: Optional[str]):
    """Seed LLM env defaults from config.yaml, apply ``--provider`` and probe credentials."""
    # config.yaml llm values are only defaults: real env vars win.
    llm_defaults = {
        "LLM_PROVIDER": cfg.llm.provider,
        "CLAUDE_MODEL": cfg.llm.claude_model,
        "GEMINI_MODEL": cfg.llm.gemini_model,
        "CLAUDE_VISION_MODEL": cfg.llm.claude_vision_model,
        "GEMINI_VISION_MODEL": cfg.llm.gemini_vision_model,
    }
    for key, val in llm_defaults.items():
        if val:
            os.environ.setdefault(key, val)

    # --provider overrides both env var and config.yaml.
    if provider:
        os.environ["LLM_PROVIDER"] = provider

    try:
        probe_llm = make_llm(temperature=0)  # early credential probe — raises RuntimeError with clear message
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    console.model_info(get_active_provider(), get_model_name(probe_llm))
    return probe_llm


async def run_missions():
    """Run the missions; with ``--replay``, replay a recorded tape instead and return its exit code."""
    cli = parse_args()
    if cli.replay:
        # Replaying a tape needs a browser, not a model: no credentials, no cost.
        from agentic_explorer.runtime.replay import run_replay_command

        return await run_replay_command(cli.replay, headed=cli.headed)
    cfg = load_app_config()
    probe_llm = _configure_llm(cfg, cli.provider)

    if cli.visual:
        from agentic_explorer.ui.launcher import start_dashboard

        start_dashboard(
            app_url=cfg.app.url, provider=get_active_provider(), model_name=get_model_name(probe_llm),
        )

    if not cfg.app.url:
        raise ValueError(
            "App URL is not configured. Set APP_URL in .env or app.url in config.yaml."
        )

    missions = await resolve_missions(cli, cfg)
    if missions is None:
        return

    # Decide every mission's graph up front so an invalid `graph:` fails before any browser starts.
    tiers = [resolve_tier(mission) for mission in missions]

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

    emb_model = os.getenv("EMBEDDING_MODEL") or cfg.llm.embedding_model
    emb_dims = int(os.getenv("EMBEDDING_DIMS", "0")) or cfg.llm.embedding_dims or 0
    store_kwargs: dict = {}
    if emb_model and emb_dims:
        store_kwargs["index"] = {"dims": emb_dims, "embed": emb_model}
        console.info(f"Embedding index: {emb_model} ({emb_dims}d)")

    async with (
        async_playwright() as playwright_instance,
        async_sqlite_saver.from_conn_string(DEFAULT_DB) as memory_saver,
        AsyncSqliteStore.from_conn_string(DEFAULT_DB, **store_kwargs) as memory_store,
    ):
        browser = await playwright_instance.chromium.launch(headless=not cli.headed, args=["--start-maximized"])

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
        graphs = {
            tier: await build_graph(
                base_tools, active_page, memory_saver, cfg.app,
                tier=tier, max_steps=cli.max_steps, quiet=cli.quiet, store=memory_store,
                runtime=cfg.runtime,
            )
            for tier in ("standard", "advanced")
        }
        console.success("Ready")

        # Ctrl-C / SIGTERM stop the batch at a safe point (GraphDrained); running the
        # same command again resumes the interrupted mission from its checkpoint.
        control = RunControl()
        uninstall_drain_handlers = install_drain_handlers(control)
        # An expired auth.json stops the run at a safe point instead of testing a login page.
        session.arm(expected=os.path.exists("auth.json"), control=control)
        try:
            await run_missions_batch(
                graphs, missions, tiers,
                app_url=cfg.app.url, store=memory_store, control=control,
                quiet=cli.quiet, visual=cli.visual, cooldown_s=mission_cooldown,
                provenance={
                    "app_name": cfg.app.name,
                    "app_url": cfg.app.url,
                    "app_revision": resolve_app_revision(cli.app_revision, cfg.app.revision),
                    "seed": cfg.app.seed,
                },
            )
        finally:
            uninstall_drain_handlers()

        # Post-batch: update procedural memory via LLM reflection. Best effort, but a
        # failure is logged and counted instead of vanishing.
        from agentic_explorer.memory import app_url_hash, update_procedural_memory
        await guarded(
            "procedural reflection",
            lambda: update_procedural_memory(memory_store, app_url_hash(cfg.app.url), make_llm(temperature=0)),
        )

        console.final_summary(len(missions))

        # Mark visual mode as completed before closing
        if cli.visual:
            from agentic_explorer.ui import state_emitter
            state_emitter.mark_completed(total_missions=len(missions))
            # Give dashboard time to show completion before we exit
            await asyncio.sleep(2)

        await browser.close()


def main():
    try:
        exit_code = asyncio.run(run_missions())
        if exit_code:  # only --replay returns one
            raise SystemExit(exit_code)
    except GraphDrained as drained:
        if drained.reason == session.DRAIN_REASON:
            console.warn(
                "The saved session (auth.json) has expired. Run `agent-auth` to sign in again, then run "
                "the same command to resume the interrupted mission (its steps are kept)."
            )
        else:
            console.warn(
                f"Stopped ({drained.reason}) at a safe point. "
                "Run the same command again to resume the interrupted mission."
            )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
