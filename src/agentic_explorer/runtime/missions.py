"""Where missions come from: a YAML file, a GitHub PR, or the known-bug catalog.

Also hosts ``--export-model``, which reads the same memory store.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import yaml
from langgraph.store.sqlite import AsyncSqliteStore

from agentic_explorer import pr_analyzer
from agentic_explorer.cli import CliArgs
from agentic_explorer.config import AppConfig, AppMeta
from agentic_explorer.memory import app_url_hash, export_app_model, generate_regression_missions
from agentic_explorer.memory_admin import DEFAULT_DB, clear_memory
from agentic_explorer.utils import console

Mission = Dict[str, Any]


def load_missions_from_file(path: str) -> List[Mission]:
    """Read the ``missions:`` list of a YAML file (empty if absent or the file is blank)."""
    with open(path, "r", encoding="utf-8") as missions_file:
        data = yaml.safe_load(missions_file) or {}
    return data.get("missions", [])


async def missions_from_pr(
    pr_url: str,
    *,
    app: AppMeta,
    mcp_config_path: Optional[str],
    output_dir: str,
    db_path: str = DEFAULT_DB,
) -> List[Mission]:
    """Generate missions from a PR diff, save them to ``<output_dir>/pr_<n>.yaml`` and return them.

    Known bugs are injected into the generation prompt when a memory database exists.
    """
    owner, repo, pr_number = pr_analyzer.parse_pr_url(pr_url)
    console.step(f"Fetching PR #{pr_number} from {owner}/{repo}...")
    pr_data = await pr_analyzer.fetch_pr_data(owner, repo, pr_number, mcp_config_path=mcp_config_path)
    console.success(f"PR: {pr_data.title} ({len(pr_data.files_changed)} files changed)")
    console.step("Generating targeted test scenarios with LLM...")

    if os.path.exists(db_path):
        async with AsyncSqliteStore.from_conn_string(db_path) as store:
            generated = await pr_analyzer.generate_missions_from_pr(pr_data, app, store=store)
    else:
        generated = await pr_analyzer.generate_missions_from_pr(pr_data, app, store=None)
    missions = generated.get("missions", [])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pr_{pr_number}.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(generated, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    console.success(f"Generated {len(missions)} missions -> {output_path}")
    return missions


async def missions_from_regression(app_url: str, db_path: str = DEFAULT_DB) -> Optional[List[Mission]]:
    """Missions targeting known open bugs. ``None`` when there is no memory database yet."""
    if not os.path.exists(db_path):
        console.warn("No memory database found. Run missions first to build the bug catalog.")
        return None
    async with AsyncSqliteStore.from_conn_string(db_path) as store:
        missions = await generate_regression_missions(store, app_url_hash(app_url))
    if missions:
        console.success(f"Generated {len(missions)} regression missions from bug catalog")
    else:
        console.warn("No open bugs found in catalog. Nothing to regress.")
    return missions


async def export_model(app_url: str, db_path: str = DEFAULT_DB, output_path: str = "app_model.json") -> bool:
    """Write the discovered application model as JSON. ``False`` when there is no memory database."""
    if not os.path.exists(db_path):
        console.warn("No memory database found. Run missions first to build the app model.")
        return False
    async with AsyncSqliteStore.from_conn_string(db_path) as store:
        model = await export_app_model(store, app_url_hash(app_url))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, default=str)
    console.success(f"Application model exported to {output_path}")
    console.info(
        f"  Pages: {len(model.get('pages', []))}, Bugs: {len(model.get('bugs', []))}, "
        f"Selectors: {len(model.get('selectors', []))}"
    )
    return True


async def resolve_missions(cli: CliArgs, cfg: AppConfig) -> Optional[List[Mission]]:
    """Collect the missions to run from every requested source.

    Also applies ``--clear-*`` and ``--export-model`` at their original point in
    the sequence (file, PR, clear, export, regression). Returns ``None`` when the
    run should end without executing anything.
    """
    missions = load_missions_from_file(cli.missions) if cli.missions else []

    if cli.pr_url:
        pr_missions = await missions_from_pr(
            cli.pr_url, app=cfg.app, mcp_config_path=cfg.paths.mcp_servers, output_dir=cli.output_dir,
        )
        if cli.execute:
            missions.extend(pr_missions)
        elif not missions:
            console.info("Use --execute to run generated missions, or --missions to run a file.")
            return None

    clear_memory(
        everything=cli.clear_all or cli.clear_memory,
        checkpoints=cli.clear_checkpoints,
        learned=cli.clear_learned,
    )

    if cli.export_model:
        if not await export_model(cfg.app.url):
            return None
        if not missions and not cli.regression:
            return None

    if cli.regression:
        regression = await missions_from_regression(cfg.app.url)
        if regression is None:
            return None
        missions.extend(regression)

    if not missions:
        console.warn("No missions found. Exiting.")
        return None
    return missions
