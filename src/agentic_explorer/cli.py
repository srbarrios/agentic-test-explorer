"""Command-line interface of ``agent-explorer``: flags, validation and a typed result."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class CliArgs:
    missions: Optional[str]
    pr_url: Optional[str]
    execute: bool
    output_dir: str
    headed: bool
    clear_memory: bool  # deprecated alias of clear_all
    clear_checkpoints: bool
    clear_learned: bool
    clear_all: bool
    max_steps: int
    provider: Optional[str]
    quiet: bool
    regression: bool
    export_model: bool
    visual: bool
    app_revision: Optional[str] = None


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--visual", action="store_true",
        help="Launch a Streamlit dashboard for real-time visual monitoring",
    )
    parser.add_argument(
        "--app-revision", type=str, default=None,
        help="Revision of the app under test (commit, version, build id), recorded in the tape header "
             "and in generated specs. Overrides APP_REVISION and app.revision in config.yaml",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> CliArgs:
    """Parse and validate ``argv`` (defaults to ``sys.argv``); exits with status 2 on errors."""
    parser = build_parser()
    args = CliArgs(**vars(parser.parse_args(argv)))

    if args.visual and importlib.util.find_spec("streamlit") is None:
        parser.error("--visual requires Streamlit. Install with: pip install agentic-test-explorer[visual]")
    if not (args.missions or args.pr_url or args.regression or args.export_model):
        parser.error("At least one of --missions, --pr-url, --regression, or --export-model is required")
    return args
