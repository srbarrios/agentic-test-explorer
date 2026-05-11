"""Clean, structured console output for agent-explorer.

Provides themed print helpers that replace raw print() calls across the
codebase.  All library warnings (LangChain deprecation, Google Cloud SDK,
etc.) are suppressed at import time so they never leak to the user's terminal.
"""

from __future__ import annotations

import os
import sys
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# ── Suppress noisy library warnings ──────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.auth")
warnings.filterwarnings("ignore", message=r".*LangChain.*")
warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r".*allowed_objects.*",
)

# Also silence langchain verbose chatter
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


# ── ANSI helpers ─────────────────────────────────────────────────────────
_NO_COLOR = os.environ.get("NO_COLOR") or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty()

def _ansi(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _dim(text: str) -> str:
    return _ansi("2", text)

def _bold(text: str) -> str:
    return _ansi("1", text)

def _green(text: str) -> str:
    return _ansi("32", text)

def _yellow(text: str) -> str:
    return _ansi("33", text)

def _red(text: str) -> str:
    return _ansi("31", text)

def _cyan(text: str) -> str:
    return _ansi("36", text)

def _magenta(text: str) -> str:
    return _ansi("35", text)


# ── Width ────────────────────────────────────────────────────────────────
_WIDTH = 60


# ── Public API ───────────────────────────────────────────────────────────

def banner(title: str, subtitle: str = "") -> None:
    line = _bold(_cyan("─" * _WIDTH))
    print(f"\n{line}")
    print(_bold(_cyan(f"  {title}")))
    if subtitle:
        print(_dim(f"  {subtitle}"))
    print(line)


def section(label: str) -> None:
    print(f"\n{_bold(_magenta('▸'))} {_bold(label)}")


def step(msg: str) -> None:
    print(f"  {_dim('•')} {msg}")


def success(msg: str) -> None:
    print(f"  {_green('✓')} {msg}")


def warn(msg: str) -> None:
    print(f"  {_yellow('!')} {_yellow(msg)}")


def fail(msg: str) -> None:
    print(f"  {_red('✗')} {_red(msg)}")


def info(msg: str) -> None:
    print(f"  {_dim(msg)}")


def state_update(node_name: str) -> None:
    print(f"\n  {_dim('┌')} {_bold(node_name)}")


def state_update_end() -> None:
    print(f"  {_dim('└──')}")


def mission_start(thread_id: str, mission_type: str) -> None:
    badge = _green(f"[{mission_type}]") if mission_type == "STANDARD" else _magenta(f"[{mission_type}]")
    banner(f"MISSION {badge} {thread_id}")


def mission_done(thread_id: str, tape_len: int, bugs_len: int) -> None:
    stats = f"actions={tape_len}  bugs={bugs_len}"
    print(f"\n  {_green('✓')} Mission complete: {_bold(thread_id)}  {_dim(stats)}")


def retry(attempt: int, max_retries: int, delay: int) -> None:
    print(f"  {_yellow('↻')} Transient error (attempt {attempt}/{max_retries}). Retrying in {delay}s...")


def final_summary(total_missions: int) -> None:
    line = _bold(_green("─" * _WIDTH))
    print(f"\n{line}")
    print(_bold(_green(f"  All {total_missions} mission(s) completed")))
    print(line)
