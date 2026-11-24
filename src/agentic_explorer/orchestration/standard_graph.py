"""Deprecated: use ``orchestration.graph.build_graph(..., tier="standard")``.

Kept as a thin wrapper so existing imports keep working.
"""

import warnings

from playwright.async_api import Page

from agentic_explorer.config import AppMeta
from agentic_explorer.orchestration.graph import build_graph as _build_graph


async def build_graph(base_tools: list, active_page: Page, checkpointer, app: AppMeta, max_steps: int = 30, quiet: bool = False, store=None):
    """Build the standard QA swarm of three behavioral persona agents."""
    warnings.warn(
        "standard_graph.build_graph is deprecated; use orchestration.graph.build_graph(tier='standard')",
        DeprecationWarning,
        stacklevel=2,
    )
    return await _build_graph(
        base_tools, active_page, checkpointer, app, tier="standard", max_steps=max_steps, quiet=quiet, store=store,
    )
