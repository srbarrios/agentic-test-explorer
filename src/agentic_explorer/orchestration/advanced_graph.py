"""Deprecated: use ``orchestration.graph.build_graph(..., tier="advanced")``.

The advanced graph hosts high-intensity behavioral personas and the open-ended
``explorer_agent``. Kept as a thin wrapper so existing imports keep working.
"""

import warnings

from playwright.async_api import Page

from agentic_explorer.config import AppMeta
from agentic_explorer.orchestration.graph import build_graph as _build_graph


async def build_advanced_graph(base_tools: list, active_page: Page, checkpointer, app: AppMeta, max_steps: int = 30, quiet: bool = False, store=None):
    """Build the advanced persona and autonomous exploration graph."""
    warnings.warn(
        "advanced_graph.build_advanced_graph is deprecated; use orchestration.graph.build_graph(tier='advanced')",
        DeprecationWarning,
        stacklevel=2,
    )
    return await _build_graph(
        base_tools, active_page, checkpointer, app, tier="advanced", max_steps=max_steps, quiet=quiet, store=store,
    )
