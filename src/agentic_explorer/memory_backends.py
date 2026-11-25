"""Adapters for the ``langmem`` package: the only module allowed to import it.

``langmem`` is useful (prompt optimization, a ready-made "manage memory" tool) but
its last release is from Oct 2025 and importing it pulls ``trustcall``, which uses
``langgraph.constants.Send`` (deprecated in LangGraph 1.0, removed in 2.0). Keeping
it behind these two small adapters means replacing it later is a change to this file,
not to the memory logic. It is imported lazily, so ``import agentic_explorer.memory``
does not load it.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence, Tuple


class PromptOptimizer(Protocol):
    """Improves a prompt from the trajectory of past missions."""

    async def optimize(
        self, prompt: str, trajectory: Sequence[Dict[str, str]], *, context: str,
    ) -> str:
        """Return an improved version of ``prompt``.

        ``trajectory`` is a list of ``{"role", "content"}`` messages describing past
        missions; ``context`` says what the prompt is for (``"batch review"`` for an
        agent prompt, ``"routing review"`` for the supervisor's routing rules).
        """


class LangmemOptimizer:
    """``PromptOptimizer`` backed by langmem's ``create_prompt_optimizer``."""

    def __init__(self, llm: Any) -> None:
        import langmem

        self._optimizer = langmem.create_prompt_optimizer(llm, kind="prompt_memory")

    async def optimize(
        self, prompt: str, trajectory: Sequence[Dict[str, str]], *, context: str,
    ) -> str:
        result = await self._optimizer.ainvoke({
            "trajectories": [(list(trajectory), {"context": context})],
            "prompt": prompt,
        })
        return result if isinstance(result, str) else str(result)


def make_observation_tool(
    store: Any, *, namespace: Tuple[str, ...], instructions: str, name: str = "record_observation",
) -> Any:
    """A LangChain tool that lets an agent record observations in ``namespace`` of ``store``."""
    import langmem

    return langmem.create_manage_memory_tool(
        namespace=namespace, instructions=instructions, store=store, name=name,
    )
