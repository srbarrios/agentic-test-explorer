"""langmem must stay replaceable.

Its last release is from Oct 2025 and importing it pulls ``trustcall``, which uses
``langgraph.constants.Send`` (deprecated in LangGraph 1.0, removed in 2.0). Only
``memory_backends`` may import it, so swapping it out later is a one-file change.
"""

import subprocess
import sys
import textwrap

from langgraph.store.memory import InMemoryStore

from agentic_explorer import memory_backends
from agentic_explorer.memory import get_memory_management_tool, update_procedural_memory, write_session_summary
from agentic_explorer.memory_backends import LangmemOptimizer


def _python(code: str) -> str:
    """Run ``code`` in a fresh interpreter (``sys.modules`` is clean there) and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    return result.stdout.strip().splitlines()[-1]


def test_importing_the_memory_module_does_not_load_langmem():
    assert _python("""
        import sys
        import agentic_explorer.memory
        print('langmem' in sys.modules)
    """) == "False"


def test_updating_procedural_memory_with_an_injected_optimizer_does_not_load_langmem():
    assert _python("""
        import asyncio, sys
        from langgraph.store.memory import InMemoryStore
        from agentic_explorer.memory import update_procedural_memory, write_session_summary

        class Optimizer:
            async def optimize(self, prompt, trajectory, *, context):
                return "optimized"

        async def main():
            store = InMemoryStore()
            await write_session_summary(store, "h", "test_new_user_01", "prompt", [], [], [])
            await update_procedural_memory(store, "h", None, optimizer=Optimizer())

        asyncio.run(main())
        print('langmem' in sys.modules)
    """) == "False"


async def test_the_default_optimizer_is_the_langmem_adapter_built_from_the_given_llm(monkeypatch):
    built = []

    class Spy:
        def __init__(self, llm):
            built.append(llm)

        async def optimize(self, prompt, trajectory, *, context):
            return "ok"

    monkeypatch.setattr(memory_backends, "LangmemOptimizer", Spy)
    store = InMemoryStore()
    await write_session_summary(store, "h", "test_new_user_01", "prompt", [], [], [])

    await update_procedural_memory(store, "h", "THE-LLM")

    assert built == ["THE-LLM"]


async def test_the_adapter_passes_the_trajectory_to_langmem_and_returns_text(monkeypatch):
    import langmem

    seen = []

    class FakeLangmemOptimizer:
        async def ainvoke(self, payload):
            seen.append(payload)
            return "new prompt"

    def fake_create(llm, kind):
        seen.append(("created", llm, kind))
        return FakeLangmemOptimizer()

    monkeypatch.setattr(langmem, "create_prompt_optimizer", fake_create)
    trajectory = [{"role": "user", "content": "Mission: x"}]

    result = await LangmemOptimizer("an-llm").optimize("base prompt", trajectory, context="batch review")

    assert result == "new prompt"
    assert seen[0] == ("created", "an-llm", "prompt_memory")
    assert seen[1] == {"trajectories": [(trajectory, {"context": "batch review"})], "prompt": "base prompt"}


async def test_the_adapter_turns_a_non_string_result_into_text(monkeypatch):
    import langmem

    class Result:
        def __str__(self):
            return "structured result"

    class FakeLangmemOptimizer:
        async def ainvoke(self, payload):
            return Result()

    monkeypatch.setattr(langmem, "create_prompt_optimizer", lambda llm, kind: FakeLangmemOptimizer())

    assert await LangmemOptimizer("llm").optimize("p", [], context="routing review") == "structured result"


def test_the_observation_tool_comes_from_the_adapter_and_keeps_its_name():
    tool = get_memory_management_tool(InMemoryStore(), "h")

    assert tool.name == "record_observation"
