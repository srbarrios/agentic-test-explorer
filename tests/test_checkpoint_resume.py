"""Existing ``agent_memory.sqlite`` files must keep resuming across upgrades.

``fixtures/checkpoint_langgraph_1_1_10.sqlite`` was written by langgraph 1.1.10
(see ``checkpoint_fixture.py``). These tests read it with whatever version is
installed now.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from checkpoint_fixture import BUG, CONFIG, build_graph

TESTS_DIR = Path(__file__).parent
FIXTURE = TESTS_DIR / "fixtures" / "checkpoint_langgraph_1_1_10.sqlite"


@pytest.fixture
def old_db(tmp_path) -> str:
    """A private copy, so resuming never mutates the committed fixture."""
    dest = tmp_path / "agent_memory.sqlite"
    shutil.copy(FIXTURE, dest)
    return str(dest)


async def test_old_checkpoint_loads_on_new_langgraph(old_db):
    async with AsyncSqliteSaver.from_conn_string(old_db) as saver:
        snapshot = await build_graph(saver).aget_state(CONFIG)

    values = snapshot.values
    assert snapshot.next == ()  # the run had finished
    assert values["step_count"] == 2
    assert [m.content for m in values["messages"]] == [
        "explore the app", "visited /login", "visited /cart",
    ]
    assert isinstance(values["messages"][0], HumanMessage)
    assert isinstance(values["messages"][1], AIMessage)
    assert values["bugs_found"] == [BUG]
    assert values["explored_paths"] == ["https://app.test/login"]
    assert values["action_tape"][0]["action"] == "navigate"


async def test_old_checkpoint_resumes_and_keeps_accumulated_state(old_db):
    async with AsyncSqliteSaver.from_conn_string(old_db) as saver:
        graph = build_graph(saver)
        out = await graph.ainvoke({"messages": [HumanMessage(content="again")]}, CONFIG)

    assert out["step_count"] == 4  # continued from 2, did not restart at 0
    assert out["bugs_found"] == [BUG, BUG]  # operator.add accumulates across runs
    assert out["messages"][0].content == "explore the app"  # mission prompt kept first


def _run_strict(code: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true", "PYTHONPATH": str(TESTS_DIR)}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True, text=True, env=env, timeout=120,
    )


def test_agent_state_roundtrips_with_strict_msgpack(tmp_path):
    """Write a fresh checkpoint and read it back with strict msgpack on."""
    db = str(tmp_path / "strict.sqlite")
    result = _run_strict(
        """
        import asyncio, sys
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from checkpoint_fixture import CONFIG, build_graph, generate

        async def main(path):
            await generate(path)
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                values = (await build_graph(saver).aget_state(CONFIG)).values
            assert values["step_count"] == 2, values
            assert [type(m).__name__ for m in values["messages"]] == ["HumanMessage", "AIMessage", "AIMessage"]
            print("OK")

        asyncio.run(main(sys.argv[1]))
        """,
        db,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout


def test_old_checkpoint_readable_with_strict_msgpack(old_db):
    result = _run_strict(
        """
        import asyncio, sys
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from checkpoint_fixture import CONFIG, build_graph

        async def main(path):
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                values = (await build_graph(saver).aget_state(CONFIG)).values
            assert values["step_count"] == 2, values
            print("OK")

        asyncio.run(main(sys.argv[1]))
        """,
        old_db,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout
