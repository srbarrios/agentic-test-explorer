"""Minimal graph over the real ``AgentState`` used to test checkpoint compatibility.

Run ``python tests/checkpoint_fixture.py <out.sqlite>`` to (re)generate a fixture
with whatever langgraph version is installed. The committed fixture
``tests/fixtures/checkpoint_langgraph_1_1_10.sqlite`` was produced with
langgraph 1.1.10 *before* the 1.2 upgrade, so reading it back under newer
versions proves that existing ``agent_memory.sqlite`` files still resume.
"""

from __future__ import annotations

import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from agentic_explorer.orchestration.graph_base import AgentState

THREAD_ID = "fixture_thread"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}
BUG = "Login race: required error shown for an already-filled field"


async def _first(state: AgentState) -> dict:
    return {
        "messages": [AIMessage(content="visited /login", id="ai-1")],
        "step_count": state.get("step_count", 0) + 1,
        "bugs_found": [BUG],
        "explored_paths": ["https://app.test/login"],
        "action_tape": [{"action": "navigate", "ok": True, "params": {"url": "https://app.test/login"}}],
    }


async def _second(state: AgentState) -> dict:
    return {
        "messages": [AIMessage(content="visited /cart", id="ai-2")],
        "step_count": state.get("step_count", 0) + 1,
    }


def build_graph(checkpointer):
    workflow = StateGraph(AgentState)  # type: ignore[arg-type]
    workflow.add_node("first", _first)  # type: ignore[arg-type]
    workflow.add_node("second", _second)  # type: ignore[arg-type]
    workflow.set_entry_point("first")
    workflow.add_edge("first", "second")
    workflow.add_edge("second", END)
    return workflow.compile(checkpointer=checkpointer)


def initial_state() -> dict:
    return {
        "messages": [HumanMessage(content="explore the app", id="human-1")],
        "next_agent": "",
        "action_tape": [],
        "step_count": 0,
        "bugs_found": [],
        "explored_paths": [],
    }


async def generate(path: str) -> None:
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(saver)
        await graph.ainvoke(initial_state(), CONFIG)


if __name__ == "__main__":
    asyncio.run(generate(sys.argv[1]))
