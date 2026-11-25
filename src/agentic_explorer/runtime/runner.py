"""Run one mission end to end: stream the graph, write traces and report, remember the episode."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl

from agentic_explorer.memory import app_url_hash, write_episode_memory
from agentic_explorer.memory_health import guarded, memory_health
from agentic_explorer.orchestration.graph_base import dedupe_bugs
from agentic_explorer.reporting import (
    append_action_tape_summary,
    build_transcript,
    generate_report,
    make_report_llm,
    report_dir,
    write_report,
)
from agentic_explorer.runtime.retry import with_retry
from agentic_explorer.runtime.tracing import TraceWriter
from agentic_explorer.tools.browser.engine import get_action_tape
from agentic_explorer.tools.browser.provenance import write_provenance
from agentic_explorer.ui import state_emitter
from agentic_explorer.utils import console
from agentic_explorer.utils.messages import message_text

Mission = Dict[str, Any]


@dataclass(frozen=True)
class MissionResult:
    thread_id: str
    bugs: List[str]
    paths: List[str]
    tape_len: int
    memory_errors: int = 0  # memory operations that failed during this mission


def _initial_state(prompt: str) -> Dict[str, Any]:
    return {
        "messages": [HumanMessage(content=prompt)],
        "next_agent": "",
        "step_count": 0,
        "action_tape": [],
        "bugs_found": [],
        "explored_paths": [],
    }


async def stream_mission(
    graph,
    *,
    thread_id: str,
    initial_state: Optional[Dict[str, Any]],
    quiet: bool,
    visual: bool,
    control: Optional[RunControl] = None,
) -> None:
    """Stream the graph once, printing ReAct lines and appending to ``traces.log``.

    ``initial_state=None`` resumes from the thread's saved checkpoint. If ``control``
    is drained the graph stops at the next step boundary and raises ``GraphDrained``.
    """
    trace = TraceWriter(os.path.join(report_dir(thread_id), "traces.log"))
    run_config = {"configurable": {"thread_id": thread_id}}
    current_node: str | None = None

    # v2 yields typed parts ({"type", "ns", "data"}) instead of (mode, payload) tuples.
    # The payloads are the same; `ns` is always () here because agents are not subgraphs.
    async for part in graph.astream(
        initial_state,
        config=run_config,
        stream_mode=["updates", "messages"],
        control=control,
        version="v2",
    ):
        if part["type"] == "updates":
            # part["data"]: {node_name: state_update}
            for node_name, state_update in part["data"].items():
                # A node that changed nothing (e.g. the Summarizer on a short history,
                # which returns {}) is reported by LangGraph as None, not as {}.
                state_update = state_update or {}
                if current_node is not None:
                    console.state_update_end()
                current_node = node_name
                console.state_update(node_name)

                if visual:
                    active = state_update.get("next_agent", node_name) if node_name == "Supervisor" else node_name
                    update_dict = {"active_node": active}
                    if "step_count" in state_update:
                        update_dict["step_count"] = state_update["step_count"]
                    state_emitter.update(**update_dict)
                    # bugs_found and explored_paths arrive as per-node DELTAS
                    # (operator.add reducer in AgentState); accumulate them
                    # rather than overwrite so the cumulative count is preserved.
                    if "bugs_found" in state_update:
                        state_emitter.add_bugs(state_update["bugs_found"])
                    if "explored_paths" in state_update:
                        state_emitter.add_paths(state_update["explored_paths"])
                    state_emitter.emit()

                if "messages" in state_update and state_update["messages"]:
                    messages = state_update["messages"] if isinstance(state_update["messages"], list) else [state_update["messages"]]
                    trace.node_update(node_name, messages)

        elif part["type"] == "messages":
            msg, meta = part["data"]
            node_name = meta.get("langgraph_node") or current_node or "?"

            # Written to traces.log (regardless of --quiet) unless it is a streaming chunk,
            # a system prompt or a message already traced; those are skipped everywhere.
            if not trace.message(node_name, msg):
                continue

            # Console: ReAct lines (compact when --quiet).
            _max = 120 if quiet else 500
            if isinstance(msg, AIMessage):
                text = message_text(msg)
                if text.strip():
                    console.react_thought(node_name, text, max_chars=_max)
            elif isinstance(msg, HumanMessage):
                text = message_text(msg)
                if text.strip():
                    console.info(f"HUMAN: {text[:120]}")

            if visual and isinstance(msg, AIMessage):
                state_emitter.append_thought(node_name, message_text(msg))
                state_emitter.emit()

    if current_node is not None:
        console.state_update_end()


async def run_mission(
    graph,
    mission: Mission,
    *,
    mission_type: str,
    app_url: str,
    store,
    quiet: bool = False,
    visual: bool = False,
    control: Optional[RunControl] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    report_llm_factory: Callable[[], Any] = make_report_llm,
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
) -> MissionResult:
    """Execute ``mission`` on ``graph``, resuming its checkpoint if the thread already has one.

    Transient API errors are retried (resuming from the checkpoint). Afterwards the
    LLM report is written to ``report_<thread_id>/test_report.md`` and the episode
    is stored in long-term memory. If ``control`` is drained, ``GraphDrained``
    propagates (no retry, no report): the checkpoint is saved and running the same
    mission again resumes it.
    """
    thread_id = str(mission["thread_id"])
    prompt = mission["prompt"]
    memory_mark = memory_health.count  # failures before this point belong to earlier missions

    console.mission_start(thread_id, mission_type)

    if visual:
        state_emitter.update(
            mission_id=thread_id,
            mission_description=prompt,
            mission_type=mission_type,
            graph_type=mission_type.lower(),
            step_count=0,
            bugs_count=0,
            bugs_found=[],
            explored_paths=[],
            last_thought="",
            last_action="",
            action_tape_recent=[],
            active_node="",
        )
        state_emitter.start_mission(thread_id, prompt, mission_type)
        state_emitter.emit()

    run_config = {"configurable": {"thread_id": thread_id}}
    existing_state = await graph.aget_state(run_config)
    resuming = bool(existing_state.values)

    os.makedirs(report_dir(thread_id), exist_ok=True)
    if provenance is not None:
        # Written once: a resumed mission keeps the header of its first run.
        write_provenance(thread_id, **provenance)
    trace_path = os.path.join(report_dir(thread_id), "traces.log")
    # A resumed mission (after a stop or a crash) keeps the trace of its first half.
    with open(trace_path, "a" if resuming else "w", encoding="utf-8") as f:
        f.write(f"\n=== RESUMED: {thread_id} ===\n" if resuming else f"=== TRACES: {thread_id} ===\n")

    resume = {"initial_state": None if resuming else _initial_state(prompt)}

    def _resume_from_checkpoint(_attempt: int, _exc: Exception) -> None:
        resume["initial_state"] = None  # retries resume from the saved checkpoint

    await with_retry(
        lambda: stream_mission(
            graph, thread_id=thread_id, initial_state=resume["initial_state"],
            quiet=quiet, visual=visual, control=control,
        ),
        on_retry=_resume_from_checkpoint,
        sleep=sleep,
    )

    console.step(f"Generating report for {thread_id}...")
    final_state = await graph.aget_state(run_config)
    bugs_found = dedupe_bugs(final_state.values.get("bugs_found", []))
    transcript = build_transcript(final_state.values.get("messages", []))
    report_text = await generate_report(thread_id, transcript, bugs_found, report_llm_factory(), sleep=sleep)
    write_report(thread_id, report_text)

    tape = get_action_tape(thread_id)
    explored_paths = final_state.values.get("explored_paths", [])

    # Write episodic memory (session summary + bug catalog). A broken store must not
    # abort the mission, but the failure is logged and counted, not silently dropped.
    if store is not None:
        await guarded(
            "episode memory write",
            lambda: write_episode_memory(
                store, app_url_hash(app_url), thread_id, prompt, tape, bugs_found, explored_paths,
            ),
        )

    console.mission_done(thread_id, len(tape), len(bugs_found))

    if visual:
        state_emitter.end_mission(thread_id)
        state_emitter.emit()

    # Includes failures inside graph nodes (semantic writes, context reads) during this mission.
    memory_errors = memory_health.count_since(memory_mark)
    append_action_tape_summary(
        thread_id, tape_len=len(tape), bugs_count=len(bugs_found), memory_errors=memory_errors,
    )
    return MissionResult(
        thread_id=thread_id, bugs=bugs_found, paths=explored_paths, tape_len=len(tape),
        memory_errors=memory_errors,
    )


def _raise_if_drained(control: RunControl) -> None:
    if control.drain_requested:
        raise GraphDrained(control.drain_reason or "shutdown")


async def run_missions_batch(
    graphs: Mapping[str, Any],
    missions: Sequence[Mission],
    tiers: Sequence[str],
    *,
    app_url: str,
    store,
    control: RunControl,
    quiet: bool = False,
    visual: bool = False,
    provenance: Optional[Mapping[str, Any]] = None,
    cooldown_s: float = 0,
    report_llm_factory: Callable[[], Any] = make_report_llm,
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
) -> List[MissionResult]:
    """Run ``missions`` one after another, each on the graph of its tier.

    Raises ``GraphDrained`` as soon as a drain was requested: before starting a
    mission and before waiting out the cooldown, so a stop never has to sit through it.
    A mission interrupted by the drain keeps its checkpoint and resumes on the next run.
    """
    do_sleep = sleep or asyncio.sleep
    results: List[MissionResult] = []
    for index, (mission, tier) in enumerate(zip(missions, tiers)):
        _raise_if_drained(control)
        results.append(await run_mission(
            graphs[tier], mission,
            mission_type=tier.upper(), app_url=app_url, store=store,
            quiet=quiet, visual=visual, control=control, provenance=provenance,
            report_llm_factory=report_llm_factory, sleep=sleep,
        ))

        if index < len(missions) - 1:
            _raise_if_drained(control)
            if cooldown_s > 0:
                console.info(f"Cooling down {cooldown_s:g}s before next mission...")
                await do_sleep(cooldown_s)
    return results
