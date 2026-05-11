"""Shared LangGraph infrastructure reused by standard and advanced graphs.

Provides:
  - ``AgentState``      — unified TypedDict extended with exploration-tracking fields
  - ``PLAYWRIGHT_TOOL_NAMES`` — names to strip from base_tools in every graph
  - ``filter_base_tools``     — convenience filter
  - ``make_agent_node``       — factory wrapping a compiled agent as a LangGraph node
  - ``make_supervisor_node``  — factory for the routing supervisor
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Dict, List, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_explorer.tools.browser.engine import get_action_tape
from agentic_explorer.utils import console
from agentic_explorer.utils.llm import make_llm  # noqa: F401  re-exported for back-compat


# ---------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    # Immutable chronological log of deterministic browser commands.
    action_tape: Annotated[List[Dict[str, Any]], operator.add]
    # Step counter for loop prevention; always replaced with the latest value.
    step_count: Annotated[int, lambda _old, new: new]
    # Bug summaries collected across all iterations (for final report and supervisor context).
    bugs_found: Annotated[List[str], operator.add]
    # URL paths navigated to, used to guide the supervisor toward unexplored areas.
    explored_paths: Annotated[List[str], operator.add]


# ---------------------------------------------------------
# Tool filtering
# ---------------------------------------------------------

PLAYWRIGHT_TOOL_NAMES = frozenset({
    "click_element", "navigate_browser", "previous_webpage",
    "extract_text", "extract_hyperlinks", "get_elements", "current_webpage",
})


def filter_base_tools(base_tools: list) -> list:
    """Strip raw PlayWrightBrowserToolkit tools from a tool list.

    Agents emit JSON intents via ``execute_browser_command`` instead of
    calling Playwright tools directly, so the toolkit tools must not be
    exposed to them.
    """
    return [t for t in base_tools if getattr(t, "name", "") not in PLAYWRIGHT_TOOL_NAMES]


# ---------------------------------------------------------
# Message introspection helpers
# ---------------------------------------------------------

def _extract_bugs(messages: Sequence[BaseMessage]) -> List[str]:
    """Pull bug summaries out of capture_bug_screenshot ToolMessage results."""
    bugs: List[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "capture_bug_screenshot":
            content = str(msg.content)
            if "Evidence captured" in content or "screenshot" in content.lower():
                bugs.append(content[:300])
    return bugs


def _extract_paths(messages: Sequence[BaseMessage]) -> List[str]:
    """Pull navigated URLs out of execute_browser_command ToolMessage results."""
    paths: List[str] = []
    _url_re = re.compile(r"navigated to (https?://\S+)")
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            paths.extend(_url_re.findall(content))
    return paths[:20]  # cap to avoid unbounded growth


# ---------------------------------------------------------
# ReAct console helpers (used inside agent nodes)
# ---------------------------------------------------------

def _msg_text(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return str(content)


def _summarize_tool_args(tool_call: dict, max_chars: int = 80) -> str:
    import json
    args = tool_call.get("args") or {}
    if not isinstance(args, dict):
        text = str(args)
    elif "action" in args:
        parts = [f"action={args['action']}"]
        for key in ("selector", "url", "value", "key"):
            if args.get(key):
                val = str(args[key])
                if len(val) > 40:
                    val = val[:37] + "..."
                parts.append(f"{key}={val!r}")
        text = " ".join(parts)
    else:
        try:
            text = json.dumps(args, ensure_ascii=False, default=str)
        except Exception:
            text = str(args)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _print_react(msg: BaseMessage, node_name: str, max_chars: int) -> None:
    if isinstance(msg, SystemMessage):
        return
    if isinstance(msg, AIMessage):
        text = _msg_text(msg)
        if text.strip():
            console.react_thought(node_name, text, max_chars=max_chars)
    elif isinstance(msg, HumanMessage):
        text = _msg_text(msg)
        if text.strip():
            console.info(f"HUMAN: {text[:120]}")


# ---------------------------------------------------------
# Node factories
# ---------------------------------------------------------

def make_agent_node(agent, *, name: str = "agent", quiet: bool = False):
    """Return an async LangGraph node function that streams agent messages in real-time.

    Uses ``astream`` internally so THOUGHT / ACTION / OBSERV lines appear on the
    console as the inner agent produces them, rather than batching at node completion.
    """
    _max = 120 if quiet else 500

    async def _node(state: AgentState, config=None) -> dict:
        thread_id = (
            (config or {}).get("configurable", {}).get("thread_id", "default")
            if config else "default"
        )
        before = len(get_action_tape(thread_id))

        out: dict = {}
        seen_count = 0
        async for snapshot in agent.astream(state, config=config):
            out = snapshot
            messages = snapshot.get("messages", [])
            for msg in messages[seen_count:]:
                _print_react(msg, name, _max)
            seen_count = len(messages)

        new_messages: Sequence[BaseMessage] = out.get("messages", [])
        new_tape = list(get_action_tape(thread_id)[before:])
        return {
            "messages": new_messages,
            "action_tape": new_tape,
            "bugs_found": _extract_bugs(new_messages),
            "explored_paths": _extract_paths(new_messages),
        }

    return _node


def make_supervisor_node(llm, agent_names: tuple, app_url: str, max_steps: int):
    """Return an async LangGraph supervisor node with step-limit reset and exploration context.

    The supervisor:
    - Increments the step counter each cycle.
    - Injects a reset directive (with exploration context) when ``max_steps`` is reached.
    - Provides the routing LLM with bugs-found and explored-paths context so it can
      steer agents toward unexplored areas.
    - Routes to one of ``agent_names`` or to ``"FINISH"``.
    """
    available_agents = ", ".join(f"'{n}'" for n in agent_names)
    routing_schema = {
        "title": "SupervisorRouting",
        "description": "Select the next agent to act or FINISH.",
        "type": "object",
        "properties": {
            "next": {"type": "string", "enum": [*agent_names, "FINISH"]},
        },
        "required": ["next"],
    }
    routing_llm = llm.with_structured_output(schema=routing_schema, method="function_calling")

    async def supervisor_node(state: AgentState) -> dict:
        current_step = state.get("step_count", 0) + 1
        reset_triggered = current_step > max_steps

        extra_messages: List[BaseMessage] = []
        if reset_triggered:
            bugs = state.get("bugs_found", [])
            paths = list(dict.fromkeys(state.get("explored_paths", [])))[:6]
            console.warn(
                f"Step limit ({max_steps}) reached at step {current_step - 1}. "
                f"Bugs so far: {len(bugs)}. Resetting to homepage."
            )
            reset_msg = HumanMessage(content=(
                f"[STEP LIMIT — step {current_step - 1}/{max_steps}] "
                f"Bugs discovered so far: {len(bugs)}. "
                f"Areas already explored: {paths or 'none'}. "
                f"Navigate back to {app_url}, reset state, and pick a COMPLETELY DIFFERENT "
                "area of the application. Do not repeat any interaction already tried."
            ))
            extra_messages = [reset_msg]
            current_step = 1

        # Build context-rich routing prompt
        bugs_ctx = f" {len(state.get('bugs_found', []))} bug(s) found so far." if state.get("bugs_found") else ""
        paths_ctx = ""
        if state.get("explored_paths"):
            unique_paths = list(dict.fromkeys(state["explored_paths"]))[:5]
            paths_ctx = f" Already explored: {unique_paths}."

        supervisor_prompt = (
            f"You are the QA Orchestrator.{bugs_ctx}{paths_ctx} "
            f"Decide which agent tests next based on the mission progress. "
            f"Available agents: {available_agents}. "
            "Respond with 'FINISH' only when the mission objective is fully achieved "
            "and sufficient areas have been covered."
        )

        messages_for_routing = list(state["messages"]) + extra_messages
        routing_request = HumanMessage(content="Based on the progress above, which agent should act next?")
        decision = await routing_llm.ainvoke(
            [SystemMessage(content=supervisor_prompt), *messages_for_routing, routing_request]
        )

        result: dict = {"next_agent": decision["next"], "step_count": current_step}
        if extra_messages:
            result["messages"] = extra_messages
        return result

    return supervisor_node


# ---------------------------------------------------------
# Graph compilation helper
# ---------------------------------------------------------

def compile_swarm(workflow, agent_registry: dict, checkpointer):
    """Wire agents → Supervisor → conditional routing → END and compile."""
    from langgraph.graph import END

    agent_names = tuple(agent_registry.keys())

    for agent_name in agent_names:
        workflow.add_edge(agent_name, "Supervisor")

    route_map = {name: name for name in agent_names}
    route_map["FINISH"] = END

    workflow.add_conditional_edges(
        "Supervisor",
        lambda state: state["next_agent"],
        route_map,
    )
    workflow.set_entry_point("Supervisor")
    return workflow.compile(checkpointer=checkpointer)
