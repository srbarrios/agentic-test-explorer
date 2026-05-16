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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from agentic_explorer.tools.browser.engine import get_action_tape
from agentic_explorer.utils import console
from agentic_explorer.utils.llm import make_llm  # noqa: F401  re-exported for back-compat


BROWSER_AGENT_RULES = (
    " Context policy: use progressive disclosure. Start with mission + current DOM; "
    "call MCP/Skill tools only for the specific feature under test, and request details "
    "only when needed. Browser policy: you are the brain, not the hands. Interact only "
    "through execute_browser_command JSON intents after get_dom_snapshot. Allowed actions: "
    "navigate, click, fill, press, select_option, hover, wait_for, scroll, extract_text, "
    "snapshot, check_page_health. Selector policy: prefer data-test-subj, then ARIA/roles, "
    "then semantic text; never use XPath, nth-child/nth-of-type, or guessed structural CSS. "
    "Failure policy: on any UI error, missing element, tool failure, or visual anomaly, call "
    "capture_bug_screenshot, then generate_reproduction_spec immediately."
)


def make_browser_agent_prompt(role: str, app_context: str, focus: str) -> str:
    """Build a compact system prompt for browser-driving QA agents."""
    return f"You are {role}.{app_context} {focus} {BROWSER_AGENT_RULES}"


# ---------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------

_ACTION_TAPE_STATE_LIMIT = 50


def _bounded_tape_reducer(old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the most recent entries in state; the full tape lives in JSONL on disk."""
    combined = (old or []) + (new or [])
    return combined[-_ACTION_TAPE_STATE_LIMIT:]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    # Recent browser commands kept in state (capped); full log persisted to JSONL.
    action_tape: Annotated[List[Dict[str, Any]], _bounded_tape_reducer]
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


def _clip_text(text: str, max_chars: int) -> str:
    """Return ``text`` capped for LLM routing/report prompts."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 24:
        return text[:max(0, max_chars)]
    return text[: max_chars - 24].rstrip() + " … [truncated]"


def _compact_message(msg: BaseMessage, max_chars: int = 900) -> str:
    """Summarize one LangChain message for supervisor routing.

    The supervisor only needs enough context to pick the next worker. Passing the
    complete historical transcript on every turn can exhaust context on long
    missions, so this keeps recent intent, tool names, and tool outcomes only.
    """
    if isinstance(msg, SystemMessage):
        return ""
    role = msg.type.upper()
    text = _clip_text(_msg_text(msg), max_chars)
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        tools = ", ".join(tc.get("name", "unknown_tool") for tc in tool_calls)
        text = f"{text} [tool_calls: {tools}]" if text else f"[tool_calls: {tools}]"
    if isinstance(msg, ToolMessage):
        tool_name = getattr(msg, "name", "tool") or "tool"
        role = f"TOOL:{tool_name}"
    return f"{role}: {text}" if text else ""


def _first_human_message(messages: Sequence[BaseMessage]) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return _clip_text(_msg_text(msg), 1800)
    return "<mission prompt unavailable>"


def _format_recent_progress(messages: Sequence[BaseMessage], *, limit: int = 12) -> str:
    compact = [line for msg in messages if (line := _compact_message(msg))]
    recent = compact[-limit:]
    if not recent:
        return "- No agent progress yet."
    return "\n".join(f"- {line}" for line in recent)


def _format_recent_actions(actions: Sequence[Dict[str, Any]], *, limit: int = 10) -> str:
    recent = list(actions)[-limit:]
    if not recent:
        return "- No browser actions recorded yet."
    lines = []
    for entry in recent:
        action = entry.get("action", "unknown")
        ok = "ok" if entry.get("ok") else "error"
        params = entry.get("params") or {}
        selector = params.get("selector") or params.get("url") or params.get("key") or ""
        lines.append(f"- {action}({selector}) -> {ok}")
    return "\n".join(lines)


def _build_routing_context(state: AgentState, extra_messages: Sequence[BaseMessage], memory_context: str = "") -> str:
    """Build a bounded routing brief for the supervisor LLM."""
    messages = list(state.get("messages", []))
    bugs = list(state.get("bugs_found", []))
    paths = list(dict.fromkeys(state.get("explored_paths", [])))[:8]
    reset_lines = [line for msg in extra_messages if (line := _compact_message(msg, max_chars=700))]

    sections = [
        "MISSION:\n" + _first_human_message(messages),
        "RECENT_PROGRESS:\n" + _format_recent_progress(messages),
        "RECENT_BROWSER_ACTIONS:\n" + _format_recent_actions(state.get("action_tape", [])),
        "BUGS_FOUND:\n" + ("\n".join(f"- {_clip_text(b, 240)}" for b in bugs[-8:]) if bugs else "- none"),
        "EXPLORED_URLS:\n" + ("\n".join(f"- {p}" for p in paths) if paths else "- none"),
    ]
    if memory_context:
        sections.append(memory_context)
    if reset_lines:
        sections.append("RESET_DIRECTIVE:\n" + "\n".join(f"- {line}" for line in reset_lines))
    return "\n\n".join(sections)


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

def make_agent_node(agent, *, name: str = "agent", quiet: bool = False, app_url_hash: str = ""):
    """Return an async LangGraph node function that streams agent messages in real-time.

    Uses ``astream`` internally so THOUGHT / ACTION / OBSERV lines appear on the
    console as the inner agent produces them, rather than batching at node completion.
    """
    _max = 120 if quiet else 500

    async def _node(state: AgentState, config=None, *, store=None) -> dict:
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

        if store and app_url_hash and new_tape:
            from agentic_explorer.memory import write_semantic_memories_from_tape
            try:
                await write_semantic_memories_from_tape(
                    store, app_url_hash, new_tape, agent_name=name,
                )
            except Exception:
                pass

        return {
            "messages": new_messages,
            "action_tape": new_tape,
            "bugs_found": _extract_bugs(new_messages),
            "explored_paths": _extract_paths(new_messages),
        }

    return _node


def make_supervisor_node(llm, agent_names: tuple, app_url: str, max_steps: int, agent_descriptions: str = None, app_url_hash: str = ""):
    """Return an async LangGraph supervisor node with step-limit reset and exploration context.

    The supervisor:
    - Increments the step counter each cycle.
    - Injects a reset directive (with exploration context) when ``max_steps`` is reached.
    - Provides the routing LLM with bugs-found and explored-paths context so it can
      steer agents toward unexplored areas.
    - Reads cross-session memory (known pages, quirks) from the Store when available.
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

    async def supervisor_node(state: AgentState, *, store=None) -> dict:
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

        memory_context = ""
        if store and app_url_hash:
            from agentic_explorer.memory import format_memory_context
            try:
                memory_context = await format_memory_context(store, app_url_hash)
            except Exception:
                pass

        # Build context-rich routing prompt
        bugs_ctx = f" {len(state.get('bugs_found', []))} bug(s) found so far." if state.get("bugs_found") else ""
        paths_ctx = ""
        if state.get("explored_paths"):
            unique_paths = list(dict.fromkeys(state["explored_paths"]))[:5]
            paths_ctx = f" Already explored: {unique_paths}."

        avail_str = f"\n{agent_descriptions}\n" if agent_descriptions else available_agents
        supervisor_prompt = (
            f"You are the QA Orchestrator.{bugs_ctx}{paths_ctx} "
            f"Decide which agent tests next based on the mission progress. "
            f"Available agents: {avail_str}. "
            "Respond with 'FINISH' only when the mission objective is fully achieved "
            "and sufficient areas have been covered."
        )

        routing_context = _build_routing_context(state, extra_messages, memory_context=memory_context)
        routing_request = HumanMessage(content=(
            "Use this compact mission context to choose the next agent. Do not request "
            "full history unless necessary; route based on current objective, recent "
            f"progress, and coverage gaps.\n\n{routing_context}\n\nWhich agent should act next?"
        ))
        decision = await routing_llm.ainvoke(
            [SystemMessage(content=supervisor_prompt), routing_request]
        )

        result: dict = {"next_agent": decision["next"], "step_count": current_step}
        if extra_messages:
            result["messages"] = extra_messages
        return result

    return supervisor_node


# ---------------------------------------------------------
# Message summarization node
# ---------------------------------------------------------

_SUMMARIZER_KEEP_RECENT = 20
_SUMMARIZER_COMPACT_LIMIT = 40


def make_summarizer_node(*, keep_recent: int = _SUMMARIZER_KEEP_RECENT):
    """Return a node that compresses old messages to prevent unbounded state growth.

    Keeps the first HumanMessage (mission prompt) and the ``keep_recent`` most
    recent messages. Messages in between are replaced with a single SystemMessage
    containing a compact text summary — no LLM call required.
    """

    async def summarizer_node(state: AgentState) -> dict:
        messages = list(state.get("messages", []))
        if len(messages) <= keep_recent + 1:
            return {}

        first_human_idx = next(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), 0
        )
        cutoff = len(messages) - keep_recent

        if cutoff <= first_human_idx + 1:
            return {}

        old_messages = messages[first_human_idx + 1 : cutoff]
        if not old_messages:
            return {}

        compact_lines = []
        for msg in old_messages:
            line = _compact_message(msg, max_chars=200)
            if line:
                compact_lines.append(line)
        compact_lines = compact_lines[-_SUMMARIZER_COMPACT_LIMIT:]

        summary_text = (
            f"PREVIOUS PROGRESS SUMMARY ({len(old_messages)} messages compacted):\n"
            + "\n".join(f"  {line}" for line in compact_lines)
        )

        removals = [
            RemoveMessage(id=msg.id)
            for msg in old_messages
            if getattr(msg, "id", None)
        ]
        if not removals:
            return {}

        return {
            "messages": removals + [SystemMessage(content=summary_text)],
        }

    return summarizer_node


# ---------------------------------------------------------
# Graph compilation helper
# ---------------------------------------------------------

def compile_swarm(workflow, agent_registry: dict, checkpointer, store=None):
    """Wire agents → Summarizer → Supervisor → conditional routing → END and compile."""
    from langgraph.graph import END

    agent_names = tuple(agent_registry.keys())

    workflow.add_node("Summarizer", make_summarizer_node())

    for agent_name in agent_names:
        workflow.add_edge(agent_name, "Summarizer")

    workflow.add_edge("Summarizer", "Supervisor")

    route_map = {name: name for name in agent_names}
    route_map["FINISH"] = END

    workflow.add_conditional_edges(
        "Supervisor",
        lambda state: state["next_agent"],
        route_map,
    )
    workflow.set_entry_point("Supervisor")
    compile_kwargs = {"checkpointer": checkpointer}
    if store is not None:
        compile_kwargs["store"] = store
    return workflow.compile(**compile_kwargs)
