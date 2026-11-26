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
from langgraph.graph.message import add_messages

from agentic_explorer.memory_health import guarded
from agentic_explorer.orchestration.findings import (
    ExtractedFindings,
    dedupe_bugs,  # noqa: F401  re-exported for callers that import it from here
    extract_findings,
    is_dup_key,
    normalize_bug_key,
    same_observation,
    split_findings,
)
from agentic_explorer.tools.browser.engine import get_action_tape
from agentic_explorer.utils import console
from agentic_explorer.utils.llm import make_llm  # noqa: F401  re-exported for back-compat
from agentic_explorer.utils.messages import clip_text, message_text


BROWSER_AGENT_RULES = (
    " Context policy: use progressive disclosure. Start with mission + current DOM; "
    "call MCP/Skill tools only for the specific feature under test, and request details "
    "only when needed. Browser policy: you are the brain, not the hands. Interact only "
    "through execute_browser_command JSON intents after get_dom_snapshot. Allowed actions: "
    "navigate, click, fill, press, select_option, hover, wait_for, scroll, extract_text, "
    "snapshot, check_page_health. Selector policy: prefer data-test-subj, then ARIA/roles, "
    "then semantic text; never use XPath, nth-child/nth-of-type, or guessed structural CSS. "
    "Failure policy: on any UI error, missing element, tool failure, or visual anomaly, call "
    "capture_bug_screenshot, then generate_reproduction_spec immediately. "
    "Bug reporting format (MANDATORY): your FINAL message before yielding control MUST "
    "contain a single <bugs_found>...</bugs_found> block listing EVERY distinct issue you "
    "observed during this turn — one JSON entry per issue. Count as an issue: any UI bug, "
    "validation error, missing element, broken interaction, visual anomaly, accessibility "
    "violation, missing label, missing empty-state messaging, missing progress indicator, "
    "missing tooltip/help, confusing flow, or any UX gap you would flag in a code review. "
    "Do NOT aggregate multiple issues into one entry. Do NOT omit minor issues — list them "
    "all. If you found zero issues, emit <bugs_found>[]</bugs_found> explicitly. Each entry "
    'must be: {"title": "<short imperative>", "summary": "<what + where + repro hint>", '
    '"location": "<url or selector>", "severity": "critical|major|minor"}. '
    "Evidence (add whenever you can): "
    '"observed" = the exact value you saw, "observed_at" = where you read it (selector or '
    'region text, and URL), "expected" = what it should have been, "confidence" = '
    '"confirmed" or "uncertain". Use "uncertain" whenever you could not determine the '
    "verdict yourself (flaky, could not reproduce, no way to know the right value, cannot "
    "tell a bug from intended behaviour): an honest uncertain is worth more than a "
    "confirmed that might be wrong, so never round up. Omit a field rather than guessing "
    "it. Example: "
    '<bugs_found>[{"title": "Login validation race", "summary": "Filling fields rapidly '
    'then clicking submit shows a required-field error for an already-filled field", '
    '"location": "/login", "severity": "major"}, {"title": "Cart total ignores discount", '
    '"summary": "Cart total shows the undiscounted price after applying SAVE20", '
    '"location": "/cart", "severity": "major", "observed": "$120.00", "observed_at": '
    '"[data-test=\\"total\\"] on /cart", "expected": "$96.00", "confidence": "confirmed"}, '
    '{"title": "Coupon may reject lowercase codes", "summary": "SAVE20 applied but save20 '
    'seemed to do nothing; the spec does not say if codes are case-sensitive", '
    '"location": "/cart", "severity": "minor", "observed": "no change after save20", '
    '"confidence": "uncertain"}]</bugs_found>'
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
    # add_messages (not operator.add) so RemoveMessage from the Summarizer node
    # actually deletes history and messages with the same id are replaced.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str
    # Recent browser commands kept in state (capped); full log persisted to JSONL.
    action_tape: Annotated[List[Dict[str, Any]], _bounded_tape_reducer]
    # Step counter for loop prevention; always replaced with the latest value.
    step_count: Annotated[int, lambda _old, new: new]
    # Bug summaries collected across all iterations (for final report and supervisor context).
    # Confirmed findings only. Findings the agent could not confirm are counted and shown apart.
    bugs_found: Annotated[List[str], operator.add]
    uncertain_findings: Annotated[List[str], operator.add]
    # Observed / expected / where, one record per finding (see orchestration/findings.py).
    bug_evidence: Annotated[List[Dict[str, str]], operator.add]
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

# Key normalisation, deduplication and the <bugs_found> parser live in findings.py;
# the underscore names stay importable from here for existing callers and tests.
_normalize_bug_key = normalize_bug_key
_is_dup_key = is_dup_key


def _extract_bugs_from_text(text: str) -> List[str]:
    """Confirmed bug descriptions from ``<bugs_found>`` JSON blocks, with prose fallback."""
    return extract_findings(text).confirmed


def _extract_turn_findings(messages: Sequence[BaseMessage]) -> ExtractedFindings:
    """Findings of one agent turn: screenshots, ``<bugs_found>`` blocks and prose.

    For each screenshot tool call, the agent passes the bug title in
    ``bug_summary`` (read from AIMessage.tool_calls, since the ToolMessage result
    "Evidence captured!..." drops that argument). An explicit ``uncertain`` verdict
    in a closing block wins over the screenshot that preceded it.

    The agent usually also calls ``generate_reproduction_spec`` with the same ``bug_summary``
    plus ``observed`` / ``expected``. That is evidence for the screenshot bug, and when the
    closing block reports the very same observation under another title it is the same defect:
    the screenshot entry is dropped instead of counting it twice.
    """
    text_found = ExtractedFindings()
    screenshots: List[str] = []
    spec_evidence: Dict[str, Dict[str, str]] = {}  # bug key -> observed / expected given to the spec tool
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                args = tc.get("args")
                if not isinstance(args, dict):
                    continue
                summary = args.get("bug_summary")
                if tc.get("name") == "capture_bug_screenshot" and summary:
                    screenshots.append(str(summary))
                elif tc.get("name") == "generate_reproduction_spec" and summary:
                    given = {k: str(args[k]).strip() for k in ("observed", "expected") if args.get(k)}
                    if given:
                        spec_evidence[normalize_bug_key(_clean_screenshot(str(summary)))] = given
            part = extract_findings(message_text(msg))
            text_found.confirmed += part.confirmed
            text_found.uncertain += part.uncertain
            text_found.evidence += part.evidence

    uncertain_keys = [normalize_bug_key(u) for u in text_found.uncertain]
    screenshots = [s for s in screenshots if not is_dup_key(normalize_bug_key(_clean_screenshot(s)), uncertain_keys)]

    kept: List[str] = []
    screenshot_evidence: List[Dict[str, str]] = []
    for shot in screenshots:
        summary = _clean_screenshot(shot)
        given = spec_evidence.get(normalize_bug_key(summary))
        if given and same_observation(given.get("observed"), given.get("expected"), text_found.evidence):
            continue  # the closing block reports this same observation
        kept.append(summary)
        if given:
            screenshot_evidence.append({"bug": summary, "confidence": "confirmed", **given})

    confirmed, uncertain = split_findings(kept + text_found.confirmed, text_found.uncertain)
    return ExtractedFindings(
        confirmed=confirmed, uncertain=uncertain, evidence=text_found.evidence + screenshot_evidence,
    )


def _clean_screenshot(summary: str) -> str:
    return summary.strip(": ").strip()[:500]


def _extract_bugs(messages: Sequence[BaseMessage]) -> List[str]:
    """Confirmed bugs of one agent turn (see ``_extract_turn_findings``)."""
    return _extract_turn_findings(messages).confirmed





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

def _compact_message(msg: BaseMessage, max_chars: int = 900) -> str:
    """Summarize one LangChain message for supervisor routing.

    The supervisor only needs enough context to pick the next worker. Passing the
    complete historical transcript on every turn can exhaust context on long
    missions, so this keeps recent intent, tool names, and tool outcomes only.
    """
    if isinstance(msg, SystemMessage):
        return ""
    role = msg.type.upper()
    text = clip_text(message_text(msg), max_chars)
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
            return clip_text(message_text(msg), 1800)
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
        "BUGS_FOUND:\n" + ("\n".join(f"- {clip_text(b, 240)}" for b in bugs[-8:]) if bugs else "- none"),
        "EXPLORED_URLS:\n" + ("\n".join(f"- {p}" for p in paths) if paths else "- none"),
    ]
    if memory_context:
        sections.append(memory_context)
    if reset_lines:
        sections.append("RESET_DIRECTIVE:\n" + "\n".join(f"- {line}" for line in reset_lines))
    return "\n\n".join(sections)


def _print_react(msg: BaseMessage, node_name: str, max_chars: int) -> None:
    if isinstance(msg, SystemMessage):
        return
    if isinstance(msg, AIMessage):
        text = message_text(msg)
        if text.strip():
            console.react_thought(node_name, text, max_chars=max_chars)
    elif isinstance(msg, HumanMessage):
        text = message_text(msg)
        if text.strip():
            console.info(f"HUMAN: {text[:120]}")


# ---------------------------------------------------------
# Node factories
# ---------------------------------------------------------

def _sanitize_messages_for_model(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Return a plain-text-only transcript safe for Anthropic adapters.

    The inner agent already owns the active system prompt and tool-calling loop,
    so historical state is passed as narrative context only:
    - drop all SystemMessage entries (prevents non-consecutive system blocks)
    - drop all ToolMessage entries (prevents orphaned tool_result blocks)
    - normalize remaining Human/AI messages to plain text content only
    """
    sanitized: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, (SystemMessage, ToolMessage)):
            continue

        text = message_text(msg).strip()
        if not text:
            continue

        if isinstance(msg, HumanMessage):
            sanitized.append(HumanMessage(content=text))
        elif isinstance(msg, AIMessage):
            sanitized.append(AIMessage(content=text))

    return sanitized


def make_agent_node(agent, *, name: str = "agent", quiet: bool = False, app_url_hash: str = ""):
    """Return an async LangGraph node function that streams agent messages in real-time.

    Uses ``astream`` internally so THOUGHT / ACTION / OBSERV lines appear on the
    console as the inner agent produces them, rather than batching at node completion.
    """
    _max = 120 if quiet else 500

    async def _node(state: AgentState, config=None, *, store=None) -> dict:
        from agentic_explorer.ui import state_emitter
        if state_emitter.is_enabled():
            state_emitter.update(active_node=name)
            state_emitter.emit()

        thread_id = (
            (config or {}).get("configurable", {}).get("thread_id", "default")
            if config else "default"
        )
        before = len(get_action_tape(thread_id))

        # Sanitize messages before passing to the inner agent to avoid Anthropic API
        # errors (non-consecutive system messages, orphaned tool_result blocks).
        filtered_state = dict(state)
        if "messages" in filtered_state:
            filtered_state["messages"] = _sanitize_messages_for_model(
                filtered_state["messages"]
            )
        n_input = len(filtered_state.get("messages", []))

        out: dict = {}
        seen_count = 0
        async for snapshot in agent.astream(filtered_state, config=config):
            out = snapshot
            messages = snapshot.get("messages", [])
            for msg in messages[seen_count:]:
                _print_react(msg, name, _max)
                if state_emitter.is_enabled() and isinstance(msg, AIMessage):
                    text = message_text(msg)
                    if text.strip():
                        state_emitter.append_thought(name, text)
                        state_emitter.emit()
            seen_count = len(messages)

        # The compiled agent streams its whole state: the (sanitized) input comes back
        # first, then what the agent produced. Only the latter is new; returning all of
        # it re-added the history and re-counted every earlier bug on each turn.
        new_messages: Sequence[BaseMessage] = list(out.get("messages", []))[n_input:]
        new_tape = list(get_action_tape(thread_id)[before:])

        if store and app_url_hash and new_tape:
            from agentic_explorer.memory import write_semantic_memories_from_tape
            await guarded(
                "semantic memory write",
                lambda: write_semantic_memories_from_tape(store, app_url_hash, new_tape, agent_name=name),
            )

        findings = _extract_turn_findings(new_messages)
        return {
            "messages": new_messages,
            "action_tape": new_tape,
            "bugs_found": findings.confirmed,
            "uncertain_findings": findings.uncertain,
            "bug_evidence": findings.evidence,
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
            memory_context = await guarded(
                "memory context read",
                lambda: format_memory_context(store, app_url_hash),
                default="",
            )

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

        from agentic_explorer.ui import state_emitter
        if state_emitter.is_enabled():
            state_emitter.update(
                active_node=decision["next"],
                step_count=current_step,
                bugs_count=len(state.get("bugs_found", [])),
                explored_paths=list(dict.fromkeys(state.get("explored_paths", [])))[:20],
            )
            state_emitter.emit()

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
    recent messages. Messages in between are replaced with a single HumanMessage
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

        # The summary takes the id of the first old message that can be addressed, so
        # add_messages replaces that message in place and the summary keeps its
        # chronological spot (after the mission prompt, before the recent messages).
        # The remaining old messages are removed.
        removable = [msg for msg in old_messages if getattr(msg, "id", None)]
        if not removable:
            return {}

        summary = HumanMessage(content=summary_text, id=removable[0].id)
        return {"messages": [summary, *(RemoveMessage(id=msg.id) for msg in removable[1:])]}

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
