"""Cross-session memory for the agentic test explorer.

Provides utilities for semantic, episodic, and procedural memory backed by
a LangGraph Store (AsyncSqliteStore or compatible).  Procedural memory
reflection and agent observation management are powered by the Langmem SDK.

Namespace layout:
  ("app", "{app_url_hash}", "pages")               — page structure knowledge
  ("app", "{app_url_hash}", "selectors")            — selector reliability tracking
  ("app", "{app_url_hash}", "quirks")               — application-specific behaviors
  ("app", "{app_url_hash}", "agent_observations")   — Langmem-managed agent observations
  ("episodes", "{app_url_hash}", "sessions")        — completed session summaries
  ("episodes", "{app_url_hash}", "bugs")            — deduplicated bug catalog
  ("procedures", "{app_url_hash}", "agent_prompts") — per-agent prompt supplements
  ("procedures", "{app_url_hash}", "routing_rules") — supervisor routing refinements
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse


def app_url_hash(url: str) -> str:
    """Deterministic short hash of an app URL for namespace scoping."""
    normalized = urlparse(url)._replace(fragment="", query="").geturl().rstrip("/")
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


# ---------------------------------------------------------
# Semantic memory: page knowledge
# ---------------------------------------------------------

async def update_page_knowledge(
    store,
    url_hash: str,
    page_url: str,
    page_title: str = "",
) -> None:
    """Upsert page knowledge from a successful navigation."""
    path = urlparse(page_url).path or "/"
    ns = ("app", url_hash, "pages")
    key = path.replace("/", "_").strip("_") or "root"

    existing = await store.aget(ns, key)
    data: Dict[str, Any] = existing.value if existing else {
        "url": path,
        "visit_count": 0,
    }
    data["visit_count"] = data.get("visit_count", 0) + 1
    data["title"] = page_title or data.get("title", "")
    data["last_seen"] = datetime.now(timezone.utc).isoformat()
    await store.aput(ns, key, data)


# ---------------------------------------------------------
# Semantic memory: selector reliability
# ---------------------------------------------------------

async def track_selector(
    store,
    url_hash: str,
    selector: str,
    success: bool,
    page_url: str = "",
) -> None:
    """Track a selector's success/failure for reliability scoring."""
    if not selector:
        return
    ns = ("app", url_hash, "selectors")
    sel_key = hashlib.md5(selector.encode()).hexdigest()[:12]

    existing = await store.aget(ns, sel_key)
    data: Dict[str, Any] = existing.value if existing else {
        "selector": selector,
        "success_count": 0,
        "failure_count": 0,
    }
    if success:
        data["success_count"] = data.get("success_count", 0) + 1
    else:
        data["failure_count"] = data.get("failure_count", 0) + 1
    if page_url:
        data["page_pattern"] = urlparse(page_url).path or "/"
    data["last_used"] = datetime.now(timezone.utc).isoformat()
    await store.aput(ns, sel_key, data)


# ---------------------------------------------------------
# Semantic memory: application quirks
# ---------------------------------------------------------

async def record_quirk(
    store,
    url_hash: str,
    description: str,
    page: str,
    category: str = "general",
    discovered_by: str = "",
) -> None:
    """Record an application quirk or unexpected behavior."""
    ns = ("app", url_hash, "quirks")
    quirk_key = hashlib.md5(f"{page}:{description[:100]}".encode()).hexdigest()[:12]

    existing = await store.aget(ns, quirk_key)
    if existing:
        data = existing.value
        data["confirmed_count"] = data.get("confirmed_count", 0) + 1
        data["last_seen"] = datetime.now(timezone.utc).isoformat()
    else:
        data = {
            "description": description[:500],
            "page": page,
            "category": category,
            "discovered_by": discovered_by,
            "confirmed_count": 1,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }
    await store.aput(ns, quirk_key, data)


# ---------------------------------------------------------
# Semantic memory: write from action tape entries
# ---------------------------------------------------------

_URL_RE = re.compile(r"navigated to (https?://\S+)")


async def write_semantic_memories_from_tape(
    store,
    url_hash: str,
    tape_entries: List[Dict[str, Any]],
    agent_name: str = "",
) -> None:
    """Extract and write semantic memories from a batch of action tape entries."""
    for entry in tape_entries:
        page_url = entry.get("page_url", "")
        page_title = entry.get("page_title", "")
        params = entry.get("params") or {}
        selector = params.get("selector", "")
        ok = entry.get("ok", False)

        if page_url and ok:
            await update_page_knowledge(store, url_hash, page_url, page_title)

        if selector:
            await track_selector(store, url_hash, selector, ok, page_url)

        if not ok and entry.get("error"):
            error = str(entry.get("error", ""))[:200]
            action = entry.get("action", "unknown")
            if page_url and len(error) > 20:
                await record_quirk(
                    store, url_hash,
                    description=f"{action} failed: {error}",
                    page=urlparse(page_url).path or "/",
                    category="error",
                    discovered_by=agent_name,
                )


# ---------------------------------------------------------
# Semantic memory: read for supervisor context
# ---------------------------------------------------------

async def _semantic_search(store, namespace, limit: int, query: str = ""):
    """Search store with optional semantic query; fall back to plain scan."""
    if query:
        try:
            return await store.asearch(namespace, query=query, limit=limit)
        except (TypeError, Exception):
            pass
    return await store.asearch(namespace, limit=limit)


async def format_memory_context(store, url_hash: str, mission_context: str = "") -> str:
    """Build a MEMORY_CONTEXT section for the supervisor routing prompt.

    When *mission_context* is provided and the store has an embedding index,
    results are ranked by semantic relevance to the current mission.
    """
    sections: List[str] = []

    pages = await store.asearch(("app", url_hash, "pages"), limit=15)
    if pages:
        page_lines = []
        for item in sorted(pages, key=lambda p: p.value.get("visit_count", 0), reverse=True):
            v = item.value
            page_lines.append(f"- {v.get('url', '?')} (visited {v.get('visit_count', 0)}x)")
        sections.append("KNOWN_PAGES:\n" + "\n".join(page_lines[:10]))

    quirks = await _semantic_search(store, ("app", url_hash, "quirks"), 8, mission_context)
    if quirks:
        quirk_lines = [f"- {q.value.get('description', '?')[:200]}" for q in quirks]
        sections.append("KNOWN_QUIRKS:\n" + "\n".join(quirk_lines))

    bugs = await _semantic_search(store, ("episodes", url_hash, "bugs"), 6, mission_context)
    if bugs:
        bug_lines = [f"- {b.value.get('summary', '?')[:200]} (seen {b.value.get('seen_count', 1)}x)" for b in bugs]
        sections.append("KNOWN_BUGS_FROM_PAST_SESSIONS:\n" + "\n".join(bug_lines))

    observations = await _semantic_search(store, ("app", url_hash, "agent_observations"), 5, mission_context)
    if observations:
        obs_lines = [f"- {o.value.get('content', str(o.value))[:200]}" for o in observations]
        sections.append("AGENT_OBSERVATIONS:\n" + "\n".join(obs_lines))

    priority_text = await prioritize_pages(store, url_hash)
    if priority_text:
        sections.append(priority_text)

    if not sections:
        return ""
    return "MEMORY_CONTEXT:\n" + "\n\n".join(sections)


# ---------------------------------------------------------
# Episodic memory: session summaries
# ---------------------------------------------------------

async def write_session_summary(
    store,
    url_hash: str,
    thread_id: str,
    mission_prompt: str,
    action_tape: List[Dict[str, Any]],
    bugs_found: List[str],
    explored_paths: List[str],
) -> None:
    """Write an episode summary for a completed test session."""
    ns = ("episodes", url_hash, "sessions")

    successful = sum(1 for e in action_tape if e.get("ok"))
    data: Dict[str, Any] = {
        "thread_id": thread_id,
        "mission_prompt_summary": mission_prompt[:300],
        "total_actions": len(action_tape),
        "successful_actions": successful,
        "bugs_found": len(bugs_found),
        "pages_covered": list(dict.fromkeys(explored_paths))[:20],
        "outcome": "bugs_found" if bugs_found else ("completed" if action_tape else "empty"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    await store.aput(ns, thread_id, data)


# ---------------------------------------------------------
# Episodic memory: bug catalog
# ---------------------------------------------------------

def _bug_fingerprint(summary: str, page: str) -> str:
    """Deterministic fingerprint for deduplicating bugs."""
    normalized = re.sub(r"\s+", " ", f"{page}:{summary[:100]}").strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


async def catalog_bug(
    store,
    url_hash: str,
    summary: str,
    page: str = "",
    session_id: str = "",
) -> None:
    """Add or increment a bug in the cross-session bug catalog."""
    ns = ("episodes", url_hash, "bugs")
    fp = _bug_fingerprint(summary, page)

    existing = await store.aget(ns, fp)
    if existing:
        data = existing.value
        data["seen_count"] = data.get("seen_count", 1) + 1
        data["last_seen"] = datetime.now(timezone.utc).isoformat()
        sessions = data.get("sessions", [])
        if session_id and session_id not in sessions:
            sessions.append(session_id)
            data["sessions"] = sessions[-10:]
    else:
        data = {
            "fingerprint": fp,
            "summary": summary[:500],
            "page": page,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "seen_count": 1,
            "sessions": [session_id] if session_id else [],
            "status": "open",
        }
    await store.aput(ns, fp, data)


# ---------------------------------------------------------
# Episodic memory: write from final state (post-mission)
# ---------------------------------------------------------

async def write_episode_memory(
    store,
    url_hash: str,
    thread_id: str,
    mission_prompt: str,
    action_tape: List[Dict[str, Any]],
    bugs_found: List[str],
    explored_paths: List[str],
) -> None:
    """Write all episodic memories for a completed mission.

    Called from main.py after report generation.
    """
    await write_session_summary(
        store, url_hash, thread_id,
        mission_prompt, action_tape, bugs_found, explored_paths,
    )

    for bug_text in bugs_found:
        page = ""
        for path in reversed(explored_paths):
            if path:
                page = urlparse(path).path or "/"
                break
        await catalog_bug(store, url_hash, bug_text, page=page, session_id=thread_id)


# ---------------------------------------------------------
# Episodic memory: recall tool for agents
# ---------------------------------------------------------

def get_recall_tool(store, url_hash: str):
    """Return a LangChain tool that agents can call to recall past findings.

    Uses semantic search when the store has an embedding index configured,
    otherwise falls back to keyword matching.
    """
    from langchain_core.tools import tool

    async def _search(namespace, query: str, limit: int):
        """Attempt semantic search; fall back to keyword scan."""
        try:
            return await store.asearch(namespace, query=query, limit=limit)
        except (TypeError, Exception):
            items = await store.asearch(namespace, limit=limit * 3)
            q = query.strip().lower()
            return [
                it for it in items
                if any(q in str(v).lower() for v in it.value.values() if isinstance(v, str))
            ][:limit]

    @tool
    async def recall_past_findings(query: str) -> str:
        """Recall bugs, quirks, and session history related to a query.

        Args:
            query: What to search for — a page path, feature name, or description
                   (e.g. "login page", "form validation", "/systems").
        """
        results: List[str] = []

        bugs = await _search(("episodes", url_hash, "bugs"), query, 8)
        if bugs:
            results.append(f"KNOWN BUGS ({len(bugs)}):")
            for b in bugs:
                v = b.value
                results.append(f"  - {v.get('summary', '?')[:200]} (seen {v.get('seen_count', 1)}x, status: {v.get('status', 'open')})")

        sessions = await _search(("episodes", url_hash, "sessions"), query, 5)
        if sessions:
            results.append(f"\nPAST SESSIONS ({len(sessions)}):")
            for s in sessions:
                v = s.value
                results.append(
                    f"  - {v.get('thread_id', '?')}: {v.get('total_actions', 0)} actions, "
                    f"{v.get('bugs_found', 0)} bugs, outcome={v.get('outcome', '?')}"
                )

        quirks = await _search(("app", url_hash, "quirks"), query, 5)
        if quirks:
            results.append(f"\nKNOWN QUIRKS ({len(quirks)}):")
            for q in quirks:
                v = q.value
                results.append(f"  - {v.get('description', '?')[:200]} (confirmed {v.get('confirmed_count', 1)}x)")

        if not results:
            return f"No past findings for '{query}'. This area may not have been tested before."

        return "\n".join(results)

    return recall_past_findings


# ---------------------------------------------------------
# Agent proactive memory tool (Langmem)
# ---------------------------------------------------------

def get_memory_management_tool(store, url_hash: str):
    """Return a Langmem tool allowing agents to proactively record observations."""
    from langmem import create_manage_memory_tool

    return create_manage_memory_tool(
        namespace=("app", url_hash, "agent_observations"),
        instructions=(
            "Record important observations about the application under test that "
            "may be useful in future testing sessions. Use this tool when you notice: "
            "unexpected UI behaviors, performance issues, confusing UX patterns, "
            "accessibility problems, or successful testing strategies worth repeating."
        ),
        store=store,
        name="record_observation",
    )


# ---------------------------------------------------------
# Procedural memory: read agent prompt supplements
# ---------------------------------------------------------

async def get_agent_prompt_supplement(store, url_hash: str, agent_name: str) -> str:
    """Read learned prompt additions for an agent, formatted for injection into system prompt."""
    ns = ("procedures", url_hash, "agent_prompts")
    existing = await store.aget(ns, agent_name)
    if not existing:
        return ""

    data = existing.value

    if "optimized_prompt" in data:
        return f"LEARNED FROM PAST SESSIONS:\n{data['optimized_prompt']}"

    sections: List[str] = []

    learned = data.get("learned_additions", [])
    if learned:
        sections.append("Key observations:\n" + "\n".join(f"- {item}" for item in learned[-8:]))

    effective = data.get("effective_strategies", [])
    if effective:
        sections.append("Effective strategies:\n" + "\n".join(f"- {item}" for item in effective[-5:]))

    avoid = data.get("avoid_strategies", [])
    if avoid:
        sections.append("Avoid:\n" + "\n".join(f"- {item}" for item in avoid[-5:]))

    if not sections:
        return ""
    return "LEARNED FROM PAST SESSIONS:\n" + "\n".join(sections)


async def get_routing_rules_supplement(store, url_hash: str) -> str:
    """Read learned routing rules for the supervisor."""
    ns = ("procedures", url_hash, "routing_rules")
    existing = await store.aget(ns, "current")
    if not existing:
        return ""

    data = existing.value

    if "optimized_prompt" in data:
        return f"LEARNED ROUTING RULES:\n{data['optimized_prompt']}"

    rules = data.get("rules", [])
    if not rules:
        return ""
    return "LEARNED ROUTING RULES:\n" + "\n".join(f"- {r}" for r in rules[-8:])


# ---------------------------------------------------------
# Procedural memory: write via LLM reflection (post-batch)
# ---------------------------------------------------------

def _format_agent_knowledge(data: Dict[str, Any]) -> str:
    """Serialize stored agent knowledge into a prompt string for the optimizer."""
    if "optimized_prompt" in data:
        return data["optimized_prompt"]
    parts: List[str] = []
    for key, label in [
        ("learned_additions", "Key observations"),
        ("effective_strategies", "Effective strategies"),
        ("avoid_strategies", "Avoid"),
    ]:
        items = data.get(key, [])
        if items:
            parts.append(f"{label}:\n" + "\n".join(f"- {i}" for i in items))
    return "\n".join(parts) if parts else ""


def _format_routing_rules(data: Dict[str, Any]) -> str:
    """Serialize stored routing rules into a prompt string for the optimizer."""
    if "optimized_prompt" in data:
        return data["optimized_prompt"]
    rules = data.get("rules", [])
    if rules:
        return "\n".join(f"- {r}" for r in rules)
    return ""


async def update_procedural_memory(store, url_hash: str, llm) -> None:
    """Reflect on recent sessions and update agent prompt supplements and routing rules.

    Uses Langmem's prompt optimizer for structured reflection on episodic memory.
    Called once after all missions in a batch complete.
    """
    from langmem import create_prompt_optimizer

    sessions = await store.asearch(("episodes", url_hash, "sessions"), limit=15)
    if not sessions:
        return

    trajectory_messages: List[Dict[str, str]] = []
    for s in sessions:
        v = s.value
        trajectory_messages.append({
            "role": "user",
            "content": f"Mission: {v.get('mission_prompt_summary', '?')}",
        })
        trajectory_messages.append({
            "role": "assistant",
            "content": (
                f"Completed with {v.get('total_actions', 0)} actions, "
                f"{v.get('bugs_found', 0)} bugs found. "
                f"Outcome: {v.get('outcome', '?')}. "
                f"Pages covered: {v.get('pages_covered', [])[:5]}"
            ),
        })

    optimizer = create_prompt_optimizer(llm, kind="prompt_memory")

    prompts_ns = ("procedures", url_hash, "agent_prompts")
    agent_names: set = set()
    for s in sessions:
        tid = s.value.get("thread_id", "")
        agent_names.add(tid)

    for agent_name in agent_names:
        existing = await store.aget(prompts_ns, agent_name)
        current_text = _format_agent_knowledge(existing.value) if existing else ""
        base_prompt = (
            f"You are {agent_name}, a QA testing agent. "
            "Your goal is to test web applications effectively.\n"
        )
        if current_text:
            base_prompt += f"\n{current_text}"

        try:
            optimized = await optimizer.ainvoke({
                "trajectories": [(trajectory_messages, {"context": "batch review"})],
                "prompt": base_prompt,
            })
            optimized_text = optimized if isinstance(optimized, str) else str(optimized)
            version = (existing.value.get("version", 0) + 1) if existing else 1
            await store.aput(prompts_ns, agent_name, {
                "agent_name": agent_name,
                "optimized_prompt": optimized_text,
                "version": version,
            })
        except Exception:
            pass

    rules_ns = ("procedures", url_hash, "routing_rules")
    existing_rules = await store.aget(rules_ns, "current")
    current_rules_text = _format_routing_rules(existing_rules.value) if existing_rules else ""
    base_rules = (
        "You are the QA Orchestrator routing supervisor. "
        "Decide which agent to route to based on mission progress and page areas.\n"
    )
    if current_rules_text:
        base_rules += f"\n{current_rules_text}"

    try:
        optimized_rules = await optimizer.ainvoke({
            "trajectories": [(trajectory_messages, {"context": "routing review"})],
            "prompt": base_rules,
        })
        optimized_text = optimized_rules if isinstance(optimized_rules, str) else str(optimized_rules)
        version = (existing_rules.value.get("version", 0) + 1) if existing_rules else 1
        await store.aput(rules_ns, "current", {
            "optimized_prompt": optimized_text,
            "version": version,
        })
    except Exception:
        pass


# ---------------------------------------------------------
# Regression testing: auto-generate missions from bug catalog
# ---------------------------------------------------------

async def generate_regression_missions(store, url_hash: str) -> List[Dict[str, Any]]:
    """Generate test missions targeting known open bugs and historically flaky areas.

    Returns a list of mission dicts compatible with the missions YAML format.
    """
    bugs = await store.asearch(("episodes", url_hash, "bugs"), limit=30)
    open_bugs = [
        b for b in bugs
        if b.value.get("status", "open") == "open" or b.value.get("seen_count", 1) > 1
    ]

    if not open_bugs:
        return []

    pages_with_bugs: Dict[str, List[str]] = {}
    for bug in open_bugs:
        page = bug.value.get("page", "/")
        summary = bug.value.get("summary", "unknown bug")[:150]
        pages_with_bugs.setdefault(page, []).append(summary)

    missions: List[Dict[str, Any]] = []
    for i, (page, bug_summaries) in enumerate(pages_with_bugs.items()):
        bug_list = "; ".join(bug_summaries[:3])
        mission = {
            "thread_id": f"regression_{i+1:02d}_{page.strip('/').replace('/', '_') or 'root'}",
            "prompt": (
                f"Regression test for page '{page}'. Previously discovered bugs in this area: {bug_list}. "
                "Verify whether these bugs still exist and look for related regressions. "
                "Use recall_past_findings to get full context before testing."
            ),
        }
        missions.append(mission)

    return missions


# ---------------------------------------------------------
# Application model export
# ---------------------------------------------------------

async def export_app_model(store, url_hash: str) -> Dict[str, Any]:
    """Export discovered application structure from the store as a structured dict.

    Returns a dict containing pages, selectors, quirks, bugs, and session stats.
    """
    import json

    model: Dict[str, Any] = {"url_hash": url_hash}

    pages = await store.asearch(("app", url_hash, "pages"), limit=50)
    if pages:
        model["pages"] = sorted(
            [p.value for p in pages],
            key=lambda v: v.get("visit_count", 0),
            reverse=True,
        )

    selectors = await store.asearch(("app", url_hash, "selectors"), limit=50)
    if selectors:
        sel_list = []
        for s in selectors:
            v = s.value
            total = v.get("success_count", 0) + v.get("failure_count", 0)
            reliability = v.get("success_count", 0) / max(total, 1)
            sel_list.append({**v, "reliability": round(reliability, 2), "total_uses": total})
        model["selectors"] = sorted(sel_list, key=lambda v: v.get("total_uses", 0), reverse=True)

    quirks = await store.asearch(("app", url_hash, "quirks"), limit=30)
    if quirks:
        model["quirks"] = [q.value for q in quirks]

    bugs = await store.asearch(("episodes", url_hash, "bugs"), limit=30)
    if bugs:
        model["bugs"] = sorted(
            [b.value for b in bugs],
            key=lambda v: v.get("seen_count", 0),
            reverse=True,
        )

    sessions = await store.asearch(("episodes", url_hash, "sessions"), limit=30)
    if sessions:
        model["session_stats"] = {
            "total_sessions": len(sessions),
            "total_actions": sum(s.value.get("total_actions", 0) for s in sessions),
            "total_bugs": sum(s.value.get("bugs_found", 0) for s in sessions),
            "pages_tested": list(dict.fromkeys(
                p for s in sessions for p in s.value.get("pages_covered", [])
            ))[:30],
        }

    return model


# ---------------------------------------------------------
# Test prioritization
# ---------------------------------------------------------

async def prioritize_pages(store, url_hash: str) -> str:
    """Score pages by risk and return a prioritized list for the supervisor.

    Scoring factors:
    - Bug density: pages with more bugs score higher
    - Selector flakiness: pages with unreliable selectors score higher
    - Recency: pages not tested recently score higher
    """
    page_scores: Dict[str, float] = {}

    bugs = await store.asearch(("episodes", url_hash, "bugs"), limit=30)
    for b in bugs:
        page = b.value.get("page", "/")
        page_scores[page] = page_scores.get(page, 0) + b.value.get("seen_count", 1) * 2

    quirks = await store.asearch(("app", url_hash, "quirks"), limit=20)
    for q in quirks:
        page = q.value.get("page", "/")
        page_scores[page] = page_scores.get(page, 0) + q.value.get("confirmed_count", 1)

    selectors = await store.asearch(("app", url_hash, "selectors"), limit=30)
    for s in selectors:
        v = s.value
        failures = v.get("failure_count", 0)
        if failures > 0:
            page = v.get("page_pattern", "/")
            page_scores[page] = page_scores.get(page, 0) + failures * 1.5

    if not page_scores:
        return ""

    ranked = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    lines = [f"- {page} (risk score: {score:.0f})" for page, score in ranked]
    return "HIGH_RISK_PAGES (prioritize testing these areas):\n" + "\n".join(lines)
