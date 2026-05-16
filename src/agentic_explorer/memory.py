"""Cross-session memory for the agentic test explorer.

Provides utilities for semantic, episodic, and procedural memory backed by
a LangGraph Store (AsyncSqliteStore or compatible).

Namespace layout:
  ("app", "{app_url_hash}", "pages")          — page structure knowledge
  ("app", "{app_url_hash}", "selectors")      — selector reliability tracking
  ("app", "{app_url_hash}", "quirks")         — application-specific behaviors
  ("episodes", "{app_url_hash}", "sessions")  — completed session summaries
  ("episodes", "{app_url_hash}", "bugs")      — deduplicated bug catalog
  ("procedures", "{app_url_hash}", "agent_prompts")  — per-agent prompt supplements
  ("procedures", "{app_url_hash}", "routing_rules")  — supervisor routing refinements
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

async def format_memory_context(store, url_hash: str) -> str:
    """Build a MEMORY_CONTEXT section for the supervisor routing prompt."""
    sections: List[str] = []

    pages = await store.asearch(("app", url_hash, "pages"), limit=15)
    if pages:
        page_lines = []
        for item in sorted(pages, key=lambda p: p.value.get("visit_count", 0), reverse=True):
            v = item.value
            page_lines.append(f"- {v.get('url', '?')} (visited {v.get('visit_count', 0)}x)")
        sections.append("KNOWN_PAGES:\n" + "\n".join(page_lines[:10]))

    quirks = await store.asearch(("app", url_hash, "quirks"), limit=8)
    if quirks:
        quirk_lines = [f"- {q.value.get('description', '?')[:200]}" for q in quirks]
        sections.append("KNOWN_QUIRKS:\n" + "\n".join(quirk_lines))

    bugs = await store.asearch(("episodes", url_hash, "bugs"), limit=6)
    if bugs:
        bug_lines = [f"- {b.value.get('summary', '?')[:200]} (seen {b.value.get('seen_count', 1)}x)" for b in bugs]
        sections.append("KNOWN_BUGS_FROM_PAST_SESSIONS:\n" + "\n".join(bug_lines))

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
    """Return a LangChain tool that agents can call to recall past findings for a page area."""
    from langchain_core.tools import tool

    @tool
    async def recall_past_findings(page_url_pattern: str) -> str:
        """Recall bugs and testing outcomes from past sessions for a page area.

        Args:
            page_url_pattern: URL path or keyword to match (e.g. "/systems", "login", "/home").
        """
        pattern = page_url_pattern.strip().lower()
        results: List[str] = []

        bugs = await store.asearch(("episodes", url_hash, "bugs"), limit=20)
        matched_bugs = [
            b for b in bugs
            if pattern in (b.value.get("page", "").lower())
            or pattern in (b.value.get("summary", "").lower())
        ]
        if matched_bugs:
            results.append(f"KNOWN BUGS ({len(matched_bugs)}):")
            for b in matched_bugs[:8]:
                v = b.value
                results.append(f"  - {v.get('summary', '?')[:200]} (seen {v.get('seen_count', 1)}x, status: {v.get('status', 'open')})")

        sessions = await store.asearch(("episodes", url_hash, "sessions"), limit=20)
        matched_sessions = [
            s for s in sessions
            if any(pattern in p.lower() for p in s.value.get("pages_covered", []))
            or pattern in s.value.get("mission_prompt_summary", "").lower()
        ]
        if matched_sessions:
            results.append(f"\nPAST SESSIONS covering this area ({len(matched_sessions)}):")
            for s in matched_sessions[:5]:
                v = s.value
                results.append(
                    f"  - {v.get('thread_id', '?')}: {v.get('total_actions', 0)} actions, "
                    f"{v.get('bugs_found', 0)} bugs, outcome={v.get('outcome', '?')}"
                )

        quirks = await store.asearch(("app", url_hash, "quirks"), limit=15)
        matched_quirks = [
            q for q in quirks
            if pattern in q.value.get("page", "").lower()
            or pattern in q.value.get("description", "").lower()
        ]
        if matched_quirks:
            results.append(f"\nKNOWN QUIRKS ({len(matched_quirks)}):")
            for q in matched_quirks[:5]:
                v = q.value
                results.append(f"  - {v.get('description', '?')[:200]} (confirmed {v.get('confirmed_count', 1)}x)")

        if not results:
            return f"No past findings found for '{page_url_pattern}'. This area may not have been tested before."

        return "\n".join(results)

    return recall_past_findings


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

    rules = existing.value.get("rules", [])
    if not rules:
        return ""
    return "LEARNED ROUTING RULES:\n" + "\n".join(f"- {r}" for r in rules[-8:])


# ---------------------------------------------------------
# Procedural memory: write via LLM reflection (post-batch)
# ---------------------------------------------------------

_REFLECTION_PROMPT = """\
You are a QA process improvement analyst. Based on the testing session summaries below, \
extract actionable lessons to improve future test sessions.

SESSION SUMMARIES:
{sessions}

CURRENT AGENT KNOWLEDGE:
{current_knowledge}

For EACH agent that appeared in these sessions, provide:
1. learned_additions: Key facts about the application discovered during testing (max 5)
2. effective_strategies: Testing approaches that found bugs or achieved good coverage (max 3)
3. avoid_strategies: Approaches that wasted time or hit dead ends (max 3)

Also provide up to 5 routing_rules for the supervisor (e.g., which agent to prefer for which area).

Respond in this exact JSON format:
{{
  "agents": {{
    "agent_name": {{
      "learned_additions": ["..."],
      "effective_strategies": ["..."],
      "avoid_strategies": ["..."]
    }}
  }},
  "routing_rules": ["..."]
}}
"""


async def update_procedural_memory(store, url_hash: str, llm) -> None:
    """Reflect on recent sessions and update agent prompt supplements and routing rules.

    Uses LLM reflection on episodic memory to generate procedural improvements.
    Called once after all missions in a batch complete.
    """
    import json

    sessions = await store.asearch(("episodes", url_hash, "sessions"), limit=15)
    if not sessions:
        return

    session_text = "\n".join(
        f"- {s.value.get('thread_id', '?')}: {s.value.get('total_actions', 0)} actions, "
        f"{s.value.get('bugs_found', 0)} bugs, outcome={s.value.get('outcome', '?')}, "
        f"pages={s.value.get('pages_covered', [])[:5]}"
        for s in sessions
    )

    agent_names = set()
    for s in sessions:
        tid = s.value.get("thread_id", "")
        for part in tid.split("_"):
            if part in ("agent",):
                continue
        agent_names.add(tid)

    current_parts: List[str] = []
    prompts_ns = ("procedures", url_hash, "agent_prompts")
    for s in sessions:
        tid = s.value.get("thread_id", "")
        existing = await store.aget(prompts_ns, tid)
        if existing:
            current_parts.append(f"{tid}: {json.dumps(existing.value, default=str)[:300]}")

    rules_existing = await store.aget(("procedures", url_hash, "routing_rules"), "current")
    if rules_existing:
        current_parts.append(f"routing_rules: {json.dumps(rules_existing.value, default=str)[:300]}")

    current_knowledge = "\n".join(current_parts) if current_parts else "No prior procedural knowledge."

    from langchain_core.messages import HumanMessage as HMsg
    prompt = _REFLECTION_PROMPT.format(sessions=session_text, current_knowledge=current_knowledge)
    response = await llm.ainvoke([HMsg(content=prompt)])

    response_text = response.content
    if isinstance(response_text, list):
        response_text = "".join(b.get("text", "") for b in response_text if isinstance(b, dict))

    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if not json_match:
        return

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError:
        return

    agents_data = result.get("agents", {})
    for agent_name, updates in agents_data.items():
        if not isinstance(updates, dict):
            continue
        existing = await store.aget(prompts_ns, agent_name)
        if existing:
            data = existing.value
            for key in ("learned_additions", "effective_strategies", "avoid_strategies"):
                new_items = updates.get(key, [])
                if isinstance(new_items, list):
                    old_items = data.get(key, [])
                    merged = list(dict.fromkeys(old_items + new_items))[-10:]
                    data[key] = merged
        else:
            data = {
                "agent_name": agent_name,
                "learned_additions": (updates.get("learned_additions") or [])[:10],
                "effective_strategies": (updates.get("effective_strategies") or [])[:5],
                "avoid_strategies": (updates.get("avoid_strategies") or [])[:5],
            }
        await store.aput(prompts_ns, agent_name, data)

    new_rules = result.get("routing_rules", [])
    if isinstance(new_rules, list) and new_rules:
        rules_ns = ("procedures", url_hash, "routing_rules")
        existing_rules = await store.aget(rules_ns, "current")
        if existing_rules:
            old_rules = existing_rules.value.get("rules", [])
            version = existing_rules.value.get("version", 0) + 1
            merged_rules = list(dict.fromkeys(old_rules + new_rules))[-10:]
        else:
            version = 1
            merged_rules = new_rules[:10]
        await store.aput(rules_ns, "current", {"version": version, "rules": merged_rules})


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
