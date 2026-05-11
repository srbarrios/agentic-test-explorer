"""
Record-and-Translate Browser Engine
-----------------------------------
Separates the AI's *brain* (LangGraph agents emitting JSON intents) from its
execution *hands* (deterministic Playwright engine).

Workflow:
  1. Agent outputs a strict JSON command via `execute_browser_command`.
  2. This engine parses + validates the command, executes it with Playwright,
     captures a DOM snapshot, and appends an immutable record to the
     per-thread "Action Tape" (in-memory + `report_<thread>/action_tape.jsonl`).
  3. When a bug is detected, `generate_playwright_spec` translates the tape
     into a reproducible `.spec.ts` file.

Supported actions (JSON `action` field):
    navigate      {"url": str}
    click         {"selector": str}
    fill          {"selector": str, "value": str}
    press         {"selector": str, "key": str}
    select_option {"selector": str, "value": str}
    hover         {"selector": str}
    wait_for      {"selector": str, "state"?: "visible"|"hidden"|"attached"}
    scroll        {"selector"?: str, "y"?: int}
    extract_text  {"selector": str}
    snapshot      {}    # DOM-only, no side effect
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from playwright.async_api import Page

from agentic_explorer.utils import console

# ---------------------------------------------------------
# Per-thread Action Tape store
# ---------------------------------------------------------
_ACTION_TAPES: Dict[str, List[Dict[str, Any]]] = {}


def get_action_tape(thread_id: str) -> List[Dict[str, Any]]:
    return _ACTION_TAPES.setdefault(thread_id, [])


def _tape_path(thread_id: str) -> str:
    os.makedirs(f"report_{thread_id}", exist_ok=True)
    return f"report_{thread_id}/action_tape.jsonl"


def _append_tape(thread_id: str, entry: Dict[str, Any]) -> None:
    get_action_tape(thread_id).append(entry)
    try:
        with open(_tape_path(thread_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - best-effort logging
        console.warn(f"Failed to persist tape entry: {exc}")


# ---------------------------------------------------------
# DOM snapshot extraction
# ---------------------------------------------------------
# Strategy: prefer Playwright's native Accessibility Tree (already strips
# scripts/styles/SVGs and reflects only perceivable/interactable nodes).
# Fall back to a pruned JS DOM walk that explicitly skips non-content tags.

# Tags that are never useful to an agent — skipped during the JS walk.
_SKIP_TAGS = frozenset([
    "script", "style", "svg", "noscript", "head", "meta", "link",
    "template", "path", "defs", "g", "symbol", "use", "clippath",
])

_DOM_SNAPSHOT_JS = r"""
(args) => {
  const limit = args.limit;
  const skipTags = new Set(args.skipTags);
  const INTERACTIVE = new Set([
    'a','button','input','select','textarea','summary','option','label'
  ]);
  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return false;
    const cs = window.getComputedStyle(el);
    return cs && cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0';
  };
  const cssPath = (el) => {
    if (!(el instanceof Element)) return '';
    const parts = [];
    while (el && el.nodeType === 1 && parts.length < 6) {
      let part = el.nodeName.toLowerCase();
      if (el.id) { part += '#' + el.id; parts.unshift(part); break; }
      const cls = (el.getAttribute('class') || '').trim().split(/\s+/).filter(Boolean).slice(0,2).join('.');
      if (cls) part += '.' + cls;
      const parent = el.parentElement;
      if (parent) {
        const siblings = Array.from(parent.children).filter(c => c.nodeName === el.nodeName);
        if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(el)+1})`;
      }
      parts.unshift(part);
      el = el.parentElement;
    }
    return parts.join(' > ');
  };
  const out = [];
  const all = document.querySelectorAll('*');
  for (const el of all) {
    if (out.length >= limit) break;
    const tag = el.tagName.toLowerCase();
    // Skip non-content tags early — no reason to feed these to the agent.
    if (skipTags.has(tag)) continue;
    const role = el.getAttribute('role');
    const testSubj = el.getAttribute('data-test-subj');
    const isBtnLike = INTERACTIVE.has(tag) || role === 'button' || role === 'link' || role === 'tab' || role === 'menuitem';
    const textRaw = (el.innerText || el.textContent || '').trim().replace(/\s+/g, ' ');
    const hasText = textRaw.length > 0 && textRaw.length < 200;
    if (!isBtnLike && !testSubj && !hasText) continue;
    if (!isVisible(el)) continue;
    out.push({
      tag,
      role: role || null,
      id: el.id || null,
      name: el.getAttribute('name') || el.getAttribute('aria-label') || null,
      testSubj: testSubj || null,
      text: hasText ? textRaw.slice(0, 120) : null,
      selector: cssPath(el),
      interactive: isBtnLike
    });
  }
  return {
    url: location.href,
    title: document.title,
    elements: out
  };
}
"""


def _flatten_ax_tree(
    node: Dict[str, Any],
    out: List[Dict[str, Any]],
    limit: int,
    depth: int = 0,
) -> None:
    """Recursively flatten Playwright's accessibility tree into a list."""
    if len(out) >= limit or not node:
        return
    role = node.get("role", "")
    name = (node.get("name") or "").strip()
    value = (node.get("value") or "")
    # Skip purely structural or invisible nodes with no useful label
    if role in ("none", "presentation", "generic") and not name:
        for child in node.get("children") or []:
            _flatten_ax_tree(child, out, limit, depth + 1)
        return
    out.append({
        "role": role,
        "name": name[:120] if name else None,
        "value": str(value)[:80] if value else None,
        "depth": depth,
    })
    for child in node.get("children") or []:
        _flatten_ax_tree(child, out, limit, depth + 1)


async def extract_dom_snapshot(page: Page, limit: int = 150) -> Dict[str, Any]:
    """Return a compact DOM digest usable by an LLM.

    Primary path: Playwright's Accessibility Tree — already excludes scripts,
    styles, SVG internals, and hidden nodes.  This is the smallest faithful
    representation of what a user can see and interact with.

    Fallback: a JS DOM walk that explicitly skips non-content tags.
    """
    url = page.url if page else ""
    title = ""
    try:
        title = await page.title()
    except Exception:
        pass

    # --- Primary: Accessibility Tree ---
    try:
        ax_root = await page.accessibility.snapshot(interesting_only=True)
        if ax_root:
            elements: List[Dict[str, Any]] = []
            _flatten_ax_tree(ax_root, elements, limit)
            return {
                "url": url,
                "title": title,
                "source": "accessibility_tree",
                "elements": elements,
            }
    except Exception:
        pass  # fall through to JS DOM walk

    # --- Fallback: pruned JS DOM walk ---
    try:
        snap = await page.evaluate(
            _DOM_SNAPSHOT_JS,
            {"limit": limit, "skipTags": list(_SKIP_TAGS)},
        )
        snap["source"] = "dom_walk"
        return snap
    except Exception as exc:
        return {"url": url, "title": title, "source": "error", "elements": [], "error": str(exc)}


def _format_snapshot_for_llm(snap: Dict[str, Any], max_elems: int = 80) -> str:
    lines = [
        f"URL: {snap.get('url', '')}",
        f"TITLE: {snap.get('title', '')}",
        f"SOURCE: {snap.get('source', 'unknown')}",
    ]
    if snap.get("error"):
        lines.append(f"SNAPSHOT_ERROR: {snap['error']}")
    elements = snap.get("elements", [])[:max_elems]
    source = snap.get("source", "")
    for i, e in enumerate(elements):
        if source == "accessibility_tree":
            indent = "  " * e.get("depth", 0)
            role = e.get("role", "")
            name = e.get("name") or ""
            value = e.get("value") or ""
            interactive = role in (
                "button", "link", "textbox", "checkbox", "radio",
                "combobox", "listbox", "menuitem", "tab", "switch",
            )
            marker = "*" if interactive else "-"
            val_part = f" value={value!r}" if value else ""
            lines.append(f"{indent}{marker} [{i}] role={role} name={name!r}{val_part}")
        else:
            marker = "*" if e.get("interactive") else "-"
            label = e.get("testSubj") or e.get("name") or e.get("id") or e.get("text") or ""
            lines.append(
                f"{marker} [{i}] <{e.get('tag', '')}> role={e.get('role')} "
                f"sel={e.get('selector')!r} label={label!r}"
            )
    if len(snap.get("elements", [])) > max_elems:
        lines.append(f"... (+{len(snap['elements']) - max_elems} more elements truncated)")
    return "\n".join(lines)


# ---------------------------------------------------------
# Command execution
# ---------------------------------------------------------
ALLOWED_ACTIONS = {
    "navigate", "click", "fill", "press", "select_option", "hover",
    "wait_for", "scroll", "extract_text", "snapshot", "check_page_health",
}

# JS that scans the live DOM for visible error banners, alerts, and active spinners.
_HEALTH_CHECK_JS = """
() => {
  const ERROR_SELECTORS = [
    '[class*="euiCallOut--danger"]',
    '[class*="toast"][class*="danger"]',
    '[role="alert"]',
    '[data-test-subj*="errorMessage"]',
    '[class*="error-message"]',
    '[class*="errorMessage"]',
  ];
  const SPINNER_SELECTORS = [
    '[aria-busy="true"]',
    '[class*="euiLoadingSpinner"]',
    '[class*="loading-spinner"]',
    '[class*="loadingSpinner"]',
  ];
  const isVisible = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return false;
    const cs = window.getComputedStyle(el);
    return cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0';
  };
  const errors = [];
  for (const sel of ERROR_SELECTORS) {
    try {
      document.querySelectorAll(sel).forEach(el => {
        if (isVisible(el)) {
          const t = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');
          if (t) errors.push(t.slice(0, 300));
        }
      });
    } catch (_) {}
  }
  let spinnerCount = 0;
  for (const sel of SPINNER_SELECTORS) {
    try { spinnerCount += document.querySelectorAll(sel).length; } catch (_) {}
  }
  return { errors: errors.slice(0, 5), spinners: spinnerCount, url: location.href, title: document.title };
}
"""

# ---------------------------------------------------------
# Selector resilience guard
# ---------------------------------------------------------
# Patterns that indicate a brittle, position-dependent selector.
_BRITTLE_SELECTOR_PATTERNS = re.compile(
    r"""
    (^/)                           # XPath starting with /
    | (/{2})                       # XPath descendant //
    | (:nth-child\s*\()            # :nth-child pseudo
    | (:nth-of-type\s*\()          # :nth-of-type pseudo — structural position
    | (>\s*(?:div|span|li|ul|ol|td|tr|th)[\s>+~,\[{$])  # bare positional div/span chains
    """,
    re.VERBOSE,
)

_RESILIENT_SELECTOR_HINTS = (
    "Use one of:\n"
    "  1. data-test-subj  → [data-test-subj='myButton']\n"
    "  2. aria-label      → [aria-label='Search bar']\n"
    "  3. Visible text    → button:has-text('Save'), text='Apply'\n"
    "  4. Semantic role   → role=button[name='Submit']\n"
    "Call get_dom_snapshot first to find a stable attribute."
)


def _validate_selector(selector: str) -> Optional[str]:
    """Return an error message if the selector is brittle, else None."""
    if not selector:
        return None  # empty / navigate — no selector to check
    if _BRITTLE_SELECTOR_PATTERNS.search(selector):
        return (
            f"BRITTLE_SELECTOR_REJECTED: '{selector}' encodes DOM position or uses XPath "
            f"and will break on minor viewport changes.\n{_RESILIENT_SELECTOR_HINTS}"
        )
    return None


@dataclass
class ExecutionResult:
    ok: bool
    action: str
    params: Dict[str, Any]
    started_at: float
    duration_ms: int
    result: Optional[str] = None
    error: Optional[str] = None
    snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_tape_entry(self) -> Dict[str, Any]:
        return {
            "ts": self.started_at,
            "action": self.action,
            "params": self.params,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "result": self.result,
            "error": self.error,
            "page_url": self.snapshot.get("url"),
            "page_title": self.snapshot.get("title"),
        }


async def _dispatch(page: Page, action: str, params: Dict[str, Any]) -> str:
    if action == "navigate":
        await page.goto(params["url"])
        return f"navigated to {params['url']}"
    if action == "click":
        await page.click(params["selector"])
        return f"clicked {params['selector']}"
    if action == "fill":
        await page.fill(params["selector"], params.get("value", ""))
        return f"filled {params['selector']}"
    if action == "press":
        await page.press(params["selector"], params["key"])
        return f"pressed {params['key']} on {params['selector']}"
    if action == "select_option":
        await page.select_option(params["selector"], params["value"])
        return f"selected {params['value']} on {params['selector']}"
    if action == "hover":
        await page.hover(params["selector"])
        return f"hovered {params['selector']}"
    if action == "wait_for":
        await page.wait_for_selector(params["selector"], state=params.get("state", "visible"))
        return f"waited for {params['selector']}"
    if action == "scroll":
        if "selector" in params and params["selector"]:
            await page.eval_on_selector(
                params["selector"], "el => el.scrollIntoView({block:'center'})"
            )
            return f"scrolled to {params['selector']}"
        y = int(params.get("y", 400))
        await page.evaluate("(y)=>window.scrollBy(0,y)", y)
        return f"scrolled window by {y}px"
    if action == "extract_text":
        txt = await page.locator(params["selector"]).first.inner_text()
        return (txt or "").strip()[:4000]
    if action == "snapshot":
        return "snapshot"
    if action == "check_page_health":
        health = await page.evaluate(_HEALTH_CHECK_JS)
        spinners = health.get("spinners", 0)
        errors = health.get("errors", [])
        if errors:
            return f"HEALTH: ERROR — {len(errors)} error banner(s): {'; '.join(errors)}"
        if spinners:
            return f"HEALTH: LOADING — {spinners} active spinner(s) (may indicate a stuck request)"
        return "HEALTH: OK — no visible error banners or spinners"
    raise ValueError(f"Unsupported action: {action}")


# ---------------------------------------------------------
# Tool factories (page- and thread-aware via RunnableConfig)
# ---------------------------------------------------------
def get_browser_command_tool(page: Page):
    """
    LangChain tool: executes a single strict JSON browser command, records it
    to the immutable Action Tape, and returns a DOM snapshot + result.
    """
    @tool
    async def execute_browser_command(command_json: str, config: RunnableConfig) -> str:
        """Execute ONE browser action described as strict JSON and return the new DOM snapshot.

        The agent is the *brain*: it only emits intents. The engine is the *hands*.

        Argument `command_json` MUST be a JSON object string, for example:
            {"action": "navigate", "url": "https://example.com/app"}
            {"action": "click", "selector": "[data-test-subj='submitButton']"}
            {"action": "fill", "selector": "input[aria-label='Search']", "value": "hello"}
            {"action": "wait_for", "selector": ".main-content"}
            {"action": "extract_text", "selector": "h1"}
            {"action": "snapshot"}

        Allowed actions: navigate, click, fill, press, select_option, hover,
        wait_for, scroll, extract_text, snapshot.

        The return value is a human-readable block with:
          STATUS, RESULT or ERROR, and a DOM_SNAPSHOT digest.
        Every invocation is appended to the per-thread Action Tape, which is
        later translated into a Playwright .spec.ts script for reproduction.
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        started = time.time()
        try:
            parsed_command = json.loads(command_json) if isinstance(command_json, str) else command_json
            if not isinstance(parsed_command, Mapping):
                raise ValueError("JSON command must be an object")
            cmd = dict(parsed_command)
        except Exception as exc:
            return f"STATUS: ERROR\nERROR: Invalid JSON ({exc}). Expected object with 'action'."
        action = (cmd.get("action") or "").strip()
        if action not in ALLOWED_ACTIONS:
            return (
                f"STATUS: ERROR\nERROR: Unknown action '{action}'. "
                f"Allowed: {sorted(ALLOWED_ACTIONS)}"
            )
        params = {k: v for k, v in cmd.items() if k != "action"}

        # Enforce resilient selector policy before touching the browser.
        selector_error = _validate_selector(params.get("selector", ""))
        if selector_error:
            return f"STATUS: ERROR\nERROR: {selector_error}"

        err: Optional[str] = None
        res: Optional[str] = None
        try:
            res = await _dispatch(page, action, params)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"

        snap = await extract_dom_snapshot(page)
        duration_ms = int((time.time() - started) * 1000)
        result = ExecutionResult(
            ok=err is None, action=action, params=params,
            started_at=started, duration_ms=duration_ms,
            result=res, error=err, snapshot=snap,
        )
        _append_tape(thread_id, result.to_tape_entry())

        status = "OK" if result.ok else "ERROR"
        body = [
            f"STATUS: {status}",
            f"ACTION: {action} {json.dumps(params, ensure_ascii=False)}",
            f"DURATION_MS: {duration_ms}",
        ]
        if result.ok:
            body.append(f"RESULT: {res}")
        else:
            body.append(f"ERROR: {err}  (Recoverable — try a different selector or action.)")
        body.append("DOM_SNAPSHOT:")
        body.append(_format_snapshot_for_llm(snap))
        return "\n".join(body)

    return execute_browser_command


def get_dom_snapshot_tool(page: Page):
    """Read-only DOM digest tool (does NOT record to the tape)."""
    @tool
    async def get_dom_snapshot() -> str:
        """Return a compact DOM snapshot of the current page (URL, title, interactive
        elements with selectors). Use this to *see* before emitting a command."""
        snap = await extract_dom_snapshot(page)
        return _format_snapshot_for_llm(snap, max_elems=80)
    return get_dom_snapshot


# ---------------------------------------------------------
# Code Generator: Action Tape -> Playwright .spec.ts
# ---------------------------------------------------------
def _ts_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")


def _tape_entry_to_ts(entry: Dict[str, Any]) -> Optional[str]:
    prefix = ""
    if not entry.get("ok"):
        err = str(entry.get("error", "Unknown Error"))
        err_multiline_safe = err.replace("\n", "\n  //   ")
        prefix = f"  // FAILED AT RECORD TIME (kept to reproduce error): {entry.get('action')} {json.dumps(entry.get('params'))}\n  //   Error: {err_multiline_safe}\n"
    a = entry["action"]
    p = entry.get("params") or {}
    if a == "navigate":
        return prefix + f"  await page.goto('{_ts_escape(p['url'])}');"
    if a == "click":
        return prefix + f"  await page.click('{_ts_escape(p['selector'])}');"
    if a == "fill":
        return prefix + f"  await page.fill('{_ts_escape(p['selector'])}', '{_ts_escape(str(p.get('value','')))}');"
    if a == "press":
        return prefix + f"  await page.press('{_ts_escape(p['selector'])}', '{_ts_escape(p['key'])}');"
    if a == "select_option":
        return prefix + f"  await page.selectOption('{_ts_escape(p['selector'])}', '{_ts_escape(str(p['value']))}');"
    if a == "hover":
        return prefix + f"  await page.hover('{_ts_escape(p['selector'])}');"
    if a == "wait_for":
        state = p.get("state", "visible")
        return prefix + f"  await page.waitForSelector('{_ts_escape(p['selector'])}', {{ state: '{state}' }});"
    if a == "scroll":
        if p.get("selector"):
            return prefix + (
                f"  await page.locator('{_ts_escape(p['selector'])}')"
                f".scrollIntoViewIfNeeded();"
            )
        return prefix + f"  await page.evaluate(() => window.scrollBy(0, {int(p.get('y', 400))}));"
    if a == "extract_text":
        return prefix + (
            f"  const _t = await page.locator('{_ts_escape(p['selector'])}').first().innerText();\n"
            f"  expect(_t.length).toBeGreaterThan(0);"
        )
    if a == "snapshot":
        return prefix + "  // snapshot (no-op in replay)"
    if a == "check_page_health":
        return prefix + "  // check_page_health (no-op in replay — health was verified at record time)"
    return prefix + f"  // unsupported action replay: {a}"


def generate_playwright_spec(
    thread_id: str,
    bug_summary: str,
    app_url: Optional[str] = None,
    storage_state_path: str = "auth.json",
) -> str:
    """Translate the recorded Action Tape into a runnable Playwright spec file.
    Returns the absolute path of the generated .spec.ts file."""
    tape = get_action_tape(thread_id)
    bug_summary_clean = bug_summary.replace("\n", " ").replace("*/", "* /")
    safe = re.sub(r"[^a-zA-Z0-9]", "_", bug_summary_clean)[:40].strip("_") or "bug"
    out_dir = f"report_{thread_id}"
    os.makedirs(out_dir, exist_ok=True)
    spec_path = os.path.join(out_dir, f"reproduction_{safe}_{int(time.time())}.spec.ts")

    steps: List[str] = []
    for entry in tape:
        line = _tape_entry_to_ts(entry)
        if line:
            steps.append(line)

    header = [
        "/**",
        f" * Auto-generated reproduction for bug: {bug_summary_clean}",
        f" * Mission thread_id: {thread_id}",
        f" * Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        " *",
        " * Prerequisites:",
        " *   npm i -D @playwright/test && npx playwright install chromium",
        f" *   Place an authenticated storage state at ./{storage_state_path}",
        " * Run:",
        " *   npx playwright test " + os.path.basename(spec_path) + " --headed",
        " */",
        "import { test, expect } from '@playwright/test';",
        "",
        f"test.use({{ storageState: '{storage_state_path}' }});",
        "",
        f"test('reproduce: {_ts_escape(bug_summary_clean)}', async ({{ page }}) => {{",
        "  test.setTimeout(120_000);",
    ]
    if app_url:
        header.append(f"  // App base URL recorded at capture time: {app_url}")
    footer = ["});", ""]
    content = "\n".join(header + steps + footer)
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.abspath(spec_path)


# ---------------------------------------------------------
# Code Generator Tool (callable by agents, e.g. on bug capture)
# ---------------------------------------------------------
def get_code_generator_tool(app_url: Optional[str] = None):
    @tool
    async def generate_reproduction_spec(bug_summary: str, config: RunnableConfig) -> str:
        """Translate the immutable Action Tape into a 100%-reproducible
        Playwright .spec.ts script a developer can run locally.

        Call this *immediately after* `capture_bug_screenshot` so every bug has
        both a visual evidence and an executable repro script."""
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        tape = get_action_tape(thread_id)
        if not tape:
            return "No Action Tape recorded yet — nothing to translate."
        try:
            path = generate_playwright_spec(thread_id, bug_summary, app_url=app_url)
            return f"Reproduction spec generated at {path} ({len(tape)} recorded actions)."
        except Exception as exc:
            return f"Failed to generate reproduction spec: {exc}"
    return generate_reproduction_spec

