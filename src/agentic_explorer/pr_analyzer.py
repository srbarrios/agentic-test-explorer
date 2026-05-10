"""PR-driven test scenario generation.

Extracts code changes and metadata from a GitHub Pull Request, then uses an
LLM to generate targeted mission YAML that covers product areas impacted by
the changes.

Data fetching strategy (in priority order):
  1. **GitHub MCP server** — if a ``"github"`` entry exists in
     ``mcp_servers.json``, the analyzer connects via ``MultiServerMCPClient``
     and calls ``get_pull_request``, ``get_pull_request_files``, and
     ``get_pull_request_diff``.
  2. **``gh`` CLI** — used as a fallback when the MCP server is not
     configured, unreachable, or missing required tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agentic_explorer.config import AppMeta
from agentic_explorer.utils.llm_json import extract_yaml_text

_PR_URL_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")

MAX_DIFF_CHARS = 100_000


@dataclass
class PRData:
    owner: str
    repo: str
    number: int
    title: str
    body: str
    diff: str
    files_changed: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    url: str = ""


def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Extract ``(owner, repo, pr_number)`` from a GitHub PR URL."""
    m = _PR_URL_RE.search(url)
    if not m:
        raise ValueError(
            f"Invalid GitHub PR URL: {url}\n"
            "Expected format: https://github.com/owner/repo/pull/123"
        )
    return m.group(1), m.group(2), int(m.group(3))


# ---------------------------------------------------------------------------
# MCP-based PR data fetching (preferred)
# ---------------------------------------------------------------------------

def _extract_mcp_text(result: Any) -> str:
    """Extract plain text from an MCP tool result.

    Results arrive as a content-block list
    (``[{"type": "text", "text": "..."}]``) or sometimes as a raw string.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(result)


def _find_github_server(mcp_config_path: Optional[Union[str, Path]]) -> Optional[dict]:
    """Locate a GitHub MCP server entry in the config file.

    Supports both project format (``mcpServers`` / ``transport``) and Claude
    Code format (``servers`` / ``type``).  Returns a single-entry dict ready
    for ``MultiServerMCPClient``, or ``None``.
    """
    resolved = Path(mcp_config_path or os.getenv("MCP_SERVERS_CONFIG", "./mcp_servers.json"))
    if not resolved.is_file():
        return None

    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None

    servers = data.get("mcpServers") or data.get("servers") or {}
    for name, cfg in servers.items():
        if "github" in name.lower():
            server_cfg = dict(cfg)
            if "type" in server_cfg and "transport" not in server_cfg:
                server_cfg["transport"] = server_cfg.pop("type")
            return {name: server_cfg}
    return None


async def _fetch_pr_data_mcp(
    owner: str, repo: str, number: int,
    mcp_config_path: Optional[Union[str, Path]],
) -> PRData:
    """Fetch PR data via the GitHub MCP server."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    server_entry = _find_github_server(mcp_config_path)
    if server_entry is None:
        raise RuntimeError("No GitHub MCP server configured")

    server_name = next(iter(server_entry))
    print(f"  Connecting to GitHub MCP server '{server_name}'...")

    client = MultiServerMCPClient(server_entry)
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}

    pr_tool = tool_map.get("get_pull_request")
    diff_tool = tool_map.get("get_pull_request_diff")
    files_tool = tool_map.get("get_pull_request_files")

    if not pr_tool:
        raise RuntimeError(
            f"GitHub MCP server '{server_name}' does not expose 'get_pull_request' tool"
        )

    pr_args = {"owner": owner, "repo": repo, "pull_number": number}

    tasks = [pr_tool.ainvoke(pr_args)]
    if diff_tool:
        tasks.append(diff_tool.ainvoke(pr_args))
    if files_tool:
        tasks.append(files_tool.ainvoke(pr_args))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    pr_result = results[0]
    if isinstance(pr_result, Exception):
        raise pr_result
    pr_text = _extract_mcp_text(pr_result)

    diff = ""
    if diff_tool and len(results) > 1 and not isinstance(results[1], Exception):
        diff = _extract_mcp_text(results[1])
    files_text = ""
    idx = 2 if diff_tool else 1
    if files_tool and len(results) > idx and not isinstance(results[idx], Exception):
        files_text = _extract_mcp_text(results[idx])

    title, body, labels = _parse_pr_metadata(pr_text)

    if len(diff) > MAX_DIFF_CHARS:
        total = len(diff)
        diff = diff[:MAX_DIFF_CHARS] + (
            f"\n\n... [diff truncated at {MAX_DIFF_CHARS:,} chars; {total:,} total]"
        )
        print(f"  Warning: PR diff truncated from {total:,} to {MAX_DIFF_CHARS:,} chars")

    files_changed = _parse_files_text(files_text) if files_text else []

    return PRData(
        owner=owner, repo=repo, number=number,
        title=title, body=body, diff=diff,
        files_changed=files_changed, labels=labels,
        url=f"https://github.com/{owner}/{repo}/pull/{number}",
    )


def _parse_pr_metadata(text: str) -> tuple[str, str, list[str]]:
    """Best-effort extraction of title, body, and labels from MCP text output."""
    try:
        data = json.loads(text)
        title = data.get("title", "")
        body = data.get("body", "") or ""
        labels_raw = data.get("labels", []) or []
        labels = [
            (lbl.get("name", "") if isinstance(lbl, dict) else str(lbl))
            for lbl in labels_raw
        ]
        return title, body, labels
    except (json.JSONDecodeError, AttributeError):
        pass
    title = text.split("\n", 1)[0][:200]
    body = text
    return title, body, []


def _parse_files_text(text: str) -> list[dict[str, Any]]:
    """Best-effort extraction of file change stats from MCP text output."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [
                {
                    "filename": f.get("filename") or f.get("path", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                }
                for f in data
            ]
    except (json.JSONDecodeError, AttributeError):
        pass
    files = []
    for line in text.strip().splitlines():
        line = line.strip("- ")
        if line:
            files.append({"filename": line, "additions": 0, "deletions": 0})
    return files


# ---------------------------------------------------------------------------
# gh CLI-based PR data fetching (fallback)
# ---------------------------------------------------------------------------

async def _run_gh(*args: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "gh", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return proc.returncode, stdout_b.decode(), stderr_b.decode()


async def _check_gh_available() -> None:
    if shutil.which("gh") is None:
        raise RuntimeError(
            "The 'gh' CLI is required but not found on your PATH.\n"
            "Install it from https://cli.github.com/"
        )
    rc, _, stderr = await _run_gh("auth", "status")
    if rc != 0:
        raise RuntimeError(
            "GitHub CLI is not authenticated. Run 'gh auth login' first.\n"
            f"Details: {stderr.strip()}"
        )


async def _fetch_pr_data_gh(owner: str, repo: str, number: int) -> PRData:
    """Fetch PR metadata and diff using the ``gh`` CLI."""
    await _check_gh_available()

    repo_slug = f"{owner}/{repo}"
    meta_task = _run_gh(
        "pr", "view", str(number), "--repo", repo_slug,
        "--json", "title,body,labels,files",
    )
    diff_task = _run_gh("pr", "diff", str(number), "--repo", repo_slug)

    (meta_rc, meta_out, meta_err), (diff_rc, diff_out, diff_err) = (
        await asyncio.gather(meta_task, diff_task)
    )

    if meta_rc != 0:
        raise RuntimeError(f"Failed to fetch PR #{number}: {meta_err.strip()}")
    if diff_rc != 0:
        raise RuntimeError(f"Failed to fetch PR diff: {diff_err.strip()}")

    meta = json.loads(meta_out)

    diff = diff_out
    if len(diff) > MAX_DIFF_CHARS:
        total = len(diff)
        diff = diff[:MAX_DIFF_CHARS] + (
            f"\n\n... [diff truncated at {MAX_DIFF_CHARS:,} chars; {total:,} total]"
        )
        print(f"  Warning: PR diff truncated from {total:,} to {MAX_DIFF_CHARS:,} chars")

    files_changed = []
    for f in meta.get("files", []) or []:
        files_changed.append({
            "filename": f.get("path", ""),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
        })

    labels = [lbl.get("name", "") for lbl in meta.get("labels", []) or []]

    return PRData(
        owner=owner, repo=repo, number=number,
        title=meta.get("title", ""),
        body=meta.get("body", "") or "",
        diff=diff,
        files_changed=files_changed, labels=labels,
        url=f"https://github.com/{owner}/{repo}/pull/{number}",
    )


# ---------------------------------------------------------------------------
# Public fetch entry point — MCP first, gh fallback
# ---------------------------------------------------------------------------

async def fetch_pr_data(
    owner: str, repo: str, number: int,
    mcp_config_path: Optional[Union[str, Path]] = None,
) -> PRData:
    """Fetch PR data, preferring the GitHub MCP server over the ``gh`` CLI."""
    try:
        print("  Trying GitHub MCP server...")
        return await _fetch_pr_data_mcp(owner, repo, number, mcp_config_path)
    except Exception as mcp_err:
        print(f"  GitHub MCP server unavailable ({mcp_err}), falling back to gh CLI...")

    return await _fetch_pr_data_gh(owner, repo, number)


# ---------------------------------------------------------------------------
# LLM-based mission generation
# ---------------------------------------------------------------------------

def _format_file_list(files: list[dict[str, Any]]) -> str:
    lines = []
    for f in files:
        name = f["filename"]
        adds = f.get("additions", 0)
        dels = f.get("deletions", 0)
        lines.append(f"- {name} (+{adds}, -{dels})")
    return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are a QA Test Architect. You analyze code changes in pull requests and generate \
targeted test missions for an autonomous testing framework.

The framework has 6 agent types, each specializing in specific UI patterns:
- listing_agent: List views, search/filter, pagination, data tables, row details, flyouts
- graph_agent: Node-link graphs, timelines, tree visualizations, SVG/Canvas renders
- chart_agent: Charts, dashboards, KPI tiles, time-range pickers, gauge widgets
- map_agent: Geographic maps, status grids, spatial overlays, marker clusters
- form_agent: Forms, multi-step wizards, validation flows, configuration screens, input fields
- explorer_agent (autonomous): Open-ended chaos testing, cross-feature integration, \
edge cases, regression sweeps (use thread_id containing "explorer" or "autonomous")

Each mission has:
  - thread_id: A unique identifier namespaced to this PR (format: pr_{number}_{agent_type}_{nn})
  - prompt: A detailed, actionable test instruction for the agent

Output FORMAT (raw YAML, no code fences):

missions:
  - thread_id: "pr_123_listing_01"
    prompt: >
      Navigate to ... verify ... interact with ...

Rules:
1. Generate 3-8 missions depending on the scope of changes.
2. Each prompt MUST be specific and actionable, referencing concrete UI areas/flows.
3. Map code changes to the MOST relevant agent type based on what UI pattern is affected.
4. Include at least one explorer/autonomous mission for broad regression coverage.
5. Thread IDs MUST follow the pattern: pr_{number}_{agenttype}_{nn}
6. Use "explorer" or "autonomous" in thread_id for chaos-exploration missions.
7. Prompts should reference the specific features/areas that the code changes touch.
8. Do NOT use placeholder values — use the actual app name and URL provided.
9. Each prompt should tell the agent what page to navigate to, what to interact with, \
and what to verify.
"""


def _build_human_message(pr: PRData, app: AppMeta) -> str:
    file_list = _format_file_list(pr.files_changed)
    return (
        f"## Application Under Test\n"
        f"- Name: {app.name}\n"
        f"- URL: {app.url}\n"
        f"- Description: {app.description}\n\n"
        f"## Pull Request #{pr.number}\n"
        f"- Title: {pr.title}\n"
        f"- URL: {pr.url}\n\n"
        f"### PR Description\n{pr.body}\n\n"
        f"### Files Changed ({len(pr.files_changed)} files)\n{file_list}\n\n"
        f"### Code Diff\n{pr.diff}\n\n"
        f"---\n\n"
        f"Based on these code changes, generate targeted test missions that cover the "
        f"UI areas most likely impacted. Focus on:\n"
        f"1. Direct functional changes (features added/modified by this PR)\n"
        f"2. Adjacent areas that could regress from these changes\n"
        f"3. Integration points between the changed code and existing features\n"
    )


def _validate_missions(data: Any, pr_number: int) -> dict:
    if not isinstance(data, dict) or "missions" not in data:
        raise ValueError("LLM response missing top-level 'missions' key")
    missions = data["missions"]
    if not isinstance(missions, list) or not missions:
        raise ValueError("'missions' must be a non-empty list")
    for i, m in enumerate(missions):
        if not isinstance(m, dict):
            raise ValueError(f"Mission {i} is not a dict")
        if "thread_id" not in m or "prompt" not in m:
            raise ValueError(f"Mission {i} missing 'thread_id' or 'prompt'")
        if not isinstance(m["thread_id"], str) or not isinstance(m["prompt"], str):
            raise ValueError(f"Mission {i} 'thread_id' and 'prompt' must be strings")
    return data


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(x in msg for x in ("503", "UNAVAILABLE", "429", "RATE_LIMIT", "RESOURCE_EXHAUSTED", "QUOTA"))


async def generate_missions_from_pr(pr_data: PRData, app: AppMeta) -> dict:
    """Use an LLM to generate targeted test missions from PR data."""
    model_name = os.getenv(
        "GEMINI_SCENARIO_MODEL",
        os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    )
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    system_msg = SystemMessage(content=_SYSTEM_PROMPT)
    human_msg = HumanMessage(content=_build_human_message(pr_data, app))

    max_retries = 3
    base_delay = 2
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([system_msg, human_msg])
            raw_yaml = extract_yaml_text(response.content)
            parsed = yaml.safe_load(raw_yaml)
            return _validate_missions(parsed, pr_data.number)
        except (yaml.YAMLError, ValueError) as e:
            last_error = e
            if attempt < max_retries - 1:
                human_msg = HumanMessage(content=(
                    f"Your previous response could not be parsed: {e}\n\n"
                    "Please output ONLY valid YAML with no code fences, matching this structure:\n\n"
                    "missions:\n"
                    '  - thread_id: "pr_NNN_agenttype_01"\n'
                    "    prompt: >\n"
                    "      Detailed test instructions...\n"
                ))
        except Exception as e:
            if _is_transient_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Transient error during scenario generation. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                last_error = e
            else:
                raise

    raise RuntimeError(
        f"Failed to generate valid mission YAML after {max_retries} attempts: {last_error}"
    )
