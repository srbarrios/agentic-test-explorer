import asyncio
import base64
import json
import os
import io
import shlex
import stat
import sys
import time
import re
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from playwright.async_api import Page
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv
load_dotenv()

# --- Vision Model Setup ---
_provider = os.getenv("LLM_PROVIDER", "").lower()
_VISION_MODEL_NAME = (
    os.getenv("CLAUDE_VISION_MODEL") if _provider == "claude"
    else os.getenv("GEMINI_VISION_MODEL") or os.getenv("GEMINI_MODEL")
) or None
try:
    from agentic_explorer.utils.llm import make_llm
    vision_model = make_llm(temperature=0, model_name=_VISION_MODEL_NAME)
except Exception as e:
    print(f"Error initializing vision model: {e}")
    vision_model = None

# ---------------------------------------------------------
# MCP & Skill Tools
# ---------------------------------------------------------
async def get_mcp_tools(config_path: Optional[Union[str, Path]] = None) -> list:
    """Load tools from user-configured MCP servers.

    Reads a Claude-Desktop-compatible JSON file of the form:
        {
          "mcpServers": {
            "<name>": {"transport": "http", "url": "https://..."},
            "<name>": {"command": "node", "args": ["server.js"]}
          }
        }

    The path is resolved from (in order):
      1. ``config_path`` argument
      2. ``MCP_SERVERS_CONFIG`` env var
      3. ``./mcp_servers.json``

    Returns an empty list (and logs an info message) if the file is missing or
    contains no servers — agents can still operate with their browser tools.
    """
    resolved = Path(config_path or os.getenv("MCP_SERVERS_CONFIG", "./mcp_servers.json"))
    if not resolved.is_file():
        print(f"ℹ️  No MCP server config found at {resolved}. Skipping MCP tools.")
        return []

    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"⚠️  Failed to read MCP config {resolved}: {exc}")
        return []

    servers = data.get("mcpServers") or {}
    if not servers:
        print(f"ℹ️  MCP config {resolved} contains no servers. Skipping MCP tools.")
        return []

    try:
        client = MultiServerMCPClient(servers)
        return await client.get_tools()
    except Exception as exc:
        print(f"⚠️  Failed to connect to MCP servers from {resolved}: {exc}")
        return []


# Root directory where the user installs Agent Skills following the
# https://agentskills.io/specification layout.
AGENT_SKILLS_ROOT = Path(os.getenv("AGENT_SKILLS_ROOT", "./agent-skills")).resolve()
# Default timeout (seconds) for `run_agent_skill_script` subprocess executions.
SKILL_SCRIPT_TIMEOUT_SECONDS = int(os.getenv("AGENT_SKILL_SCRIPT_TIMEOUT", "60"))
# Cap the textual output returned by script executions to avoid flooding the LLM.
SKILL_SCRIPT_OUTPUT_LIMIT = 8000


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Locate a skill directory by name under AGENT_SKILLS_ROOT.

    Skills may be nested under category sub-folders (e.g.
    ``skills/<category>/<skill_name>``), so we recursively search for a
    directory that matches the requested skill name and contains SKILL.md.
    """
    if not AGENT_SKILLS_ROOT.is_dir():
        return None
    target = skill_name.strip().strip("/").lower()
    if not target:
        return None
    # Fast path: explicit relative path supplied.
    direct = (AGENT_SKILLS_ROOT / skill_name / "SKILL.md")
    if direct.is_file():
        return direct.parent
    for skill_md in AGENT_SKILLS_ROOT.rglob("SKILL.md"):
        if skill_md.parent.name.lower() == target:
            return skill_md.parent
    return None


def _read_text_safe(path: Path, max_chars: int = 20000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unable to read {path.name}: {exc}>"
    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n… [truncated {len(content) - max_chars} chars]"
    return content


@tool
def fetch_agent_skill(skill_name: str) -> str:
    """
    Load an installed Agent Skill from the user's local skills directory.

    Follows the https://agentskills.io/specification progressive-disclosure model:
      * Always returns the SKILL.md contents.
      * Appends every Markdown file found recursively under references/.
      * Lists available scripts/ entries (names + exec bits) so the agent can
        decide whether to invoke `run_agent_skill_script`.
      * Lists available assets/ filenames for awareness.

    Provide either the bare skill directory name (e.g. "my-skill") or
    a relative path under the configured skills root (e.g. "skills/category/my-skill").
    """
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        return (
            f"Skill '{skill_name}' not found under {AGENT_SKILLS_ROOT}. "
            "Install skills under AGENT_SKILLS_ROOT (default './agent-skills/') following "
            "the https://agentskills.io/specification layout."
        )

    sections: list[str] = []
    rel_root = skill_dir.relative_to(AGENT_SKILLS_ROOT)
    sections.append(f"# Skill: {skill_dir.name}\n_Location: {rel_root}_")

    # 1. SKILL.md (required by spec)
    skill_md = skill_dir / "SKILL.md"
    sections.append("## SKILL.md\n" + _read_text_safe(skill_md))

    # 2. references/ — recursive Markdown documentation
    references_dir = skill_dir / "references"
    if references_dir.is_dir():
        md_files = sorted(p for p in references_dir.rglob("*.md") if p.is_file())
        if md_files:
            ref_section = ["## references/"]
            for md in md_files:
                rel = md.relative_to(skill_dir)
                ref_section.append(f"### {rel.as_posix()}\n{_read_text_safe(md)}")
            sections.append("\n\n".join(ref_section))

    # 3. scripts/ — list with exec info (bodies are NOT loaded, per progressive disclosure)
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        script_entries = []
        for path in sorted(p for p in scripts_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(skill_dir).as_posix()
            mode = path.stat().st_mode
            executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
            script_entries.append(f"- `{rel}`{' (executable)' if executable else ''}")
        if script_entries:
            sections.append(
                "## scripts/\n"
                "Invoke these via the `run_agent_skill_script` tool. "
                "See https://agentskills.io/skill-creation/using-scripts for guidance.\n"
                + "\n".join(script_entries)
            )

    # 4. assets/ — awareness only
    assets_dir = skill_dir / "assets"
    if assets_dir.is_dir():
        asset_entries = [
            f"- `{p.relative_to(skill_dir).as_posix()}`"
            for p in sorted(assets_dir.rglob("*")) if p.is_file()
        ]
        if asset_entries:
            sections.append("## assets/\n" + "\n".join(asset_entries))

    return "\n\n".join(sections)


@tool
async def run_agent_skill_script(
    skill_name: str,
    script_path: str,
    arguments: str = "",
    timeout_seconds: int = SKILL_SCRIPT_TIMEOUT_SECONDS,
) -> str:
    """
    Execute a script shipped inside an Agent Skill's `scripts/` folder.

    Args:
        skill_name: The skill directory name (e.g. "my-skill") or a
            relative path under the configured skills root.
        script_path: Script path RELATIVE to the skill root (e.g. "scripts/extract.py").
            Must stay inside the skill's scripts/ directory.
        arguments: Optional shell-style argument string forwarded to the script.
        timeout_seconds: Hard timeout for the subprocess. Defaults to 60s.

    Returns:
        A textual report containing exit code, stdout and stderr (truncated).
    """
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        return (
            f"Skill '{skill_name}' not found under {AGENT_SKILLS_ROOT}. "
            "Install it following the https://agentskills.io/specification layout."
        )

    scripts_dir = (skill_dir / "scripts").resolve()
    if not scripts_dir.is_dir():
        return f"Skill '{skill_dir.name}' has no scripts/ directory."

    # Allow callers to pass either "extract.py" or "scripts/extract.py".
    candidate = (skill_dir / script_path).resolve()
    if not candidate.exists():
        candidate = (scripts_dir / script_path).resolve()

    try:
        candidate.relative_to(scripts_dir)
    except ValueError:
        return f"Refusing to execute '{script_path}': path escapes the skill's scripts/ directory."

    if not candidate.is_file():
        return f"Script '{script_path}' not found inside {scripts_dir}."

    # Determine how to invoke the script based on shebang / extension.
    suffix = candidate.suffix.lower()
    argv: list[str]
    if suffix == ".py":
        argv = [sys.executable, str(candidate)]
    elif suffix in {".js", ".mjs", ".cjs"}:
        argv = ["node", str(candidate)]
    elif suffix in {".sh", ".bash"}:
        argv = ["bash", str(candidate)]
    else:
        mode = candidate.stat().st_mode
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            argv = [str(candidate)]
        else:
            return (
                f"Don't know how to run '{candidate.name}' (unsupported extension "
                f"'{suffix or 'none'}' and not marked executable)."
            )

    if arguments:
        try:
            argv.extend(shlex.split(arguments))
        except ValueError as exc:
            return f"Failed to parse arguments: {exc}"

    env = os.environ.copy()
    # Help scripts that want to resolve their own skill root.
    env.setdefault("AGENT_SKILL_DIR", str(skill_dir))

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(skill_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        return f"Runtime not found for '{candidate.name}': {exc}"

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return f"Script timed out after {timeout_seconds}s: {script_path}"

    def _clip(b: bytes) -> str:
        text = b.decode("utf-8", errors="replace")
        if len(text) > SKILL_SCRIPT_OUTPUT_LIMIT:
            text = text[:SKILL_SCRIPT_OUTPUT_LIMIT] + f"\n… [truncated {len(text) - SKILL_SCRIPT_OUTPUT_LIMIT} chars]"
        return text

    return (
        f"Executed: {' '.join(shlex.quote(a) for a in argv)}\n"
        f"Exit code: {proc.returncode}\n"
        f"--- stdout ---\n{_clip(stdout_b) or '<empty>'}\n"
        f"--- stderr ---\n{_clip(stderr_b) or '<empty>'}"
    )

# ---------------------------------------------------------
# Visual & Page Tools (Tool Factories)
# ---------------------------------------------------------
def get_visual_validation_tool(page: Page):
    @tool
    async def analyze_visual_state(validation_context: str) -> str:
        """
        Captures a screenshot of the current page and uses Gemini to
        analyze it for visual anomalies based on the provided validation_context.
        """
        if not vision_model:
            return "Error: Visual validation model is not available."

        print(f"Executing visual analysis with context: '{validation_context}'...")
        screenshot_bytes = await page.screenshot(full_page=False)

        try:
            image = Image.open(io.BytesIO(screenshot_bytes))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as err:
            return f"Error during screenshot processing: {err}"

        system_prompt = (
            "You are an expert UI QA analyst validating visual rendering. "
            "Analyze the provided image. Respond only with 'PASS' if the visual state is correct "
            "according to the user's instructions. If there is a rendering issue or visual anomaly, "
            "respond with 'FAIL: ' followed by a detailed description of the error."
        )

        human_prompt = HumanMessage(
            content=[
                {"type": "text", "text": f"Instruction: {validation_context}"},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_str}"}
            ]
        )

        try:
            response = await vision_model.ainvoke([
                SystemMessage(content=system_prompt),
                human_prompt
            ])
            return f"Visual Analysis complete. Result: {response.content}"
        except Exception as err:
            return f"Error interacting with the vision model: {err}"

    return analyze_visual_state

def get_screenshot_tool(page: Page):
    @tool
    async def capture_bug_screenshot(bug_summary: str, config: RunnableConfig) -> str:
        """
        USE THIS TOOL immediately when you find a bug, missing element, or visual anomaly.
        Provide a short, descriptive bug_summary (e.g., 'missing_search_bar' or 'chart_render_error').
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        screenshot_dir = f"report_{thread_id}/screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', bug_summary)[:30].strip('_')
        filename = f"{screenshot_dir}/bug_{safe_name}_{int(time.time())}.png"

        try:
            await page.screenshot(path=filename, full_page=True)
            return f"Evidence captured! Screenshot successfully saved to {filename}"
        except Exception as err:
            return f"Failed to capture screenshot: {str(err)}"

    return capture_bug_screenshot
