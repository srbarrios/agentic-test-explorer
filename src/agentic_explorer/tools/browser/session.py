"""Telling that the saved session (``auth.json``) is no longer valid.

When the session expires the app redirects to a login page. Without this check the agent
explores the login form as if it were the application, the tape fills with noise and the
"bug" it reports is the login page itself.

The check only runs when a session was loaded at start (``arm(expected=True, ...)``).
It is deliberately conservative, because a false alarm stops a healthy mission:

* the page landed on a login path (``/login``, ``/signin``, ``/sso``, ``/auth``), or
* it shows a password field right after the page changed, on a path that is not about
  passwords or accounts (so a "change password" page is fine).

Not a lost session: the agent navigated to a login URL on purpose, or it already typed a
password itself (it is logging in).

On detection the command answers ``STATUS: AUTH_REQUIRED`` and the run is drained: the
checkpoint and the tape are kept, the process exits with status 130, and running the same
command again after ``agent-auth`` resumes the mission.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

from agentic_explorer.tools.browser.redaction import is_sensitive_selector

# One pattern for the runtime check and for the generated spec (which renders it as a JS regex).
LOGIN_PATH_PATTERN = r"/(log-?in|sign-?in|auth|sso)(/|$)"
_LOGIN_RE = re.compile(LOGIN_PATH_PATTERN, re.IGNORECASE)
# A password field here is normal: the page is about passwords or accounts.
_ACCOUNT_PATH_RE = re.compile(r"/(pass(word)?|settings|security|account|profile|register|sign-?up|reset|forgot)", re.IGNORECASE)

DRAIN_REASON = "auth_required"
AUTH_REQUIRED_MESSAGE = (
    "The saved session (auth.json) is no longer valid: the app sent you to a login page. "
    "The mission is stopping at a safe point; the operator must run `agent-auth` and then run the same "
    "command again to resume. Do NOT report the login page as a bug and do not call more tools: "
    "end your turn now with <bugs_found>[]</bugs_found>."
)

_state: Dict[str, Any] = {"expected": False, "control": None}


def is_login_path(path: str) -> bool:
    return bool(_LOGIN_RE.search(path or ""))


def arm(*, expected: bool, control: Any = None) -> None:
    """Enable the check (``expected``: a session was loaded) and set what to drain on detection."""
    _state["expected"] = bool(expected)
    _state["control"] = control


def reset() -> None:
    arm(expected=False, control=None)


def is_armed() -> bool:
    return bool(_state["expected"])


def request_stop() -> None:
    """Ask the run to stop at the next safe point, if there is something to stop."""
    control = _state["control"]
    if control is not None:
        control.request_drain(DRAIN_REASON)


def _agent_is_logging_in(tape: List[Mapping[str, Any]]) -> bool:
    for entry in tape:
        params = entry.get("params") or {}
        if entry.get("action") == "fill" and (params.get("redacted") or is_sensitive_selector(params.get("selector"))):
            return True
    return False


def looks_like_session_loss(
    *,
    action: str,
    params: Mapping[str, Any],
    landed_path: str,
    previous_path: Optional[str],
    password_field: bool,
    tape: List[Mapping[str, Any]],
) -> bool:
    """True when the command ended on a login page the agent did not ask for."""
    if action == "navigate" and is_login_path(urlparse(str(params.get("url", ""))).path):
        return False
    if _agent_is_logging_in(tape):
        return False
    if is_login_path(landed_path):
        return True
    page_changed = previous_path is None or previous_path != landed_path
    return password_field and page_changed and not _ACCOUNT_PATH_RE.search(landed_path or "")


async def check_page(page: Any, *, action: str, params: Mapping[str, Any], tape: List[Mapping[str, Any]]) -> bool:
    """Run the check against the live page; never raises (a broken page is not a lost session)."""
    if not is_armed():
        return False
    try:
        landed_path = urlparse(page.url).path or "/"
        previous = tape[-1].get("page_url") if tape else None
        previous_path = (urlparse(previous).path or "/") if previous else None
        password_field = (await page.locator('input[type="password"]').count()) > 0
    except Exception:
        return False
    return looks_like_session_loss(
        action=action, params=params, landed_path=landed_path, previous_path=previous_path,
        password_field=password_field, tape=tape,
    )
