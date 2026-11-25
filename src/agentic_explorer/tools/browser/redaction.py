"""Keep typed secrets out of every artifact.

The agent types into the page through ``fill`` and the engine records the command, so a
password or token typed during a mission used to end up in plain text in
``action_tape.jsonl``, in the generated ``.spec.ts`` and in ``traces.log``.

A fill is treated as sensitive when its selector looks like a secret field (password, token,
OTP, API key, ...) or when the element itself is ``type=password``. The browser still gets
the real value; the tape, the tool output and the spec only see ``REDACTED``. Values that
were redacted are remembered for the life of the process (never written anywhere) so the
trace writer can scrub them from message text too.

Not covered: the LLM's own conversation history inside the LangGraph checkpoint
(``agent_memory.sqlite``) still contains what the agent typed. Do not put real secrets in
mission prompts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Set

REDACTED = "<redacted>"
_SENSITIVE_RE = re.compile(
    r"pass(?:word|wd)?|pwd|token|secret|otp|2fa|mfa|api[-_ ]?key|cvv|card[-_ ]?number|ssn",
    re.IGNORECASE,
)
_MIN_SCRUBBED_LEN = 4  # shorter values would mangle ordinary text when scrubbed

_SECRETS: Set[str] = set()


def is_sensitive_selector(selector: Any) -> bool:
    """True when ``selector`` looks like it targets a secret field."""
    return bool(selector) and bool(_SENSITIVE_RE.search(str(selector)))


def redact_params(
    action: str, params: Mapping[str, Any], *, element_is_password: bool = False,
) -> Dict[str, Any]:
    """A copy of ``params`` that is safe to record; only ``fill`` of a sensitive field changes."""
    safe = dict(params)
    value = safe.get("value")
    if action != "fill" or not value:
        return safe
    if element_is_password or is_sensitive_selector(safe.get("selector")):
        if len(str(value)) >= _MIN_SCRUBBED_LEN:
            _SECRETS.add(str(value))
        safe["value"] = REDACTED
        safe["redacted"] = True
    return safe


def scrub(text: str) -> str:
    """``text`` with every remembered secret replaced by the placeholder."""
    for secret in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


def forget_all() -> None:
    """Drop the remembered secrets (tests)."""
    _SECRETS.clear()
