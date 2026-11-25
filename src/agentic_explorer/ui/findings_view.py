"""What the Bugs tab shows: each bug with its evidence, uncertain findings in their own list.

Kept free of Streamlit so it can be tested; ``dashboard.py`` only draws the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from agentic_explorer.orchestration.findings import find_evidence


@dataclass
class FindingsView:
    confirmed: List[Tuple[str, str]] = field(default_factory=list)  # (bug text, markdown body)
    uncertain: List[Tuple[str, str]] = field(default_factory=list)


def bug_markdown(bug: str, evidence: Sequence[Mapping[str, str]]) -> str:
    """The bug text followed by Observed / Expected / Where, as Markdown."""
    record: Dict[str, str] = dict(find_evidence(bug, list(evidence)) or {})
    lines = [bug, ""]
    lines.append(f"**Observed:** {record['observed']}" if record.get("observed")
                 else "**Observed:** _no observed value recorded_")
    if record.get("expected"):
        lines.append(f"**Expected:** {record['expected']}")
    where = record.get("observed_at") or record.get("location")
    if where:
        lines.append(f"**Where:** `{where}`")
    if record.get("severity"):
        lines.append(f"**Severity:** {record['severity']}")
    return "  \n".join(lines)


def split_findings_for_display(state: Mapping[str, Any]) -> FindingsView:
    """Confirmed bugs and uncertain findings from a ``.agent_state.json`` payload (any age)."""
    evidence = state.get("bug_evidence") or []
    return FindingsView(
        confirmed=[(b, bug_markdown(b, evidence)) for b in state.get("bugs_found") or []],
        uncertain=[(b, bug_markdown(b, evidence)) for b in state.get("uncertain_findings") or []],
    )
