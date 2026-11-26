"""The ``<bugs_found>`` contract: parsing, evidence, confidence and deduplication.

An agent closes a turn with a JSON block. Besides ``title`` / ``summary`` / ``location`` /
``severity`` an entry may carry evidence (all optional, so old blocks stay valid):

* ``observed``    the exact value the agent saw
* ``observed_at`` where it was read (selector or region text, URL)
* ``expected``    what it should have been
* ``confidence``  ``"confirmed"`` (default) or ``"uncertain"``

Design: ``bugs_found`` in the graph state stays a list of ``"title: summary"`` strings
and holds only **confirmed** findings, so every existing count, report and memory write
keeps its meaning. Uncertain findings and the evidence travel in two parallel state
fields (``uncertain_findings`` and ``bug_evidence``), matched back to a bug through the
same normalised key that deduplicates bugs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CONFIRMED = "confirmed"
UNCERTAIN = "uncertain"
CONFIDENCES = (CONFIRMED, UNCERTAIN)
SEVERITIES = ("critical", "major", "minor")
CONTRACT_FIELDS = (
    "title", "summary", "location", "severity", "observed", "observed_at", "expected", "confidence",
)
_REQUIRED_FIELDS = ("title", "summary")
_EVIDENCE_FIELDS = ("observed", "observed_at", "expected", "location", "severity")
_MAX_ENTRY_CHARS = 500
_MAX_VALUE_CHARS = 500

_BUGS_TAG_RE = re.compile(r"<bugs_found>\s*(.*?)\s*</bugs_found>", re.DOTALL)
# Conservative fallback: only fire on explicit "BUG FOUND" / "BUG:" prefixes,
# optionally bold-wrapped, at line/sentence start. Avoids false positives from
# narrative mentions of "bug" elsewhere in summaries.
_BUG_PROSE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:BUG\s+FOUND|BUG|ISSUE\s+FOUND)\s*(?:\*\*)?\s*[:\-—]\s*([^\n]{8,300})",
    re.IGNORECASE,
)


# ---------------------------------------------------------
# Keys and deduplication (shared with graph_base and the dashboard)
# ---------------------------------------------------------

def normalize_bug_key(text: str) -> str:
    """Normalize a bug entry to a stable comparable form.

    Agents emit the same finding in three different shapes:
    - snake_case slug from ``capture_bug_screenshot`` (``form_validation_race``)
    - structured title:summary from the closing tag
    - prose like ``BUG FOUND: Empty cart lacks messaging on /cart page.``

    Title-prefix matching (``is_dup_key``) collapses these together.
    """
    title = text.split(":", 1)[0] if ":" in text else text
    title = title.split(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()[:60]


def is_dup_key(new_key: str, seen_keys: Sequence[str], *, min_overlap: int = 12) -> bool:
    """True if ``new_key`` is essentially the same bug as one already seen.

    Match when either key fully prefixes the other and the shorter side has
    at least ``min_overlap`` chars, so two genuinely different bugs that
    share an opening word ('Login error', 'Login redirect') stay separate.
    """
    for existing in seen_keys:
        short, long = (new_key, existing) if len(new_key) <= len(existing) else (existing, new_key)
        if len(short) >= min_overlap and long.startswith(short):
            return True
    return False


def dedupe_bugs(bugs: Sequence[str]) -> List[str]:
    """Return *bugs* with duplicates removed (first-seen order preserved)."""
    out: List[str] = []
    seen: List[str] = []
    for bug in bugs:
        key = normalize_bug_key(bug)
        if not key or is_dup_key(key, seen):
            continue
        seen.append(key)
        out.append(bug)
    return out


def split_findings(confirmed: Sequence[str], uncertain: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Dedupe both lists; an uncertain finding that repeats a confirmed one is dropped."""
    confirmed_out = dedupe_bugs(confirmed)
    confirmed_keys = [normalize_bug_key(b) for b in confirmed_out]
    uncertain_out = [u for u in dedupe_bugs(uncertain) if not is_dup_key(normalize_bug_key(u), confirmed_keys)]
    return confirmed_out, uncertain_out


# ---------------------------------------------------------
# Parsing
# ---------------------------------------------------------

@dataclass
class ExtractedFindings:
    confirmed: List[str] = field(default_factory=list)
    uncertain: List[str] = field(default_factory=list)
    evidence: List[Dict[str, str]] = field(default_factory=list)


def one_line(text: str) -> str:
    """Collapse a multi-line value onto one line so it cannot break Markdown lists."""
    return " ⏎ ".join(part.strip() for part in str(text).splitlines() if part.strip())


def _clean_entry(entry: str) -> str:
    return entry.strip(": ").strip()[:_MAX_ENTRY_CHARS]


def _value(raw: Any) -> str:
    return "" if raw is None else str(raw).strip()[:_MAX_VALUE_CHARS]


def _confidence(raw: Any) -> str:
    """Absent means confirmed (old blocks); anything but ``confirmed`` is never rounded up."""
    if raw is None or str(raw).strip() == "":
        return CONFIRMED
    return CONFIRMED if str(raw).strip().lower() == CONFIRMED else UNCERTAIN


def extract_findings(text: str) -> ExtractedFindings:
    """Parse the ``<bugs_found>`` blocks of ``text`` (plus the ``BUG FOUND:`` prose fallback).

    Never raises: invalid JSON or unexpected shapes simply yield nothing for that block.
    """
    candidates: List[Tuple[str, str, Optional[Dict[str, str]]]] = []  # (entry, confidence, evidence record)

    for match in _BUGS_TAG_RE.finditer(text):
        try:
            entries = json.loads(match.group(1))
        except (ValueError, TypeError):
            continue
        if not isinstance(entries, list):
            continue
        for item in entries:
            if isinstance(item, dict):
                entry = _clean_entry(f"{item.get('title', '')}: {item.get('summary', '')}")
                confidence = _confidence(item.get("confidence"))
                record: Dict[str, str] = {"bug": entry, "confidence": confidence}
                for name in _EVIDENCE_FIELDS:
                    value = _value(item.get(name))
                    if value:
                        record[name] = value
                candidates.append((entry, confidence, record))
            elif isinstance(item, str):
                candidates.append((_clean_entry(item), CONFIRMED, None))

    found = ExtractedFindings()
    kept: Dict[str, Optional[Dict[str, str]]] = {}
    for wanted in (CONFIRMED, UNCERTAIN):  # confirmed first, so it wins over an uncertain twin
        seen_keys = [normalize_bug_key(b) for b in found.confirmed] if wanted == UNCERTAIN else []
        target = found.confirmed if wanted == CONFIRMED else found.uncertain
        for entry, confidence, record in candidates:
            key = normalize_bug_key(entry)
            if confidence != wanted or not key or is_dup_key(key, seen_keys):
                continue
            seen_keys.append(key)
            target.append(entry)
            kept[entry] = record

    # Prose fallback: "BUG FOUND: ..." in narrative summaries. Never overrides an uncertain verdict.
    known = [normalize_bug_key(b) for b in found.confirmed + found.uncertain]
    for match in _BUG_PROSE_RE.finditer(text):
        entry = _clean_entry(match.group(1))
        key = normalize_bug_key(entry)
        if not key or is_dup_key(key, known):
            continue
        known.append(key)
        found.confirmed.append(entry)

    found.evidence = [rec for entry in found.confirmed + found.uncertain if (rec := kept.get(entry))]
    return found


def _squash(value: Any) -> str:
    return " ".join(str(value or "").split())


def same_observation(observed: Any, expected: Any, records: Sequence[Mapping[str, str]]) -> bool:
    """True when some evidence record describes the very same observation.

    Titles are free text and one defect often gets two (one for the screenshot, another for the
    closing block), so the observed value is the reliable key: it must be non-empty and equal
    (ignoring whitespace) to a record's, and the expected values must agree when both exist.
    """
    seen = _squash(observed)
    if not seen:
        return False
    for rec in records:
        if _squash(rec.get("observed")) != seen:
            continue
        wanted, other = _squash(expected), _squash(rec.get("expected"))
        if not wanted or not other or wanted == other:
            return True
    return False


def find_evidence(bug_text: str, records: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """The evidence record that belongs to ``bug_text`` (the richest one if it was reported twice)."""
    key = normalize_bug_key(bug_text)
    if not key:
        return None
    best: Optional[Dict[str, str]] = None
    best_score = -1
    for rec in records:
        rec_key = normalize_bug_key(rec.get("bug", ""))
        if rec_key and (rec_key == key or is_dup_key(key, [rec_key])):
            score = sum(1 for name in ("observed", "expected", "observed_at") if rec.get(name))
            if score >= best_score:
                best, best_score = rec, score
    return best


# ---------------------------------------------------------
# Local validation of the contract (prompt examples are checked against it in tests)
# ---------------------------------------------------------

def validate_contract_entry(entry: Any) -> List[str]:
    """Problems found in one entry of a ``<bugs_found>`` block; an empty list means valid."""
    if not isinstance(entry, dict):
        return ["entry is not an object"]
    problems = [f"missing required field {name!r}" for name in _REQUIRED_FIELDS if not entry.get(name)]
    problems += [f"unknown field {name!r}" for name in entry if name not in CONTRACT_FIELDS]
    if "severity" in entry and entry["severity"] not in SEVERITIES:
        problems.append(f"severity must be one of {SEVERITIES}")
    if "confidence" in entry and entry["confidence"] not in CONFIDENCES:
        problems.append(f"confidence must be one of {CONFIDENCES}")
    problems += [
        f"{name!r} must be a string" for name in ("observed", "observed_at", "expected", "location")
        if name in entry and not isinstance(entry[name], str)
    ]
    return problems
