"""Pins how a mission ``thread_id`` selects the standard or advanced graph.

``run_missions`` decides with ``any(kw in thread_id.lower() for kw in
ADVANCED_KEYWORDS)``. The substring match is a known footgun (see the
``explorer`` case below); a later task replaces it with an explicit field.
"""

import pytest

from agentic_explorer.orchestration.personas import ADVANCED_KEYWORDS


def _is_advanced(thread_id: str) -> bool:
    return any(kw in thread_id.lower() for kw in ADVANCED_KEYWORDS)


def test_keyword_list_is_pinned():
    assert ADVANCED_KEYWORDS == (
        "accessibility", "a11y", "data_heavy", "data-heavy", "impatient",
        "returning", "explorer", "chaos", "autonomous",
    )


@pytest.mark.parametrize("thread_id", [
    "test_new_user_01",
    "test_power_user_03",
    "test_adversarial_02",
    "pr_123_new_user_01",
])
def test_standard_thread_ids(thread_id):
    assert _is_advanced(thread_id) is False


@pytest.mark.parametrize("thread_id", [
    "test_accessibility_01",
    "test_A11Y_keyboard",  # case-insensitive
    "test_data_heavy_01",
    "test_data-heavy_02",
    "test_impatient_01",
    "test_returning_01",
    "test_explorer_01",
    "pr_123_explorer_01",
])
def test_advanced_thread_ids(thread_id):
    assert _is_advanced(thread_id) is True


def test_substring_match_misroutes_standard_mission_mentioning_a_keyword():
    # Documents current behaviour: a *new-user* mission whose id merely contains
    # "explorer" is sent to the advanced graph.
    assert _is_advanced("test_new_user_explorer_notes") is True
