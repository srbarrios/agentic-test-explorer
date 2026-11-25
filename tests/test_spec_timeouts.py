"""A generated spec fails as fast as the recording did, and says when it asserts no symptom."""

import re
from pathlib import Path

import pytest

from agentic_explorer.tools.browser import engine

TAPE = [
    {"ts": 1.0, "action": "navigate", "params": {"url": "https://app.example/cart"}, "ok": True,
     "page_url": "https://app.example/cart"},
    {"ts": 2.0, "action": "click", "params": {"selector": "[data-test-subj='checkout']"}, "ok": False,
     "error": "TimeoutError: waiting for locator", "page_url": "https://app.example/cart"},
]
EXTRACT_TAPE = [
    {"ts": 1.0, "action": "navigate", "params": {"url": "https://app.example/cart"}, "ok": True},
    {"ts": 2.0, "action": "extract_text", "params": {"selector": "[data-test='total']"}, "ok": True, "result": "$120.00"},
]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    engine._ACTION_TAPES.clear()


def _spec(tape, **kwargs):
    engine._ACTION_TAPES["t1"] = [dict(e) for e in tape]
    return open(engine.generate_playwright_spec("t1", "Checkout missing", app_url="https://app.example", **kwargs),
                encoding="utf-8").read()


def test_the_spec_uses_the_same_timeouts_as_the_recording(workdir):
    content = _spec(TAPE)
    assert f"actionTimeout: {engine.ACTION_TIMEOUT_MS}" in content
    assert f"navigationTimeout: {engine.NAVIGATION_TIMEOUT_MS}" in content
    assert "storageState: 'auth.json'" in content


def test_the_recording_context_and_the_spec_share_one_source_of_truth():
    main_src = (Path(engine.__file__).parents[2] / "main.py").read_text(encoding="utf-8")
    assert "set_default_timeout(ACTION_TIMEOUT_MS)" in main_src
    assert "set_default_navigation_timeout(NAVIGATION_TIMEOUT_MS)" in main_src
    assert (engine.ACTION_TIMEOUT_MS, engine.NAVIGATION_TIMEOUT_MS) == (5000, 15000)  # what the engine always used


def test_a_failing_step_cannot_outlast_the_action_timeout():
    # The whole-test budget stays generous; it is the per-action timeout that bounds a missing element.
    assert engine.ACTION_TIMEOUT_MS < 10_000


def test_a_replay_only_spec_warns_that_passing_does_not_mean_fixed(workdir):
    content = _spec(TAPE)
    assert "test.info().annotations.push({ type: 'warning'" in content
    assert "passing does NOT mean the bug is fixed" in content
    head = content.split("import {")[0]
    assert "no symptom assertion" in head.lower()


def test_a_spec_with_an_expected_value_does_not_warn(workdir):
    content = _spec(EXTRACT_TAPE, expected="$96.00", observed="$120.00")
    assert "toContain('$96.00')" in content
    assert "annotations.push" not in content
    assert "no symptom assertion" not in content.lower()


def test_expected_without_a_step_to_assert_against_still_warns(workdir):
    content = _spec(TAPE, expected="Checkout visible")
    assert "// expected: Checkout visible" in content
    assert "annotations.push" in content


def test_observed_alone_keeps_the_weak_check_and_warns(workdir):
    content = _spec(EXTRACT_TAPE, observed="$120.00")
    assert "toBeGreaterThan(0)" in content and "annotations.push" in content


def test_the_annotation_sits_inside_the_test_before_any_step(workdir):
    content = _spec(TAPE)
    body = content.split("async ({ page }) => {")[1]
    assert body.index("annotations.push") < body.index("--- BUG STEPS ---")
    assert re.search(r"test\.setTimeout\(120_000\);", content)
