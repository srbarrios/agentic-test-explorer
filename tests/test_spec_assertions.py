"""A generated spec asserts the real expected value when the bug has one, not just "text exists"."""

import os

import pytest

from agentic_explorer.tools.browser import engine

TAPE = [
    {"ts": 1.0, "action": "navigate", "params": {"url": "https://app.example/cart"}, "ok": True,
     "page_url": "https://app.example/cart"},
    {"ts": 2.0, "action": "extract_text", "params": {"selector": "[data-test=\"total\"]"}, "ok": True,
     "result": "$120.00", "page_url": "https://app.example/cart"},
]
GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "spec_with_expected.spec.ts")


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    engine._ACTION_TAPES.clear()


def _spec(tape=TAPE, **evidence):
    engine._ACTION_TAPES["t1"] = [dict(e) for e in tape]
    path = engine.generate_playwright_spec("t1", "Total ignores discount", app_url="https://app.example", **evidence)
    return open(path, encoding="utf-8").read()


def test_expected_value_becomes_a_real_assertion(workdir):
    content = _spec(expected="$96.00", observed="$120.00")
    assert "expect(_t).toContain('$96.00');" in content
    assert "expect(_t.length).toBeGreaterThan(0)" not in content
    assert "observed at record time: $120.00" in content


def test_observed_without_expected_stays_a_comment_and_keeps_the_weak_check(workdir):
    content = _spec(observed="$120.00")
    assert "expect(_t.length).toBeGreaterThan(0);" in content
    assert "observed at record time: $120.00" in content
    assert "no expected value recorded" in content
    assert "toContain" not in content


def test_without_evidence_the_spec_is_unchanged(workdir):
    content = _spec()
    assert "expect(_t.length).toBeGreaterThan(0);" in content
    assert "observed at record time" not in content


def test_only_the_last_extract_text_gets_the_assertion(workdir):
    tape = TAPE + [{"ts": 3.0, "action": "extract_text", "params": {"selector": "h1"}, "ok": True}]
    content = _spec(tape=tape, expected="Cart")
    assert content.count("expect(_t).toContain('Cart');") == 1
    assert content.count("expect(_t.length).toBeGreaterThan(0);") == 1
    assert content.index("toBeGreaterThan") < content.index("toContain")


def test_expected_with_quotes_and_newlines_is_escaped(workdir):
    content = _spec(expected="it's\n\"fine\" \\ ok", observed="a */ b\nc")
    assert "toContain('it\\'s\\n\"fine\" \\\\ ok')" in content
    assert "a * / b c" in content  # cannot close a comment, stays on one line


def test_failed_extract_text_is_not_asserted(workdir):
    tape = [{"ts": 1.0, "action": "extract_text", "params": {"selector": "h1"}, "ok": False, "error": "timeout"}]
    content = _spec(tape=tape, expected="Cart")
    assert "toContain('Cart')" not in content
    assert "expected: Cart" in content  # kept as a comment so the evidence is not lost


def test_no_extract_text_keeps_the_evidence_as_comments(workdir):
    tape = [{"ts": 1.0, "action": "click", "params": {"selector": "button"}, "ok": True}]
    content = _spec(tape=tape, expected="Saved", observed="nothing")
    assert "// expected: Saved" in content and "// observed at record time: nothing" in content


def test_spec_with_expected_matches_golden(workdir, monkeypatch):
    monkeypatch.setattr(engine.time, "strftime", lambda *_a, **_k: "2026-01-01 00:00:00")
    monkeypatch.setattr(engine.time, "time", lambda: 1_700_000_000)
    actual = _spec(expected="$96.00", observed="$120.00")
    if os.environ.get("UPDATE_GOLDEN"):
        open(GOLDEN, "w", encoding="utf-8").write(actual)
    assert actual == open(GOLDEN, encoding="utf-8").read()


async def test_the_agent_tool_passes_observed_and_expected_through(workdir):
    engine._ACTION_TAPES["t1"] = [dict(e) for e in TAPE]
    tool = engine.get_code_generator_tool(app_url="https://app.example")
    out = await tool.ainvoke(
        {"bug_summary": "Total ignores discount", "expected": "$96.00", "observed": "$120.00"},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert "Reproduction spec generated" in out
    spec_file = next((workdir / "report_t1").glob("reproduction_*.spec.ts"))
    assert "toContain('$96.00')" in spec_file.read_text()


async def test_the_agent_tool_still_works_with_only_a_summary(workdir):
    engine._ACTION_TAPES["t1"] = [dict(e) for e in TAPE]
    tool = engine.get_code_generator_tool()
    out = await tool.ainvoke({"bug_summary": "x"}, config={"configurable": {"thread_id": "t1"}})
    assert "Reproduction spec generated" in out
