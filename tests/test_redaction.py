"""Typed secrets never reach the tape, the generated spec or the trace log."""

import json

import pytest
from langchain_core.messages import AIMessage

from agentic_explorer.runtime.tracing import TraceWriter
from agentic_explorer.tools.browser import engine, redaction

SECRET = "hunter2-correct-horse"


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    redaction.forget_all()
    yield tmp_path
    redaction.forget_all()
    engine._ACTION_TAPES.clear()


# --- deciding what is sensitive ------------------------------------------------

@pytest.mark.parametrize("selector", [
    "input[type='password']", "#password", "[data-test='login-password']", "input[name=pwd]",
    "[aria-label='API key']", "#otp-code", "input.auth-token", "[name='cvv']",
])
def test_sensitive_selectors_are_recognised(selector):
    assert redaction.is_sensitive_selector(selector)


@pytest.mark.parametrize("selector", ["#user-name", "input[name='email']", "[data-test='username']", "#search", ""])
def test_ordinary_selectors_are_not(selector):
    assert not redaction.is_sensitive_selector(selector)


def test_fill_values_of_sensitive_fields_are_redacted_and_remembered():
    out = redaction.redact_params("fill", {"selector": "#password", "value": SECRET})
    assert out == {"selector": "#password", "value": redaction.REDACTED, "redacted": True}
    assert redaction.scrub(f"typed {SECRET} here") == f"typed {redaction.REDACTED} here"


def test_the_element_type_alone_is_enough():
    out = redaction.redact_params("fill", {"selector": "#field-7", "value": SECRET}, element_is_password=True)
    assert out["value"] == redaction.REDACTED


def test_ordinary_fills_and_other_actions_are_untouched():
    params = {"selector": "#user-name", "value": "standard_user"}
    assert redaction.redact_params("fill", params) == params
    assert redaction.redact_params("click", {"selector": "#password"}) == {"selector": "#password"}
    assert redaction.redact_params("fill", {"selector": "#password", "value": ""}) == {"selector": "#password", "value": ""}


def test_the_original_params_are_not_mutated():
    params = {"selector": "#password", "value": SECRET}
    redaction.redact_params("fill", params)
    assert params["value"] == SECRET


def test_very_short_secrets_are_not_scrubbed_from_text():
    redaction.redact_params("fill", {"selector": "#password", "value": "ab"})
    assert redaction.scrub("about a cab") == "about a cab"


# --- the tape ---------------------------------------------------------------------

class _Page:
    async def eval_on_selector(self, selector, js):
        return "password" if "pw" in selector else "text"


async def _run(page, command, monkeypatch):
    async def fake_dispatch(_page, action, params):
        return f"{action} done"

    async def fake_snapshot(_page, limit=150):
        return {"url": "https://app.example/login", "title": "Login", "source": "dom", "elements": []}

    monkeypatch.setattr(engine, "_dispatch", fake_dispatch)
    monkeypatch.setattr(engine, "extract_dom_snapshot", fake_snapshot)
    tool = engine.get_browser_command_tool(page=page)
    return await tool.ainvoke({"command_json": json.dumps(command)}, config={"configurable": {"thread_id": "t1"}})


async def test_a_typed_password_never_reaches_the_tape_or_the_tool_output(clean, monkeypatch):
    dispatched = {}

    async def spy_dispatch(_page, action, params):
        dispatched.update(params)
        return "filled"

    async def fake_snapshot(_page, limit=150):
        return {"url": "u", "title": "t", "source": "dom", "elements": []}

    monkeypatch.setattr(engine, "_dispatch", spy_dispatch)
    monkeypatch.setattr(engine, "extract_dom_snapshot", fake_snapshot)
    tool = engine.get_browser_command_tool(page=_Page())
    out = await tool.ainvoke(
        {"command_json": json.dumps({"action": "fill", "selector": "#password", "value": SECRET})},
        config={"configurable": {"thread_id": "t1"}},
    )

    assert dispatched["value"] == SECRET  # the browser still receives the real value
    [entry] = engine.get_action_tape("t1")
    assert entry["params"]["value"] == redaction.REDACTED and entry["params"]["redacted"] is True
    assert SECRET not in out
    assert SECRET not in (clean / "report_t1" / "action_tape.jsonl").read_text()


async def test_an_unrecognised_selector_is_caught_by_the_element_type(clean, monkeypatch):
    await _run(_Page(), {"action": "fill", "selector": "#pw-field", "value": SECRET}, monkeypatch)
    [entry] = engine.get_action_tape("t1")
    assert entry["params"]["value"] == redaction.REDACTED


async def test_an_ordinary_fill_is_recorded_as_is(clean, monkeypatch):
    await _run(_Page(), {"action": "fill", "selector": "#user-name", "value": "standard_user"}, monkeypatch)
    [entry] = engine.get_action_tape("t1")
    assert entry["params"]["value"] == "standard_user" and "redacted" not in entry["params"]


async def test_a_page_that_cannot_report_the_element_type_does_not_break_the_command(clean, monkeypatch):
    class Broken:
        async def eval_on_selector(self, *_a):
            raise RuntimeError("no such element")

    await _run(Broken(), {"action": "fill", "selector": "#user-name", "value": "x"}, monkeypatch)
    assert len(engine.get_action_tape("t1")) == 1


# --- the generated spec --------------------------------------------------------------

def _tape_with_login():
    return [
        {"ts": 1.0, "action": "navigate", "params": {"url": "https://app.example/login"}, "ok": True, "page_url": "https://app.example/login"},
        {"ts": 2.0, "action": "fill", "params": {"selector": "#user-name", "value": "standard_user"}, "ok": True},
        {"ts": 3.0, "action": "fill", "params": {"selector": "#password", "value": redaction.REDACTED, "redacted": True}, "ok": True},
        {"ts": 4.0, "action": "fill", "params": {"selector": "#otp", "value": redaction.REDACTED, "redacted": True}, "ok": True},
    ]


def test_spec_reads_redacted_values_from_the_environment(clean):
    engine._ACTION_TAPES["t1"] = _tape_with_login()
    path = engine.generate_playwright_spec("t1", "Login breaks", app_url="https://app.example")
    content = open(path, encoding="utf-8").read()
    assert "await page.fill('#password', process.env.REPLAY_SECRET_1 ?? '');" in content
    assert "await page.fill('#otp', process.env.REPLAY_SECRET_2 ?? '');" in content
    assert "await page.fill('#user-name', 'standard_user');" in content
    assert f"'{redaction.REDACTED}'" not in content  # the placeholder is never typed into a field
    assert "REPLAY_SECRET_1" in content.split("import {")[0]  # the header says how to supply it


def test_a_spec_without_redacted_values_has_no_secret_instructions(clean):
    engine._ACTION_TAPES["t1"] = [{"ts": 1.0, "action": "click", "params": {"selector": "button"}, "ok": True}]
    content = open(engine.generate_playwright_spec("t1", "x"), encoding="utf-8").read()
    assert "REPLAY_SECRET" not in content


# --- the trace log ----------------------------------------------------------------------

def test_the_trace_log_scrubs_remembered_secrets(clean):
    redaction.redact_params("fill", {"selector": "#password", "value": SECRET})
    msg = AIMessage(content="", tool_calls=[{"name": "execute_browser_command", "id": "c1",
                    "args": {"command_json": json.dumps({"action": "fill", "selector": "#password", "value": SECRET})}}])
    writer = TraceWriter(str(clean / "traces.log"))
    writer.message("agent", msg)
    writer.node_update("agent", [AIMessage(content=f"I typed {SECRET}", id="m2")])
    text = (clean / "traces.log").read_text()
    assert SECRET not in text and redaction.REDACTED in text
