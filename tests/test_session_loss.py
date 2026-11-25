"""An expired session is a first-class state: the mission stops cleanly instead of testing a login page."""

import json

import pytest

from agentic_explorer.tools.browser import engine, provenance, session


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session.reset()
    yield tmp_path
    session.reset()
    engine._ACTION_TAPES.clear()
    provenance._PROVENANCE.clear()


# --- the detector ------------------------------------------------------------------

@pytest.mark.parametrize("path, expected", [
    ("/login", True), ("/auth/sso/start", True), ("/users/sign-in", True), ("/signin/", True), ("/SSO", True),
    ("/", False), ("/inventory.html", False), ("/cart", False), ("/authors", False), ("/blogin", False),
])
def test_login_paths(path, expected):
    assert session.is_login_path(path) is expected


def _lost(**kw):
    base = dict(action="click", params={"selector": "a"}, landed_path="/", previous_path="/home",
                password_field=False, tape=[])
    base.update(kw)
    return session.looks_like_session_loss(**base)


def test_a_redirect_to_a_login_path_is_a_lost_session():
    assert _lost(landed_path="/login")


def test_a_password_form_after_a_page_change_is_a_lost_session_even_at_the_root():
    assert _lost(landed_path="/", previous_path="/inventory.html", password_field=True)
    assert _lost(action="navigate", params={"url": "https://app.test/"}, landed_path="/", previous_path=None, password_field=True)


def test_a_password_field_on_the_same_page_is_not():
    assert not _lost(landed_path="/settings", previous_path="/settings", password_field=True)
    assert not _lost(landed_path="/cart", previous_path="/cart", password_field=True)  # e.g. a confirm-password modal


@pytest.mark.parametrize("path", ["/settings/password", "/account/security", "/profile", "/register", "/sign-up",
                                  "/reset-password", "/forgot"])
def test_pages_about_passwords_or_accounts_are_not_a_lost_session(path):
    assert not _lost(landed_path=path, previous_path="/home", password_field=True)


def test_going_to_a_login_url_on_purpose_is_not_a_lost_session():
    assert not _lost(action="navigate", params={"url": "https://app.test/login"}, landed_path="/login")


def test_if_the_agent_already_typed_a_password_it_is_logging_in_itself():
    tape = [{"action": "fill", "params": {"selector": "#password", "value": "<redacted>", "redacted": True}}]
    assert not _lost(landed_path="/login", tape=tape)


def test_an_ordinary_page_is_not_a_lost_session():
    assert not _lost(landed_path="/cart", previous_path="/inventory.html")


# --- wired into the command tool ---------------------------------------------------------

class _Loc:
    def __init__(self, n):
        self._n = n

    async def count(self):
        return self._n


class _Page:
    def __init__(self, url, passwords=0):
        self.url = url
        self._passwords = passwords

    def locator(self, selector):
        return _Loc(self._passwords if "password" in selector else 0)


async def _command(page, command, monkeypatch, thread="t1"):
    async def fake_dispatch(_page, action, params):
        return f"{action} done"

    async def fake_snapshot(_page, limit=150):
        return {"url": page.url, "title": "T", "source": "dom", "elements": []}

    monkeypatch.setattr(engine, "_dispatch", fake_dispatch)
    monkeypatch.setattr(engine, "extract_dom_snapshot", fake_snapshot)
    tool = engine.get_browser_command_tool(page=page)
    return await tool.ainvoke({"command_json": json.dumps(command)}, config={"configurable": {"thread_id": thread}})


class _Control:
    def __init__(self):
        self.reasons = []

    def request_drain(self, reason):
        self.reasons.append(reason)


async def test_a_lost_session_stops_the_mission_and_tells_the_agent_why(monkeypatch):
    control = _Control()
    session.arm(expected=True, control=control)
    out = await _command(_Page("https://app.test/login"), {"action": "click", "selector": "#buy"}, monkeypatch)
    assert out.startswith("STATUS: AUTH_REQUIRED")
    assert "agent-auth" in out and "<bugs_found>[]</bugs_found>" in out
    assert control.reasons == ["auth_required"]
    [entry] = engine.get_action_tape("t1")
    assert entry["auth_required"] is True


async def test_nothing_happens_when_no_session_was_loaded(monkeypatch):
    control = _Control()
    session.arm(expected=False, control=control)
    out = await _command(_Page("https://app.test/login"), {"action": "navigate", "url": "https://app.test/login"}, monkeypatch)
    assert "AUTH_REQUIRED" not in out and control.reasons == []
    assert "auth_required" not in engine.get_action_tape("t1")[0]


async def test_a_normal_command_is_untouched(monkeypatch):
    control = _Control()
    session.arm(expected=True, control=control)
    out = await _command(_Page("https://app.test/cart"), {"action": "click", "selector": "#buy"}, monkeypatch)
    assert out.startswith("STATUS: OK") and control.reasons == []


async def test_a_password_form_after_navigating_to_the_root_is_caught(monkeypatch):
    control = _Control()
    session.arm(expected=True, control=control)
    out = await _command(_Page("https://app.test/", passwords=1), {"action": "navigate", "url": "https://app.test/"}, monkeypatch)
    assert "AUTH_REQUIRED" in out and control.reasons == ["auth_required"]


async def test_a_page_that_cannot_count_password_fields_does_not_break_the_command(monkeypatch):
    class Broken(_Page):
        def locator(self, selector):
            raise RuntimeError("page closed")

    session.arm(expected=True, control=_Control())
    out = await _command(Broken("https://app.test/cart"), {"action": "click", "selector": "#buy"}, monkeypatch)
    assert out.startswith("STATUS: OK")


async def test_the_session_is_not_checked_when_there_is_no_control_to_stop(monkeypatch):
    session.arm(expected=True, control=None)  # e.g. a standalone use of the engine
    out = await _command(_Page("https://app.test/login"), {"action": "click", "selector": "#buy"}, monkeypatch)
    assert "AUTH_REQUIRED" in out  # still reported to the agent, there is just nothing to drain


# --- re-authentication is recorded -------------------------------------------------------

def test_a_changed_auth_file_is_recorded_in_the_header_history(clean):
    (clean / "auth.json").write_text('{"cookies": []}')
    meta = provenance.write_provenance("t1", app_name="A", app_url="https://a.test")
    assert meta.get("auth_history", []) == []

    assert provenance.record_auth_if_changed("t1") is False  # same file: nothing to add

    (clean / "auth.json").write_text('{"cookies": [{"name": "sid"}]}')
    assert provenance.record_auth_if_changed("t1") is True
    history = provenance.load_provenance("t1")["auth_history"]
    assert len(history) == 1 and history[0]["sha256"] != meta["auth_state"]["sha256"]
    assert "SUPERSECRET" not in json.dumps(history)

    assert provenance.record_auth_if_changed("t1") is False  # already recorded
    on_disk = json.loads((clean / "report_t1" / "tape_meta.json").read_text())
    assert len(on_disk["auth_history"]) == 1
    assert on_disk["auth_state"]["sha256"] == meta["auth_state"]["sha256"]  # the first run's fingerprint is kept


def test_recording_auth_history_without_a_header_is_a_noop(clean):
    assert provenance.record_auth_if_changed("never-started") is False
