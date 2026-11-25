"""Tape provenance: where a recording came from, so a SETUP failure is not mistaken for a bug."""

import hashlib
import json
import os

import pytest

from agentic_explorer.cli import parse_args
from agentic_explorer.config import AppMeta, load_app_config
from agentic_explorer.tools.browser import engine, provenance


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("APP_REVISION", raising=False)
    yield tmp_path
    engine._ACTION_TAPES.clear()
    provenance._PROVENANCE.clear()


def _meta(**kw):
    base = dict(app_name="Shop", app_url="https://app.example", app_revision=None, seed=())
    base.update(kw)
    return base


# --- revision resolution -----------------------------------------------------

def test_revision_priority_cli_then_env_then_config(monkeypatch):
    monkeypatch.setenv("APP_REVISION", "env-rev")
    assert provenance.resolve_app_revision("cli-rev", "cfg-rev") == "cli-rev"
    assert provenance.resolve_app_revision(None, "cfg-rev") == "env-rev"
    monkeypatch.delenv("APP_REVISION")
    assert provenance.resolve_app_revision(None, "cfg-rev") == "cfg-rev"


def test_revision_is_none_when_nobody_gives_one(monkeypatch):
    monkeypatch.delenv("APP_REVISION", raising=False)
    assert provenance.resolve_app_revision(None, None) is None
    assert provenance.resolve_app_revision("  ", "") is None


def test_cli_accepts_app_revision():
    assert parse_args(["--missions", "m.yaml", "--app-revision", "abc123"]).app_revision == "abc123"
    assert parse_args(["--missions", "m.yaml"]).app_revision is None


def test_config_reads_revision_and_seed(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "app:\n  url: https://a.example\n  revision: v1.2.3\n  seed:\n    - user qa@example.com exists\n    - cart is empty\n"
    )
    app = load_app_config(cfg_file).app
    assert app.revision == "v1.2.3"
    assert app.seed == ("user qa@example.com exists", "cart is empty")


def test_config_seed_accepts_a_single_string(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("app:\n  url: https://a.example\n  seed: one precondition\n")
    assert load_app_config(cfg_file).app.seed == ("one precondition",)


def test_appmeta_defaults_keep_old_configs_valid():
    app = AppMeta()
    assert app.revision is None and app.seed == ()


# --- the header --------------------------------------------------------------

def test_header_has_expected_fields_and_null_revision(workdir):
    (workdir / "auth.json").write_text('{"cookies": [{"name": "sid", "value": "SUPERSECRET"}]}')
    meta = provenance.write_provenance("t1", **_meta(seed=("cart is empty",)))
    assert meta["type"] == "header"
    assert meta["thread_id"] == "t1"
    assert meta["app_url"] == "https://app.example" and meta["app_name"] == "Shop"
    assert meta["app_revision"] is None
    assert meta["seed"] == ["cart is empty"]
    assert meta["started_at"]
    assert meta["tool_version"]
    assert meta["auth_state"]["sha256"] == hashlib.sha256((workdir / "auth.json").read_bytes()).hexdigest()
    assert meta["auth_state"]["modified"]


def test_header_never_contains_auth_content(workdir):
    (workdir / "auth.json").write_text('{"cookies": [{"name": "sid", "value": "SUPERSECRET"}]}')
    provenance.write_provenance("t1", **_meta())
    on_disk = (workdir / "report_t1" / "tape_meta.json").read_text()
    assert "SUPERSECRET" not in on_disk and "cookies" not in on_disk


def test_missing_auth_file_is_recorded_as_absent(workdir):
    meta = provenance.write_provenance("t1", **_meta())
    assert meta["auth_state"] == {"path": "auth.json", "sha256": None, "modified": None}


def test_header_is_written_once_and_resume_keeps_the_original(workdir):
    first = provenance.write_provenance("t1", **_meta(app_revision="r1"))
    again = provenance.write_provenance("t1", **_meta(app_revision="r2"))
    assert again == first and again["app_revision"] == "r1"
    assert json.loads((workdir / "report_t1" / "tape_meta.json").read_text())["app_revision"] == "r1"


def test_header_does_not_touch_the_tape_file(workdir):
    provenance.write_provenance("t1", **_meta())
    assert not (workdir / "report_t1" / "action_tape.jsonl").exists()
    engine._append_tape("t1", {"ts": 1.0, "action": "navigate", "params": {"url": "https://app.example"}, "ok": True})
    lines = (workdir / "report_t1" / "action_tape.jsonl").read_text().splitlines()
    assert [json.loads(line)["action"] for line in lines] == ["navigate"]


def test_load_provenance_reads_disk_after_restart(workdir):
    provenance.write_provenance("t1", **_meta(app_revision="r1"))
    provenance._PROVENANCE.clear()
    assert provenance.load_provenance("t1")["app_revision"] == "r1"
    assert provenance.load_provenance("nope") is None


# --- the generated spec ------------------------------------------------------

TAPE = [
    {"ts": 1.0, "action": "navigate", "params": {"url": "https://app.example/home"}, "ok": True,
     "page_url": "https://app.example/home"},
    {"ts": 2.0, "action": "click", "params": {"selector": "[data-test-subj='save']"}, "ok": True,
     "page_url": "https://app.example/home"},
]


def _spec(workdir, tape=TAPE, with_header=True, **meta_kw):
    engine._ACTION_TAPES["t1"] = [dict(e) for e in tape]
    if with_header:
        (workdir / "auth.json").write_text("{}")
        provenance.write_provenance("t1", **_meta(**meta_kw))
    path = engine.generate_playwright_spec("t1", "Save fails", app_url="https://app.example")
    return open(path, encoding="utf-8").read()


def test_spec_carries_provenance_in_its_header(workdir):
    content = _spec(workdir, app_revision="abc123", seed=("cart is empty",))
    head = content.split("import {")[0]
    assert "App revision: abc123" in head
    assert "auth.json sha256:" in head
    assert "cart is empty" in head
    assert "Session recorded:" in head


def test_spec_says_revision_was_not_recorded_when_null(workdir):
    assert "App revision: not recorded" in _spec(workdir)


def test_spec_separates_setup_check_from_bug_steps(workdir):
    content = _spec(workdir)
    assert "SETUP_FAILURE" in content
    setup = content.index("SETUP CHECK")
    steps = content.index("// --- BUG STEPS ---")
    assert setup < steps < content.index("await page.click(")
    assert content.index("await page.goto('https://app.example/home')") > steps


def test_spec_setup_check_ignores_login_pages_the_recording_visited(workdir):
    tape = [{"ts": 1.0, "action": "navigate", "params": {"url": "https://app.example/login"}, "ok": True,
             "page_url": "https://app.example/login"}]
    content = _spec(workdir, tape=tape)
    assert '"/login"' in content  # recorded path is allow-listed


def test_old_tape_without_header_still_generates_a_valid_spec(workdir):
    content = _spec(workdir, with_header=False)
    assert "App revision: not recorded" in content
    assert "await page.click(" in content
    assert "// --- BUG STEPS ---" in content
    assert "import { test, expect } from '@playwright/test';" in content


def test_spec_matches_golden(workdir, monkeypatch):
    monkeypatch.setattr(engine.time, "strftime", lambda *_a, **_k: "2026-01-01 00:00:00")
    monkeypatch.setattr(engine.time, "time", lambda: 1_700_000_000)
    monkeypatch.setattr(provenance, "_now_iso", lambda: "2026-01-01T00:00:00+00:00")
    (workdir / "auth.json").write_text("{}")
    os.utime(workdir / "auth.json", (1_700_000_000, 1_700_000_000))
    engine._ACTION_TAPES["t1"] = [dict(e) for e in TAPE]
    provenance.write_provenance("t1", **_meta(app_revision="abc123", seed=("cart is empty",)))
    path = engine.generate_playwright_spec("t1", "Save fails", app_url="https://app.example")
    actual = open(path, encoding="utf-8").read()
    golden = os.path.join(os.path.dirname(__file__), "fixtures", "spec_with_provenance.spec.ts")
    if os.environ.get("UPDATE_GOLDEN"):
        open(golden, "w", encoding="utf-8").write(actual)
    assert actual == open(golden, encoding="utf-8").read()


# --- wiring into the mission runner ------------------------------------------

async def test_runner_writes_the_header_once_per_mission(workdir):
    from tests.test_runner import FINAL, STEP, FakeGraph, _run  # reuse the fake graph and report LLM

    prov = _meta(app_revision="abc123")
    await _run(FakeGraph([[STEP]], final_values=FINAL), provenance=prov)
    meta = json.loads((workdir / "report_test_new_user_01" / "tape_meta.json").read_text())
    assert meta["app_revision"] == "abc123" and meta["type"] == "header"

    await _run(FakeGraph([[STEP]], final_values=FINAL), provenance=_meta(app_revision="other"))
    meta = json.loads((workdir / "report_test_new_user_01" / "tape_meta.json").read_text())
    assert meta["app_revision"] == "abc123"  # a resumed mission keeps its first header


async def test_runner_without_provenance_writes_no_header(workdir):
    from tests.test_runner import FINAL, STEP, FakeGraph, _run

    await _run(FakeGraph([[STEP]], final_values=FINAL))
    assert not (workdir / "report_test_new_user_01" / "tape_meta.json").exists()
