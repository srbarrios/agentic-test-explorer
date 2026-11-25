"""Tests for ``agentic_explorer.runtime.missions``: where missions come from.

Real ``AsyncSqliteStore`` files in a temp dir are used for the memory-backed
sources; the GitHub/LLM calls of the PR source are replaced with fakes.
"""

import json
from types import SimpleNamespace

import pytest
import yaml
from langgraph.store.sqlite import AsyncSqliteStore

from agentic_explorer import pr_analyzer
from agentic_explorer.cli import CliArgs
from agentic_explorer.config import AppMeta
from agentic_explorer.memory import app_url_hash, catalog_bug
from agentic_explorer.runtime import missions as missions_module
from agentic_explorer.runtime.missions import (
    export_model,
    load_missions_from_file,
    missions_from_pr,
    missions_from_regression,
    resolve_missions,
)

APP_URL = "https://app.test"


async def _seed_bug(db: str, *, summary="Cart total wrong", page="/cart", status=None) -> None:
    async with AsyncSqliteStore.from_conn_string(db) as store:
        await catalog_bug(store, app_url_hash(APP_URL), summary, page=page, session_id="s1")
        if status:
            ns = ("episodes", app_url_hash(APP_URL), "bugs")
            item = (await store.asearch(ns))[0]
            await store.aput(ns, item.key, {**item.value, "status": status})


# ---------------------------------------------------------------- YAML file

def test_load_missions_from_file(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.dump({"missions": [{"thread_id": "a", "prompt": "p"}]}), encoding="utf-8")

    assert load_missions_from_file(str(path)) == [{"thread_id": "a", "prompt": "p"}]


@pytest.mark.parametrize("content", ["", "other: 1\n", "missions: []\n"])
def test_load_missions_from_file_without_missions_is_empty(tmp_path, content):
    path = tmp_path / "m.yaml"
    path.write_text(content, encoding="utf-8")

    assert load_missions_from_file(str(path)) == []


# ---------------------------------------------------------------- regression

async def test_regression_without_database_returns_none(tmp_path, capsys):
    assert await missions_from_regression(APP_URL, str(tmp_path / "missing.sqlite")) is None
    assert "No memory database found" in capsys.readouterr().out


async def test_regression_builds_one_mission_per_buggy_page(tmp_path, capsys):
    db = str(tmp_path / "mem.sqlite")
    await _seed_bug(db, summary="Cart total wrong", page="/cart")

    missions = await missions_from_regression(APP_URL, db)

    assert [m["thread_id"] for m in missions] == ["regression_01_cart"]
    assert "Regression test for page '/cart'" in missions[0]["prompt"]
    assert "Cart total wrong" in missions[0]["prompt"]
    assert "Generated 1 regression missions" in capsys.readouterr().out


async def test_regression_with_only_fixed_bugs_is_empty_and_warns(tmp_path, capsys):
    db = str(tmp_path / "mem.sqlite")
    await _seed_bug(db, status="fixed")

    assert await missions_from_regression(APP_URL, db) == []
    assert "No open bugs found in catalog" in capsys.readouterr().out


# ---------------------------------------------------------------- export model

async def test_export_model_without_database_returns_false(tmp_path, capsys):
    out = tmp_path / "app_model.json"

    assert await export_model(APP_URL, str(tmp_path / "missing.sqlite"), str(out)) is False
    assert not out.exists()
    assert "No memory database found" in capsys.readouterr().out


async def test_export_model_writes_json_with_bugs(tmp_path, capsys):
    db = str(tmp_path / "mem.sqlite")
    out = tmp_path / "app_model.json"
    await _seed_bug(db)

    assert await export_model(APP_URL, db, str(out)) is True

    model = json.loads(out.read_text(encoding="utf-8"))
    assert model["url_hash"] == app_url_hash(APP_URL)
    assert model["bugs"][0]["page"] == "/cart"
    assert "Application model exported to" in capsys.readouterr().out


# ---------------------------------------------------------------- PR source

@pytest.fixture
def fake_pr(monkeypatch):
    seen = {}

    async def fetch(owner, repo, number, *, mcp_config_path=None):
        seen["fetch"] = (owner, repo, number, mcp_config_path)
        return SimpleNamespace(title="Add cart", files_changed=["a.py", "b.py"])

    async def generate(pr_data, app, store=None):
        seen["store"] = store
        return {"missions": [{"thread_id": "pr_7_new_user_01", "prompt": "check cart"}]}

    monkeypatch.setattr(pr_analyzer, "fetch_pr_data", fetch)
    monkeypatch.setattr(pr_analyzer, "generate_missions_from_pr", generate)
    return seen


async def test_pr_source_writes_yaml_and_returns_missions(tmp_path, fake_pr):
    out_dir = tmp_path / "out"

    missions = await missions_from_pr(
        "https://github.com/acme/shop/pull/7",
        app=AppMeta(name="Shop", url=APP_URL, description=""),
        mcp_config_path="mcp.json",
        output_dir=str(out_dir),
        db_path=str(tmp_path / "missing.sqlite"),
    )

    assert missions == [{"thread_id": "pr_7_new_user_01", "prompt": "check cart"}]
    assert yaml.safe_load((out_dir / "pr_7.yaml").read_text(encoding="utf-8")) == {"missions": missions}
    assert fake_pr["fetch"] == ("acme", "shop", 7, "mcp.json")
    assert fake_pr["store"] is None  # no memory yet, so no bug history is injected


async def test_pr_source_passes_the_store_when_a_database_exists(tmp_path, fake_pr):
    db = str(tmp_path / "mem.sqlite")
    await _seed_bug(db)

    await missions_from_pr(
        "https://github.com/acme/shop/pull/7",
        app=AppMeta(name="Shop", url=APP_URL, description=""),
        mcp_config_path=None,
        output_dir=str(tmp_path / "out"),
        db_path=db,
    )

    assert isinstance(fake_pr["store"], AsyncSqliteStore)


# ---------------------------------------------------------------- resolve_missions

FILE_MISSION = {"thread_id": "from_file", "prompt": "p"}
PR_MISSION = {"thread_id": "pr_7_new_user_01", "prompt": "p"}
REG_MISSION = {"thread_id": "regression_01_cart", "prompt": "p"}


def _cli(**overrides):
    base = dict(
        missions=None, pr_url=None, execute=False, output_dir="missions", headed=False,
        clear_memory=False, clear_checkpoints=False, clear_learned=False, clear_all=False,
        max_steps=30, provider=None, quiet=False, regression=False, export_model=False,
        visual=False,
    )
    base.update(overrides)
    return CliArgs(**base)


@pytest.fixture
def sources(monkeypatch, tmp_path):
    """Replace every source with a recorder; each returns what the test configures."""
    calls = []
    cfg = SimpleNamespace(app=AppMeta(name="Shop", url=APP_URL), paths=SimpleNamespace(mcp_servers="mcp.json"))
    config = {"file": [FILE_MISSION], "pr": [PR_MISSION], "regression": [REG_MISSION], "export": True}

    monkeypatch.setattr(missions_module, "load_missions_from_file", lambda path: calls.append("file") or list(config["file"]))

    async def pr(url, **kwargs):
        calls.append(("pr", url, kwargs["output_dir"], kwargs["mcp_config_path"]))
        return list(config["pr"])

    async def regression(app_url):
        calls.append("regression")
        return None if config["regression"] is None else list(config["regression"])

    async def export(app_url):
        calls.append("export")
        return config["export"]

    monkeypatch.setattr(missions_module, "missions_from_pr", pr)
    monkeypatch.setattr(missions_module, "missions_from_regression", regression)
    monkeypatch.setattr(missions_module, "export_model", export)
    monkeypatch.setattr(
        missions_module, "clear_memory", lambda **kw: calls.append(("clear", kw)) or [],
    )
    return SimpleNamespace(calls=calls, config=config, cfg=cfg)


async def test_resolve_file_only(sources):
    result = await resolve_missions(_cli(missions="m.yaml"), sources.cfg)

    assert result == [FILE_MISSION]
    assert sources.calls[0] == "file"


async def test_resolve_pr_without_execute_and_no_file_ends_the_run(sources, capsys):
    result = await resolve_missions(_cli(pr_url="https://github.com/a/b/pull/7"), sources.cfg)

    assert result is None
    assert "Use --execute to run generated missions" in capsys.readouterr().out
    assert not any(isinstance(c, tuple) and c[0] == "clear" for c in sources.calls)  # stopped before clearing


async def test_resolve_pr_without_execute_keeps_the_file_missions(sources):
    result = await resolve_missions(_cli(missions="m.yaml", pr_url="u"), sources.cfg)

    assert result == [FILE_MISSION]  # generated PR missions are saved but not run


async def test_resolve_pr_with_execute_appends_generated_missions(sources):
    result = await resolve_missions(_cli(missions="m.yaml", pr_url="u", execute=True), sources.cfg)

    assert result == [FILE_MISSION, PR_MISSION]
    assert ("pr", "u", "missions", "mcp.json") in sources.calls


async def test_resolve_clears_memory_with_the_requested_flags(sources):
    await resolve_missions(
        _cli(missions="m.yaml", clear_all=True, clear_checkpoints=True, clear_learned=False), sources.cfg,
    )

    assert ("clear", {"everything": True, "checkpoints": True, "learned": False}) in sources.calls


async def test_resolve_deprecated_clear_memory_flag_means_everything(sources):
    await resolve_missions(_cli(missions="m.yaml", clear_memory=True), sources.cfg)

    assert ("clear", {"everything": True, "checkpoints": False, "learned": False}) in sources.calls


async def test_resolve_export_without_database_ends_the_run_even_with_missions(sources):
    sources.config["export"] = False

    assert await resolve_missions(_cli(missions="m.yaml", export_model=True), sources.cfg) is None


async def test_resolve_export_alone_ends_after_exporting(sources):
    sources.config["file"] = []

    assert await resolve_missions(_cli(export_model=True), sources.cfg) is None
    assert "regression" not in sources.calls


async def test_resolve_export_then_continues_with_missions(sources):
    assert await resolve_missions(_cli(missions="m.yaml", export_model=True), sources.cfg) == [FILE_MISSION]


async def test_resolve_export_then_regression_runs_regression_missions(sources):
    sources.config["file"] = []

    assert await resolve_missions(_cli(export_model=True, regression=True), sources.cfg) == [REG_MISSION]


async def test_resolve_regression_without_database_ends_the_run(sources):
    sources.config["regression"] = None

    assert await resolve_missions(_cli(missions="m.yaml", regression=True), sources.cfg) is None


async def test_resolve_regression_appends_to_file_missions(sources):
    assert await resolve_missions(_cli(missions="m.yaml", regression=True), sources.cfg) == [FILE_MISSION, REG_MISSION]


async def test_resolve_with_no_missions_warns_and_ends(sources, capsys):
    sources.config["file"] = []

    assert await resolve_missions(_cli(missions="m.yaml"), sources.cfg) is None
    assert "No missions found. Exiting." in capsys.readouterr().out
