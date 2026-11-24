"""Tests for ``agentic_explorer.cli`` (flags and their validation)."""

import pytest

from agentic_explorer import cli
from agentic_explorer.cli import CliArgs, parse_args


def test_defaults_with_a_missions_file():
    args = parse_args(["--missions", "m.yaml"])

    assert args == CliArgs(
        missions="m.yaml", pr_url=None, execute=False, output_dir="missions",
        headed=False, clear_memory=False, clear_checkpoints=False, clear_learned=False,
        clear_all=False, max_steps=30, provider=None, quiet=False, regression=False,
        export_model=False, visual=False,
    )


def test_all_flags_are_parsed():
    args = parse_args([
        "--pr-url", "https://github.com/o/r/pull/7", "--execute", "--output-dir", "out",
        "--headed", "--max-steps", "50", "--provider", "claude", "--quiet",
        "--clear-checkpoints", "--clear-learned",
    ])

    assert args.pr_url == "https://github.com/o/r/pull/7"
    assert args.execute and args.headed and args.quiet
    assert args.output_dir == "out"
    assert args.max_steps == 50
    assert args.provider == "claude"
    assert args.clear_checkpoints and args.clear_learned
    assert not args.clear_all and not args.clear_memory


@pytest.mark.parametrize("source", [["--regression"], ["--export-model"]])
def test_regression_and_export_model_are_valid_sources(source):
    assert parse_args(source) is not None


@pytest.mark.parametrize("argv", [[], ["--headed"], ["--clear-all"], ["--execute", "--quiet"]])
def test_a_source_is_required(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(argv)

    assert exit_info.value.code == 2
    assert "At least one of --missions, --pr-url, --regression, or --export-model" in capsys.readouterr().err


def test_unknown_provider_is_rejected(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--missions", "m.yaml", "--provider", "openai"])

    assert "invalid choice" in capsys.readouterr().err


def test_visual_requires_streamlit(monkeypatch, capsys):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit):
        parse_args(["--missions", "m.yaml", "--visual"])

    assert "--visual requires Streamlit" in capsys.readouterr().err


def test_visual_is_accepted_when_streamlit_is_installed(monkeypatch):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())

    assert parse_args(["--missions", "m.yaml", "--visual"]).visual is True
