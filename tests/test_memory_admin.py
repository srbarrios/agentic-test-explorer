"""Tests for ``agentic_explorer.memory_admin`` (the --clear-* flags).

The real SQLite checkpoint saver stores state in ``checkpoints`` and ``writes``;
the long-term store uses ``store`` and ``store_vectors``.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from agentic_explorer.memory_admin import clear_memory

FIXTURE = Path(__file__).parent / "fixtures" / "checkpoint_langgraph_1_1_10.sqlite"


def _count(db: str, table: str) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def full_db(tmp_path) -> str:
    """A DB shaped like agent_memory.sqlite: checkpoint tables plus the store tables."""
    path = str(tmp_path / "agent_memory.sqlite")
    shutil.copy(FIXTURE, path)  # real `checkpoints` + `writes` written by langgraph
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE store (prefix TEXT, key TEXT, value TEXT);
        CREATE TABLE store_vectors (prefix TEXT, key TEXT, field_name TEXT);
        INSERT INTO store VALUES ('app', 'k1', '{}'), ('app', 'k2', '{}');
        INSERT INTO store_vectors VALUES ('app', 'k1', 'summary');
    """)
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------- everything

def test_everything_deletes_database_files_and_visual_state(tmp_path):
    db = tmp_path / "agent_memory.sqlite"
    state = tmp_path / ".agent_state.json"
    for f in (db, Path(f"{db}-wal"), Path(f"{db}-shm"), state):
        f.write_text("x")

    cleared = clear_memory(everything=True, db_path=str(db), state_files=[str(state)])

    assert not db.exists() and not Path(f"{db}-wal").exists() and not Path(f"{db}-shm").exists()
    assert not state.exists()
    assert sorted(cleared) == sorted([str(db), f"{db}-wal", f"{db}-shm", str(state)])


def test_everything_tolerates_missing_files(tmp_path):
    cleared = clear_memory(
        everything=True, db_path=str(tmp_path / "nope.sqlite"),
        state_files=[str(tmp_path / ".agent_state.json")],
    )
    assert cleared == []


# ---------------------------------------------------------------- selective

def test_checkpoints_only_keeps_learned_memory(full_db):
    assert _count(full_db, "checkpoints") > 0 and _count(full_db, "writes") > 0

    clear_memory(checkpoints=True, db_path=full_db)

    assert _count(full_db, "checkpoints") == 0
    # Regression: the old code targeted Postgres table names and never emptied this one.
    assert _count(full_db, "writes") == 0
    assert _count(full_db, "store") == 2
    assert _count(full_db, "store_vectors") == 1


def test_learned_only_keeps_checkpoints(full_db):
    checkpoints_before = _count(full_db, "checkpoints")
    writes_before = _count(full_db, "writes")

    clear_memory(learned=True, db_path=full_db)

    assert _count(full_db, "store") == 0
    # Regression: embeddings were left behind when only `store` was emptied.
    assert _count(full_db, "store_vectors") == 0
    assert _count(full_db, "checkpoints") == checkpoints_before
    assert _count(full_db, "writes") == writes_before


def test_both_flags_clear_both_kinds(full_db):
    clear_memory(checkpoints=True, learned=True, db_path=full_db)

    for table in ("checkpoints", "writes", "store", "store_vectors"):
        assert _count(full_db, table) == 0, table


def test_selective_clear_without_database_is_a_noop(tmp_path, capsys):
    cleared = clear_memory(checkpoints=True, db_path=str(tmp_path / "missing.sqlite"))

    assert cleared == []
    assert "No memory database to clear" in capsys.readouterr().out


def test_learned_without_store_table_warns(tmp_path, capsys):
    db = str(tmp_path / "only_checkpoints.sqlite")
    shutil.copy(FIXTURE, db)

    clear_memory(learned=True, db_path=db)

    assert "No store table found" in capsys.readouterr().out
    assert _count(db, "checkpoints") > 0


def test_no_flags_does_nothing(full_db, capsys):
    assert clear_memory(db_path=full_db) == []
    assert capsys.readouterr().out == ""
    assert _count(full_db, "store") == 2
