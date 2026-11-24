"""Administration of the ``agent_memory.sqlite`` file behind the ``--clear-*`` flags.

``memory.py`` owns reading/writing memories through the LangGraph store; this
module only empties or deletes the SQLite file and its sidecars.
"""

from __future__ import annotations

import os
import sqlite3
from typing import List, Sequence

from agentic_explorer.utils import console

DEFAULT_DB = "agent_memory.sqlite"
# Written by Visual Mode; removed together with the database.
STATE_FILES = (".agent_state.json", ".agent_state.json.tmp", ".latest_vision.jpg")

# AsyncSqliteSaver tables (mission state, resumable by thread_id).
CHECKPOINT_TABLES = ("checkpoints", "writes")
# AsyncSqliteStore tables (learned memory); vectors reference the store rows.
STORE_VECTOR_TABLE = "store_vectors"
STORE_TABLE = "store"


def clear_memory(
    *,
    everything: bool = False,
    checkpoints: bool = False,
    learned: bool = False,
    db_path: str = DEFAULT_DB,
    state_files: Sequence[str] = STATE_FILES,
) -> List[str]:
    """Delete memory according to the flags and return what was removed.

    ``everything`` deletes the database (plus ``-wal``/``-shm``) and the Visual
    Mode state files. Otherwise ``checkpoints`` empties mission state (keeping
    learned memory) and ``learned`` empties the long-term store (keeping
    checkpoints); both may be combined. With no flag nothing happens.
    """
    if everything:
        return _delete_everything(db_path, state_files)
    if checkpoints or learned:
        return _clear_selectively(db_path, checkpoints=checkpoints, learned=learned)
    return []


def _delete_everything(db_path: str, state_files: Sequence[str]) -> List[str]:
    console.section("Clearing all memory")
    deleted: List[str] = []
    for path in (db_path, f"{db_path}-wal", f"{db_path}-shm", *state_files):
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)
            console.info(f"Deleted {path}")
    return deleted


def _clear_selectively(db_path: str, *, checkpoints: bool, learned: bool) -> List[str]:
    if not os.path.exists(db_path):
        console.info("No memory database to clear")
        return []

    console.section("Selective memory clearing")
    cleared: List[str] = []
    conn = sqlite3.connect(db_path)
    try:
        if checkpoints:
            for table in CHECKPOINT_TABLES:
                if _empty_table(conn, table):
                    cleared.append(table)
                    console.info(f"Cleared table: {table}")
            conn.commit()
            console.success("Checkpoints cleared (learned memory preserved)")
        if learned:
            # Embeddings first: they reference store rows.
            if _empty_table(conn, STORE_VECTOR_TABLE):
                cleared.append(STORE_VECTOR_TABLE)
            if _empty_table(conn, STORE_TABLE):
                cleared.append(STORE_TABLE)
                conn.commit()
                console.success("Learned memory cleared (checkpoints preserved)")
            else:
                console.warn("No store table found")
    finally:
        conn.close()
    return cleared


def _empty_table(conn: sqlite3.Connection, table: str) -> bool:
    """DELETE every row of ``table``; False if the table does not exist."""
    try:
        conn.execute(f"DELETE FROM {table}")
    except sqlite3.OperationalError:
        return False
    return True
