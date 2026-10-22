"""SQLite database for TrustRAG query logging and workspace management.

V2 features:
- Workspaces table (id, name, created_at, privacy)
- Privacy support: 'private' | 'shared'
- Non-destructive migration of existing v1 data into 'default' workspace
- Query log table with workspace_id tracking
"""
from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Any

from trustrag.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_connection(db_path: Path = CONFIG.sqlite_path) -> sqlite3.Connection:
    """Return an SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Path = CONFIG.sqlite_path) -> None:
    """Initialize workspaces and query_log tables with migrations."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        # 1. Workspaces table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                privacy TEXT NOT NULL CHECK(privacy IN ('private', 'shared'))
            )
            """
        )

        # 2. Query log table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                grounded INTEGER NOT NULL,
                risk_score REAL NOT NULL,
                latency_ms INTEGER NOT NULL,
                workspace_id TEXT DEFAULT 'default'
            )
            """
        )

        # Non-destructive column check for query_log.workspace_id
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(query_log)")
        cols = [r["name"] for r in cursor.fetchall()]
        if "workspace_id" not in cols:
            conn.execute("ALTER TABLE query_log ADD COLUMN workspace_id TEXT DEFAULT 'default'")

        # 3. Seed default workspace if not present
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR IGNORE INTO workspaces (id, name, created_at, privacy)
            VALUES ('default', 'Default Schemes Workspace', ?, 'shared')
            """,
            (now_iso,),
        )
        conn.commit()


def create_workspace(
    id: str,
    name: str,
    privacy: str = "shared",
    db_path: Path = CONFIG.sqlite_path,
) -> dict[str, Any]:
    """Create a new workspace."""
    init_db(db_path)
    ws_id = id.strip()
    if not ws_id:
        raise ValueError("Workspace id cannot be empty.")
    if privacy not in ("private", "shared"):
        raise ValueError("Privacy must be either 'private' or 'shared'.")

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO workspaces (id, name, created_at, privacy)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                privacy = excluded.privacy
            """,
            (ws_id, name.strip(), now_iso, privacy),
        )
        conn.commit()

    return {
        "id": ws_id,
        "name": name.strip(),
        "created_at": now_iso,
        "privacy": privacy,
    }


def get_workspace(id: str, db_path: Path = CONFIG.sqlite_path) -> dict[str, Any] | None:
    """Retrieve workspace details by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, created_at, privacy FROM workspaces WHERE id = ?",
            (id.strip(),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def list_workspaces(db_path: Path = CONFIG.sqlite_path) -> list[dict[str, Any]]:
    """List all configured workspaces."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, created_at, privacy FROM workspaces ORDER BY created_at ASC")
        return [dict(r) for r in cursor.fetchall()]


def delete_workspace(id: str, db_path: Path = CONFIG.sqlite_path) -> bool:
    """Delete a workspace record. Prevents deleting 'default'."""
    init_db(db_path)
    if id == "default":
        raise ValueError("Cannot delete the default workspace.")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM workspaces WHERE id = ?", (id.strip(),))
        conn.commit()
        return cursor.rowcount > 0


def log_query(
    question: str,
    answer: str,
    grounded: bool,
    risk_score: float,
    latency_ms: int,
    workspace_id: str = "default",
    db_path: Path = CONFIG.sqlite_path,
) -> int:
    """Log an answered query to the database and return the row id."""
    init_db(db_path)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO query_log (ts, question, answer, grounded, risk_score, latency_ms, workspace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                question,
                answer,
                1 if grounded else 0,
                float(risk_score),
                int(latency_ms),
                workspace_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_recent_logs(limit: int = 50, workspace_id: str | None = None, db_path: Path = CONFIG.sqlite_path) -> list[dict[str, Any]]:
    """Retrieve recent query logs, optionally filtered by workspace."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if workspace_id:
            cursor.execute(
                """
                SELECT id, ts, question, answer, grounded, risk_score, latency_ms, workspace_id
                FROM query_log
                WHERE workspace_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (workspace_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, ts, question, answer, grounded, risk_score, latency_ms, workspace_id
                FROM query_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "question": r["question"],
                "answer": r["answer"],
                "grounded": bool(r["grounded"]),
                "risk_score": r["risk_score"],
                "latency_ms": r["latency_ms"],
                "workspace_id": r["workspace_id"] if "workspace_id" in r.keys() else "default",
            }
            for r in rows
        ]


if __name__ == "__main__":
    init_db()
    print("Workspaces:", list_workspaces())
    print("Recent logs:", get_recent_logs(2))
