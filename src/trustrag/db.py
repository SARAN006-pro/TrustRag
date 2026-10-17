"""SQLite logging database for TrustRAG queries.

Logs every /ask request per SPEC section 9:
(ts, question, answer, grounded, risk_score, latency_ms).
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
    """Initialize the query_log table if it does not exist."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                grounded INTEGER NOT NULL,
                risk_score REAL NOT NULL,
                latency_ms INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def log_query(
    question: str,
    answer: str,
    grounded: bool,
    risk_score: float,
    latency_ms: int,
    db_path: Path = CONFIG.sqlite_path,
) -> int:
    """Log an answered query to the database and return the row id."""
    init_db(db_path)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO query_log (ts, question, answer, grounded, risk_score, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                question,
                answer,
                1 if grounded else 0,
                float(risk_score),
                int(latency_ms),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_recent_logs(limit: int = 50, db_path: Path = CONFIG.sqlite_path) -> list[dict[str, Any]]:
    """Retrieve recent query logs."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, ts, question, answer, grounded, risk_score, latency_ms
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
            }
            for r in rows
        ]


if __name__ == "__main__":
    init_db()
    log_query("What is PM-KISAN?", "PM-KISAN provides Rs 6,000.", True, 0.05, 120)
    print("Recent logs:", get_recent_logs(5))
