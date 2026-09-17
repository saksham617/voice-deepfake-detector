"""Tiny SQLite store for the live-call feature: the verified-phone profile + call history.

Deliberately a single-user demo store (one profile row, id='default') — there is no auth or
multi-tenant concept in this hackathon build. Mirrors src/models/reporting.py's sqlite style
(WAL, Row factory, idempotent schema). Lives in the gitignored data/processed/ dir.

  profiles: the user's phone number and whether it's been OTP-verified
  calls:    one row per inbound call — number, timestamps, and final verdict/confidence
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _REPO_ROOT / "data" / "processed" / "telephony.db"
_PROFILE_ID = "default"  # single-user demo

_db_ready = False


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_db_ready() -> None:
    global _db_ready
    if _db_ready:
        return
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                phone TEXT,
                phone_verified INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                from_number TEXT,
                to_number TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                status TEXT NOT NULL DEFAULT 'active',
                final_verdict TEXT,
                final_confidence REAL
            );
            """
        )
    _db_ready = True


def get_profile() -> dict:
    ensure_db_ready()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE id = ?", (_PROFILE_ID,)).fetchone()
    if row is None:
        return {"phone": None, "phone_verified": False}
    return {"phone": row["phone"], "phone_verified": bool(row["phone_verified"])}


def set_phone_verified(phone: str) -> dict:
    ensure_db_ready()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO profiles (id, phone, phone_verified, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET phone=excluded.phone, phone_verified=1,
                                          updated_at=excluded.updated_at
            """,
            (_PROFILE_ID, phone.strip(), time.time()),
        )
    return get_profile()


def record_call_start(call_id: str, from_number: str, to_number: str) -> None:
    ensure_db_ready()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO calls (id, from_number, to_number, started_at, status)
            VALUES (?, ?, ?, ?, 'active')
            ON CONFLICT(id) DO NOTHING
            """,
            (call_id, from_number, to_number, time.time()),
        )


def record_call_end(call_id: str, final_verdict: str, final_confidence: float) -> None:
    ensure_db_ready()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE calls SET ended_at=?, status='ended', final_verdict=?, final_confidence=?
            WHERE id=?
            """,
            (time.time(), final_verdict, float(final_confidence), call_id),
        )


def list_calls(limit: int = 100) -> list[dict]:
    ensure_db_ready()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM calls ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
