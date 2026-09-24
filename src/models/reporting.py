"""Report storage for the voice deepfake detector, backed by SQLite.

A standalone logging/history store, independent of every detection model in
this project (the CNN spoof detector in src/models/inference.py, the ECAPA-
TDNN speaker verifier in src/models/speaker_verification.py, and any future
IndicWav2Vec/AASIST work) -- nothing here imports torch or any model code.
Any feature can log a result into it once it produces one; this module has
no opinion on how that result was produced.

SQLite (not Postgres) is a deliberate choice for a hackathon prototype: one
file on disk, zero setup, good enough for a single-process demo backend.
"""

import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Lives alongside speaker_verification.py's enrollment store
# (data/processed/speaker_embeddings/); data/processed/* is gitignored
# except for explicit exceptions (see .gitignore), so this runtime database
# is never committed.
DB_PATH = _PROJECT_ROOT / "data" / "processed" / "reports.db"

REPORT_TYPES = {"voice", "message", "speaker_verification"}
VERDICTS = {"spoof", "bonafide", "suspicious", "safe"}
# Review workflow states shown in the frontend Reports table. New reports start "open".
STATUSES = {"open", "reviewed", "dismissed"}
DEFAULT_STATUS = "open"

# Module-level flag so CREATE TABLE IF NOT EXISTS only runs once per process
# -- mirrors the _cached_model lazy-init pattern in speaker_verification.py,
# just for a DB handle instead of a loaded model.
_db_ready = False


class InvalidReportError(ValueError):
    """Raised when report fields fail validation (bad type/verdict/confidence_score)."""


class ReportNotFoundError(Exception):
    """Raised by get_report()/delete_report() when the given id doesn't exist."""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_db_ready() -> None:
    """Create the reports table if it doesn't already exist. Idempotent and
    safe to call from every CRUD function below as well as app startup."""
    global _db_ready
    if _db_ready:
        return
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                claimed_identity TEXT,
                user_notes TEXT,
                phone_number TEXT,
                status TEXT NOT NULL DEFAULT 'open'
            )
            """
        )
        # Migrate a pre-existing reports.db created before phone_number/status existed:
        # CREATE TABLE IF NOT EXISTS won't add columns to an already-present table, so add
        # any missing column in place (idempotent, preserves existing rows).
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(reports)")}
        if "phone_number" not in existing:
            conn.execute("ALTER TABLE reports ADD COLUMN phone_number TEXT")
        if "status" not in existing:
            conn.execute("ALTER TABLE reports ADD COLUMN status TEXT NOT NULL DEFAULT 'open'")
    _db_ready = True


def _validate_report_fields(
    report_type: str, verdict: str, confidence_score: float, status: str = DEFAULT_STATUS
) -> None:
    if report_type not in REPORT_TYPES:
        raise InvalidReportError(
            f"Invalid type '{report_type}'. Must be one of: {sorted(REPORT_TYPES)}"
        )
    if verdict not in VERDICTS:
        raise InvalidReportError(f"Invalid verdict '{verdict}'. Must be one of: {sorted(VERDICTS)}")
    if not isinstance(confidence_score, (int, float)) or isinstance(confidence_score, bool):
        raise InvalidReportError(f"confidence_score must be a number, got {confidence_score!r}")
    if not (0.0 <= confidence_score <= 1.0):
        raise InvalidReportError(f"confidence_score must be between 0 and 1, got {confidence_score}")
    if status not in STATUSES:
        raise InvalidReportError(f"Invalid status '{status}'. Must be one of: {sorted(STATUSES)}")


def create_report(
    report_type: str,
    verdict: str,
    confidence_score: float,
    claimed_identity: str | None = None,
    user_notes: str | None = None,
    phone_number: str | None = None,
    status: str = DEFAULT_STATUS,
) -> dict:
    """Validate and insert a new report. Returns the stored report (including
    its assigned id and server-set timestamp) as a dict.

    ``phone_number`` is optional caller/contact metadata (the live-call WS has no
    caller ID today, so auto-created call reports leave it None); ``status`` is the
    review-workflow state, defaulting to "open".

    Raises InvalidReportError if report_type/verdict/status aren't in the allowed
    sets above, or confidence_score isn't in [0, 1].
    """
    _validate_report_fields(report_type, verdict, confidence_score, status)
    ensure_db_ready()

    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO reports
                (timestamp, type, verdict, confidence_score, claimed_identity,
                 user_notes, phone_number, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, report_type, verdict, confidence_score, claimed_identity,
             user_notes, phone_number, status),
        )
        report_id = cursor.lastrowid

    return get_report(report_id)


def list_reports() -> list[dict]:
    """Return all reports, most recently created first."""
    ensure_db_ready()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM reports ORDER BY timestamp DESC, id DESC").fetchall()
    return [dict(row) for row in rows]


def list_reports_by_phone_number(phone_number: str) -> list[dict]:
    """Return every report against one phone number, most recently created
    first -- the per-number timeline shown on the Reports > Number page."""
    ensure_db_ready()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM reports WHERE phone_number = ? ORDER BY timestamp DESC, id DESC",
            (phone_number,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_report(report_id: int) -> dict:
    """Return one report by id. Raises ReportNotFoundError if it doesn't exist."""
    ensure_db_ready()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    if row is None:
        raise ReportNotFoundError(f"No report found with id {report_id}.")
    return dict(row)


def delete_report(report_id: int) -> None:
    """Delete a report by id. Raises ReportNotFoundError if it doesn't exist."""
    ensure_db_ready()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        deleted_count = cursor.rowcount
    if deleted_count == 0:
        raise ReportNotFoundError(f"No report found with id {report_id}.")
