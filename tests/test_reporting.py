"""Unit tests for src/models/reporting.py, exercised directly (no FastAPI,
no ML models) to prove this module stands entirely on its own -- it's a
plain SQLite-backed logging store that any detection feature can write
results into once it exists, not something that depends on the CNN spoof
detector or the ECAPA-TDNN speaker verifier being loaded or even installed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import reporting


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point reporting at a fresh temp DB per test, instead of the real
    data/processed/reports.db -- keeps tests from leaving mock data behind
    or colliding with each other. Resets the lazy _db_ready flag too, since
    it's keyed on process lifetime, not on DB_PATH."""
    monkeypatch.setattr(reporting, "DB_PATH", tmp_path / "test_reports.db")
    monkeypatch.setattr(reporting, "_db_ready", False)


def test_module_has_no_dependency_on_any_ml_model():
    """Importing reporting.py alone must never pull in torch/speechbrain --
    that's the whole point of it being a standalone logging system. Runs in
    a fresh subprocess so an unrelated test module having already imported
    torch earlier in the session can't hide a real dependency here."""
    script = (
        "import sys; "
        "from src.models import reporting; "
        "assert 'torch' not in sys.modules, 'reporting.py pulled in torch'; "
        "assert 'speechbrain' not in sys.modules, 'reporting.py pulled in speechbrain'; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=_PROJECT_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_create_report_returns_stored_row_with_id_and_timestamp():
    report = reporting.create_report(
        report_type="voice", verdict="spoof", confidence_score=0.92, claimed_identity="Boss"
    )
    assert report["id"] is not None
    assert report["timestamp"]
    assert report["type"] == "voice"
    assert report["verdict"] == "spoof"
    assert report["confidence_score"] == 0.92
    assert report["claimed_identity"] == "Boss"
    assert report["user_notes"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"report_type": "banana", "verdict": "spoof", "confidence_score": 0.5},
        {"report_type": "voice", "verdict": "banana", "confidence_score": 0.5},
        {"report_type": "voice", "verdict": "spoof", "confidence_score": 1.5},
        {"report_type": "voice", "verdict": "spoof", "confidence_score": -0.1},
    ],
)
def test_create_report_rejects_invalid_fields(kwargs):
    with pytest.raises(reporting.InvalidReportError):
        reporting.create_report(**kwargs)


def test_list_reports_sorted_most_recent_first():
    first = reporting.create_report(report_type="voice", verdict="spoof", confidence_score=0.9)
    second = reporting.create_report(report_type="message", verdict="suspicious", confidence_score=0.6)
    third = reporting.create_report(
        report_type="speaker_verification", verdict="safe", confidence_score=0.8
    )

    listed_ids = [r["id"] for r in reporting.list_reports()]
    assert listed_ids == [third["id"], second["id"], first["id"]]


def test_get_report_returns_matching_row():
    created = reporting.create_report(report_type="voice", verdict="bonafide", confidence_score=0.7)
    fetched = reporting.get_report(created["id"])
    assert fetched == created


def test_get_report_raises_for_missing_id():
    with pytest.raises(reporting.ReportNotFoundError):
        reporting.get_report(99999)


def test_delete_report_removes_it_and_subsequent_get_raises():
    created = reporting.create_report(report_type="voice", verdict="spoof", confidence_score=0.5)
    reporting.delete_report(created["id"])

    with pytest.raises(reporting.ReportNotFoundError):
        reporting.get_report(created["id"])
    assert created["id"] not in [r["id"] for r in reporting.list_reports()]


def test_delete_report_raises_for_missing_id():
    with pytest.raises(reporting.ReportNotFoundError):
        reporting.delete_report(99999)
