"""Unit tests for backend/scoring/session_report.py — the LEVEL->verdict mapping and the
per-session risk accumulator that turns a live-call session into a report row. Pure logic,
no FastAPI/torch."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scoring.risk_engine import RiskLevel, RiskState  # noqa: E402
from backend.scoring.session_report import (  # noqa: E402
    SessionRiskAccumulator,
    verdict_for_level,
)


def _state(level: RiskLevel, score: float) -> RiskState:
    return RiskState(score=score, level=level, raw_prob=score, consecutive_high=0,
                     alert=False, n_chunks=1)


def test_verdict_mapping_covers_all_levels():
    assert verdict_for_level(RiskLevel.HIGH) == "spoof"
    assert verdict_for_level(RiskLevel.MEDIUM) == "suspicious"
    assert verdict_for_level(RiskLevel.LOW) == "bonafide"
    assert verdict_for_level(RiskLevel.NONE) == "bonafide"


def test_empty_session_is_skipped():
    acc = SessionRiskAccumulator()
    assert acc.to_report_kwargs(duration_seconds=3.0) is None


def test_peak_level_drives_verdict_not_the_final_window():
    acc = SessionRiskAccumulator()
    acc.add(_state(RiskLevel.NONE, 0.10))
    acc.add(_state(RiskLevel.HIGH, 0.90))   # peak
    acc.add(_state(RiskLevel.LOW, 0.45))    # ends lower, but peak stays HIGH
    kwargs = acc.to_report_kwargs(duration_seconds=12.0)
    assert kwargs["verdict"] == "spoof"
    assert kwargs["confidence_score"] == 0.90  # peak rolling score
    assert kwargs["report_type"] == "voice"
    assert kwargs["phone_number"] is None
    assert "peak HIGH" in kwargs["user_notes"]


def test_all_clean_session_is_bonafide():
    acc = SessionRiskAccumulator()
    for _ in range(5):
        acc.add(_state(RiskLevel.NONE, 0.08))
    kwargs = acc.to_report_kwargs(duration_seconds=5.0)
    assert kwargs["verdict"] == "bonafide"


def test_medium_peak_is_suspicious():
    acc = SessionRiskAccumulator()
    acc.add(_state(RiskLevel.LOW, 0.45))
    acc.add(_state(RiskLevel.MEDIUM, 0.65))
    kwargs = acc.to_report_kwargs(duration_seconds=8.0)
    assert kwargs["verdict"] == "suspicious"
    assert kwargs["confidence_score"] == 0.65
