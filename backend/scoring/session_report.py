"""Turn a finished live-call session's risk history into a report row.

Pure and synchronous (no I/O, no FastAPI) so it's unit-testable on its own; the WebSocket
handler feeds it each ``RiskState`` as scores arrive and, when the call ends, asks it for the
``reporting.create_report`` kwargs. Kept separate from websocket.py so the LEVEL->verdict
mapping is easy to find and adjust.

Verdict mapping (RiskLevel -> reports schema verdict), decided for this project:
    HIGH   -> "spoof"       (sustained high fake-probability)
    MEDIUM -> "suspicious"
    LOW    -> "bonafide"    (mild elevation, still treated as genuine)
    NONE   -> "bonafide"    (below the LOW threshold)

confidence_score is the session's PEAK rolling risk score, i.e. peak fake-probability in
[0,1] — high for spoof, low for bonafide (it measures fake-likelihood, not verdict-certainty).
"""

from __future__ import annotations

from backend.scoring.risk_engine import RiskLevel, RiskState

# Highest-severity-wins ordering for tracking a session's peak level.
_LEVEL_ORDER: dict[RiskLevel, int] = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
}

_VERDICT_BY_LEVEL: dict[RiskLevel, str] = {
    RiskLevel.HIGH: "spoof",
    RiskLevel.MEDIUM: "suspicious",
    RiskLevel.LOW: "bonafide",
    RiskLevel.NONE: "bonafide",
}


def verdict_for_level(level: RiskLevel) -> str:
    """Map a risk level to a reports-schema verdict. The single place to adjust this."""
    return _VERDICT_BY_LEVEL[level]


class SessionRiskAccumulator:
    """Accumulates the per-window risk states of one live-call session."""

    def __init__(self) -> None:
        self.scored_windows = 0
        self.peak_score = 0.0
        self.peak_level = RiskLevel.NONE
        self.level_counts: dict[RiskLevel, int] = {lvl: 0 for lvl in RiskLevel}

    def add(self, state: RiskState) -> None:
        self.scored_windows += 1
        self.peak_score = max(self.peak_score, state.score)
        self.level_counts[state.level] += 1
        if _LEVEL_ORDER[state.level] > _LEVEL_ORDER[self.peak_level]:
            self.peak_level = state.level

    def to_report_kwargs(
        self, duration_seconds: float, phone_number: str | None = None
    ) -> dict | None:
        """Build reporting.create_report kwargs for this session, or None if it should be
        skipped (no window was ever scored — e.g. a call that connected and immediately
        dropped, or was pure non-speech that the VAD filtered out)."""
        if self.scored_windows == 0:
            return None

        counts = "  ".join(
            f"{lvl.value}:{self.level_counts[lvl]}"
            for lvl in (RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)
            if self.level_counts[lvl]
        )
        notes = (
            f"Live call: {duration_seconds:.0f}s, {self.scored_windows} windows scored, "
            f"peak {self.peak_level.value}. Windows per level: {counts}."
        )
        return {
            "report_type": "voice",
            "verdict": verdict_for_level(self.peak_level),
            "confidence_score": round(self.peak_score, 4),
            "user_notes": notes,
            "phone_number": phone_number,
        }
