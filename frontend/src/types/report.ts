/**
 * Report types + backend<->UI mapping.
 *
 * The backend (backend/api/legacy.py ReportResponse) is the source of truth. ReportsPage's
 * UI needs a `tier` (safe|medium|high) it doesn't store server-side, so we derive it from
 * `verdict` here rather than duplicating a column. This is the "frontend does the mapping"
 * reconciliation, keeping the backend schema lean.
 */

import type { RiskTier } from "../components/VerdictCard";

/** A report exactly as the backend returns it (GET /reports). */
export interface BackendReport {
  id: number;
  timestamp: string; // ISO 8601
  type: "voice" | "message" | "speaker_verification";
  verdict: "spoof" | "bonafide" | "suspicious" | "safe";
  confidence_score: number; // 0-1
  claimed_identity: string | null;
  user_notes: string | null;
  phone_number: string | null;
  status: "open" | "reviewed" | "dismissed";
}

/** Derive the UI risk tier from the backend verdict. */
export function verdictToTier(verdict: BackendReport["verdict"]): RiskTier {
  switch (verdict) {
    case "spoof":
      return "high";
    case "suspicious":
      return "medium";
    default: // bonafide | safe
      return "safe";
  }
}

/** Runtime validation of a GET /reports payload (an array of reports). */
export function isBackendReportArray(value: unknown): value is BackendReport[] {
  return (
    Array.isArray(value) &&
    value.every(
      (r) =>
        r != null &&
        typeof r === "object" &&
        typeof (r as BackendReport).id === "number" &&
        typeof (r as BackendReport).verdict === "string" &&
        typeof (r as BackendReport).type === "string",
    )
  );
}
