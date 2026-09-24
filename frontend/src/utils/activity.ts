/**
 * Maps a backend report (GET /reports) into the activity-feed shape shown on
 * OverviewPage and ActivityPage. One mapping, used by both, so the two pages
 * can't drift into describing the same report differently.
 */

import type { RiskTier } from "../components/VerdictCard";
import { verdictToTier, type BackendReport } from "../types/report";
import { formatRelativeTime } from "./time";

export type ActivityKind = "voice" | "message" | "contact";

export interface ActivityItem {
  id: string;
  kind: ActivityKind;
  title: string;
  detail: string;
  tier: RiskTier;
  timestamp: string;
}

const KIND_BY_REPORT_TYPE: Record<BackendReport["type"], ActivityKind> = {
  voice: "voice",
  message: "message",
  speaker_verification: "contact",
};

function titleFor(report: BackendReport): string {
  switch (report.type) {
    case "voice":
      return report.verdict === "spoof" ? "Voice check — AI-generated" : "Voice check — genuine";
    case "message":
      return report.verdict === "suspicious" ? "Message check — flagged" : "Message check — genuine";
    case "speaker_verification":
      return report.verdict === "safe" ? "Contact verified" : "Contact mismatch reported";
    default:
      return "Check completed";
  }
}

function detailFor(report: BackendReport): string {
  if (report.user_notes) return report.user_notes;
  if (report.claimed_identity) return `Claimed identity: ${report.claimed_identity}`;
  return `${Math.round(report.confidence_score * 100)}% confidence`;
}

export function reportToActivityItem(report: BackendReport): ActivityItem {
  return {
    id: String(report.id),
    kind: KIND_BY_REPORT_TYPE[report.type],
    title: titleFor(report),
    detail: detailFor(report),
    tier: verdictToTier(report.verdict),
    timestamp: formatRelativeTime(report.timestamp),
  };
}
