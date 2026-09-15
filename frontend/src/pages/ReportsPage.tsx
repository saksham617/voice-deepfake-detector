import { useState } from "react";
import { Link } from "react-router-dom";
import { MessageIcon, PhoneIcon, WaveIcon } from "../components/icons";
import { TIER_STYLES, type RiskTier } from "../components/VerdictCard";

type ReportType = "voice" | "message";
type ReportStatus = "open" | "reviewed" | "dismissed";

interface Report {
  id: string;
  date: string; // ISO
  type: ReportType;
  tier: RiskTier;
  summary: string;
  phoneNumber?: string;
  notes: string;
  status: ReportStatus;
}

// TODO(remove-when-backend-ready): replace with a real /reports endpoint once
// the reporting API is live. Same local-mock pattern as OverviewPage's
// RECENT_ACTIVITY.
const REPORTS: Report[] = [
  {
    id: "r1",
    date: "2026-09-14T09:12:00.000Z",
    type: "voice",
    tier: "high",
    summary: "call_recording_014.mp3 — AI-generated voice detected",
    phoneNumber: "+1 202-555-0143",
    notes: "Caller impersonated a bank representative and requested an OTP.",
    status: "open",
  },
  {
    id: "r2",
    date: "2026-09-13T18:40:00.000Z",
    type: "message",
    tier: "high",
    summary: "\"Your account will be suspended, verify now…\"",
    phoneNumber: "+1 202-555-0143",
    notes: "SMS phishing link disguised as a bank verification request.",
    status: "reviewed",
  },
  {
    id: "r3",
    date: "2026-09-12T11:05:00.000Z",
    type: "voice",
    tier: "medium",
    summary: "recording_08b.wav — inconclusive result",
    notes: "Borderline confidence score; flagged for manual review.",
    status: "reviewed",
  },
  {
    id: "r4",
    date: "2026-09-10T15:20:00.000Z",
    type: "message",
    tier: "medium",
    summary: "\"Limited time offer, click here to claim…\"",
    phoneNumber: "+1 415-555-0199",
    notes: "Promotional spam with a suspicious link shortener.",
    status: "dismissed",
  },
  {
    id: "r5",
    date: "2026-09-08T08:02:00.000Z",
    type: "voice",
    tier: "safe",
    summary: "voiceprint_check.wav — genuine",
    notes: "Verified against an enrolled contact with high similarity.",
    status: "reviewed",
  },
];

const STATUS_LABEL: Record<ReportStatus, string> = {
  open: "Open",
  reviewed: "Reviewed",
  dismissed: "Dismissed",
};

const STATUS_DOT: Record<ReportStatus, string> = {
  open: "bg-high",
  reviewed: "bg-safe",
  dismissed: "bg-ink-faint",
};

const TIER_LABEL: Record<RiskTier, string> = {
  safe: "Safe",
  medium: "Medium",
  high: "High risk",
};

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ReportsPage() {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h2 className="text-xl font-semibold text-ink">Reports</h2>
          <p className="mt-1 text-sm text-ink-dim">
            Past voice-check and message-check reports.
          </p>
        </header>

        <div className="overflow-hidden rounded-2xl border border-line bg-panel">
          <div className="hidden items-center gap-3 border-b border-line bg-panel-raised px-5 py-3 text-[11px] font-medium uppercase tracking-wider text-ink-faint sm:grid sm:grid-cols-[110px_90px_1fr_100px_100px]">
            <span>Date</span>
            <span>Type</span>
            <span>Details</span>
            <span>Verdict</span>
            <span>Status</span>
          </div>

          {REPORTS.map((report) => {
            const expanded = expandedId === report.id;
            const styles = TIER_STYLES[report.tier];
            const TypeIcon = report.type === "voice" ? WaveIcon : MessageIcon;

            return (
              <div key={report.id} className="border-b border-line last:border-0">
                <button
                  type="button"
                  onClick={() => setExpandedId(expanded ? null : report.id)}
                  aria-expanded={expanded}
                  className="grid w-full grid-cols-1 items-center gap-2 px-5 py-3 text-left text-sm transition-colors hover:bg-panel-raised sm:grid-cols-[110px_90px_1fr_100px_100px] sm:gap-3"
                >
                  <span className="text-xs text-ink-dim sm:text-sm">{formatDate(report.date)}</span>
                  <span className="flex items-center gap-1.5 text-xs text-ink-dim sm:text-sm">
                    <TypeIcon className="h-3.5 w-3.5" />
                    {report.type === "voice" ? "Voice" : "Message"}
                  </span>
                  <span className="truncate text-ink" title={report.summary}>
                    {report.summary}
                  </span>
                  <span
                    className={`inline-flex w-fit items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${styles.chip} ${styles.text}`}
                  >
                    {TIER_LABEL[report.tier]}
                  </span>
                  <span className="flex items-center gap-1.5 text-xs text-ink-dim">
                    <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[report.status]}`} />
                    {STATUS_LABEL[report.status]}
                  </span>
                </button>

                {expanded && (
                  <div className="border-t border-line bg-panel-raised px-5 py-4 text-sm">
                    <p className="text-ink-dim">{report.notes}</p>
                    {report.phoneNumber && (
                      <Link
                        to={`/reports/number/${encodeURIComponent(report.phoneNumber)}`}
                        className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-safe hover:underline"
                      >
                        <PhoneIcon className="h-3.5 w-3.5" />
                        View timeline for {report.phoneNumber}
                      </Link>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
