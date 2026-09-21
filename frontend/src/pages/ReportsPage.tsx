import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { MessageIcon, PhoneIcon, WaveIcon } from "../components/icons";
import { TIER_STYLES, type RiskTier } from "../components/VerdictCard";
import { deleteReport, fetchReports } from "../services/api";
import { verdictToTier, type BackendReport } from "../types/report";

const STATUS_LABEL: Record<BackendReport["status"], string> = {
  open: "Open",
  reviewed: "Reviewed",
  dismissed: "Dismissed",
};

const STATUS_DOT: Record<BackendReport["status"], string> = {
  open: "bg-high",
  reviewed: "bg-safe",
  dismissed: "bg-ink-faint",
};

const TIER_LABEL: Record<RiskTier, string> = {
  safe: "Safe",
  medium: "Medium",
  high: "High risk",
};

const VERDICT_SUMMARY: Record<BackendReport["verdict"], string> = {
  spoof: "AI-generated voice detected",
  suspicious: "Suspicious — flagged for review",
  bonafide: "Genuine voice",
  safe: "No threat detected",
};

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function typeLabel(type: BackendReport["type"]): string {
  return type === "message" ? "Message" : "Voice";
}

/** The "Details" text: prefer the report's own notes, else a verdict-derived summary. */
function summaryOf(r: BackendReport): string {
  return r.user_notes?.trim() || VERDICT_SUMMARY[r.verdict];
}

export function ReportsPage() {
  const [reports, setReports] = useState<BackendReport[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchReports(controller.signal)
      .then((data) => setReports(data))
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err?.message ?? "Could not load reports.");
        setReports([]);
      });
    return () => controller.abort();
  }, []);

  async function handleDelete(id: number) {
    if (!window.confirm("Delete this report? This can't be undone.")) return;
    setDeletingId(id);
    try {
      await deleteReport(id);
      setReports((prev) => (prev ? prev.filter((r) => r.id !== id) : prev));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete the report.");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h2 className="text-xl font-semibold text-ink">Reports</h2>
          <p className="mt-1 text-sm text-ink-dim">
            Past live-call, voice-check and message-check reports.
          </p>
        </header>

        {error && (
          <div className="mb-4 rounded-xl border border-high/40 bg-high/10 px-4 py-3 text-sm text-high">
            {error}
          </div>
        )}

        <div className="overflow-hidden rounded-2xl border border-line bg-panel">
          <div className="hidden items-center gap-3 border-b border-line bg-panel-raised px-5 py-3 text-[11px] font-medium uppercase tracking-wider text-ink-faint sm:grid sm:grid-cols-[110px_90px_1fr_100px_100px]">
            <span>Date</span>
            <span>Type</span>
            <span>Details</span>
            <span>Verdict</span>
            <span>Status</span>
          </div>

          {reports === null ? (
            <p className="px-5 py-8 text-center text-sm text-ink-dim">Loading reports…</p>
          ) : reports.length === 0 ? (
            <p className="px-5 py-8 text-center text-sm text-ink-dim">No reports yet.</p>
          ) : (
            reports.map((report) => {
              const expanded = expandedId === report.id;
              const tier = verdictToTier(report.verdict);
              const styles = TIER_STYLES[tier];
              const TypeIcon = report.type === "message" ? MessageIcon : WaveIcon;

              return (
                <div key={report.id} className="border-b border-line last:border-0">
                  <button
                    type="button"
                    onClick={() => setExpandedId(expanded ? null : report.id)}
                    aria-expanded={expanded}
                    className="grid w-full grid-cols-1 items-center gap-2 px-5 py-3 text-left text-sm transition-colors hover:bg-panel-raised sm:grid-cols-[110px_90px_1fr_100px_100px] sm:gap-3"
                  >
                    <span className="text-xs text-ink-dim sm:text-sm">{formatDate(report.timestamp)}</span>
                    <span className="flex items-center gap-1.5 text-xs text-ink-dim sm:text-sm">
                      <TypeIcon className="h-3.5 w-3.5" />
                      {typeLabel(report.type)}
                    </span>
                    <span className="truncate text-ink" title={summaryOf(report)}>
                      {summaryOf(report)}
                    </span>
                    <span
                      className={`inline-flex w-fit items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${styles.chip} ${styles.text}`}
                    >
                      {TIER_LABEL[tier]}
                    </span>
                    <span className="flex items-center gap-1.5 text-xs text-ink-dim">
                      <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[report.status]}`} />
                      {STATUS_LABEL[report.status]}
                    </span>
                  </button>

                  {expanded && (
                    <div className="border-t border-line bg-panel-raised px-5 py-4 text-sm">
                      <div className="flex flex-wrap gap-x-6 gap-y-1 text-ink-dim">
                        <span>
                          Confidence:{" "}
                          <span className="font-medium text-ink">
                            {Math.round(report.confidence_score * 100)}%
                          </span>
                        </span>
                        {report.claimed_identity && (
                          <span>
                            Claimed identity:{" "}
                            <span className="font-medium text-ink">{report.claimed_identity}</span>
                          </span>
                        )}
                      </div>

                      <div className="mt-3 flex items-center justify-between gap-4">
                        {report.phone_number ? (
                          <Link
                            to={`/reports/number/${encodeURIComponent(report.phone_number)}`}
                            className="inline-flex items-center gap-1.5 text-sm font-medium text-safe hover:underline"
                          >
                            <PhoneIcon className="h-3.5 w-3.5" />
                            View timeline for {report.phone_number}
                          </Link>
                        ) : (
                          <span className="text-xs text-ink-faint">No caller number recorded.</span>
                        )}

                        <button
                          type="button"
                          onClick={() => handleDelete(report.id)}
                          disabled={deletingId === report.id}
                          className="rounded-lg border border-high/40 px-3 py-1.5 text-xs font-medium text-high transition-colors hover:bg-high/10 disabled:opacity-50"
                        >
                          {deletingId === report.id ? "Deleting…" : "Delete report"}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
