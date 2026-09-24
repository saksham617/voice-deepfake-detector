import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { AlertIcon, MessageIcon, PhoneIcon, WaveIcon } from "../components/icons";
import { TIER_STYLES } from "../components/VerdictCard";
import { ApiError, fetchReportsByPhoneNumber } from "../services/api";
import { verdictToTier, type BackendReport } from "../types/report";

type EventType = BackendReport["type"];

const TYPE_ICON: Record<EventType, typeof WaveIcon> = {
  voice: WaveIcon,
  message: MessageIcon,
  speaker_verification: PhoneIcon,
};

const TYPE_LABEL: Record<EventType, string> = {
  voice: "Voice check",
  message: "Message check",
  speaker_verification: "Contact check",
};

function summaryFor(report: BackendReport): string {
  if (report.type === "voice") {
    return report.verdict === "spoof" ? "AI-generated voice detected" : "Voice check — genuine";
  }
  if (report.type === "message") {
    return report.verdict === "suspicious" ? "Phishing message flagged" : "Message check — genuine";
  }
  return report.verdict === "safe" ? "Contact verified" : "Contact mismatch reported";
}

function detailFor(report: BackendReport): string {
  if (report.user_notes) return report.user_notes;
  return `${Math.round(report.confidence_score * 100)}% confidence.`;
}

function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function NumberTimelinePage() {
  const { phoneNumber = "" } = useParams<{ phoneNumber: string }>();
  const decoded = decodeURIComponent(phoneNumber);

  const [reports, setReports] = useState<BackendReport[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!decoded) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchReportsByPhoneNumber(decoded, controller.signal)
      .then((data) => {
        setReports(data);
        setLoading(false);
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof ApiError ? err.message : "Could not load this number's history.");
        setLoading(false);
      });
    return () => controller.abort();
  }, [decoded]);

  // Oldest first, so the timeline below reads top-to-bottom as it happened
  // (the backend returns newest-first, matching ReportsPage's list view).
  const events = [...reports].reverse();
  const highRiskCount = events.filter((r) => verdictToTier(r.verdict) === "high").length;
  const hasPattern = highRiskCount >= 2;

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <Link to="/reports" className="text-xs font-medium text-ink-dim hover:text-ink">
            ← Back to reports
          </Link>
          <h2 className="mt-2 text-xl font-semibold text-ink">{decoded || "Unknown number"}</h2>
          <p className="mt-1 text-sm text-ink-dim">
            Combined voice, message, and call activity for this number.
          </p>
        </header>

        {hasPattern && (
          <div className="mb-6 flex items-start gap-3 rounded-2xl border border-high/30 bg-high-dim px-5 py-4">
            <AlertIcon className="mt-0.5 h-5 w-5 shrink-0 text-high" />
            <div>
              <p className="text-sm font-semibold text-high">Coordinated attack pattern detected</p>
              <p className="mt-1 text-sm text-ink-dim">
                Multiple high-risk events from this number within a short window — a phishing
                message followed by a call is a common vishing setup.
              </p>
            </div>
          </div>
        )}

        {loading ? (
          <p className="py-8 text-center text-sm text-ink-faint">Loading history…</p>
        ) : error ? (
          <p className="py-8 text-center text-sm text-ink-faint">{error}</p>
        ) : events.length === 0 ? (
          <p className="py-8 text-center text-sm text-ink-faint">No activity recorded for this number yet.</p>
        ) : (
          <div className="relative space-y-4 pl-6">
            <div className="absolute bottom-2 left-[11px] top-2 w-px bg-line" />
            {events.map((report) => {
              const Icon = TYPE_ICON[report.type];
              const styles = TIER_STYLES[verdictToTier(report.verdict)];
              return (
                <div key={report.id} className="relative">
                  <span
                    className={`absolute -left-6 grid h-6 w-6 place-items-center rounded-full ${styles.chip} ${styles.text}`}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </span>
                  <div className="rounded-2xl border border-line bg-panel p-4">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-xs font-medium uppercase tracking-wider text-ink-faint">
                        {TYPE_LABEL[report.type]}
                      </span>
                      <span className="text-xs text-ink-faint">{formatDateTime(report.timestamp)}</span>
                    </div>
                    <p className="mt-1.5 text-sm font-medium text-ink">{summaryFor(report)}</p>
                    <p className="mt-1 text-xs text-ink-dim">{detailFor(report)}</p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
