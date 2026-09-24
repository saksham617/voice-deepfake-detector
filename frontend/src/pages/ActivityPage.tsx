import { useEffect, useState } from "react";
import { ContactsIcon, MessageIcon, WaveIcon } from "../components/icons";
import { ApiError, fetchReports } from "../services/api";
import { reportToActivityItem, type ActivityItem, type ActivityKind } from "../utils/activity";
import type { RiskTier as ActivityTier } from "../components/VerdictCard";

const KIND_ICON: Record<ActivityKind, typeof WaveIcon> = {
  voice: WaveIcon,
  message: MessageIcon,
  contact: ContactsIcon,
};

const KIND_LABEL: Record<ActivityKind, string> = {
  voice: "Voice",
  message: "Message",
  contact: "Contact",
};

const TIER_STYLE: Record<ActivityTier, { chip: string; text: string }> = {
  safe: { chip: "bg-safe-dim", text: "text-safe" },
  medium: { chip: "bg-medium-dim", text: "text-medium" },
  high: { chip: "bg-high-dim", text: "text-high" },
};

const FILTERS = ["all", "voice", "message", "contact"] as const;
type Filter = (typeof FILTERS)[number];

const FILTER_LABEL: Record<Filter, string> = {
  all: "All",
  voice: "Voice",
  message: "Message",
  contact: "Contacts",
};

export function ActivityPage() {
  const [filter, setFilter] = useState<Filter>("all");
  const [allActivity, setAllActivity] = useState<ActivityItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchReports(controller.signal)
      .then((reports) => setAllActivity(reports.map(reportToActivityItem)))
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof ApiError ? err.message : "Could not load activity.");
      });
    return () => controller.abort();
  }, []);

  const items = allActivity.filter((item) => filter === "all" || item.kind === filter);

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-3xl">
        <header className="mb-6">
          <h2 className="text-xl font-semibold text-ink">Activity</h2>
          <p className="mt-1 text-sm text-ink-dim">
            Full history of voice checks, message checks, and contact verifications.
          </p>
        </header>

        <div className="mb-5 inline-flex rounded-xl border border-line bg-panel p-1">
          {FILTERS.map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition-colors ${
                filter === f ? "bg-panel-raised text-ink" : "text-ink-dim hover:text-ink"
              }`}
            >
              {FILTER_LABEL[f]}
            </button>
          ))}
        </div>

        <div className="rounded-2xl border border-line bg-panel p-5">
          {error ? (
            <p className="py-8 text-center text-sm text-ink-faint">{error}</p>
          ) : items.length === 0 ? (
            <p className="py-8 text-center text-sm text-ink-faint">
              No {FILTER_LABEL[filter].toLowerCase()} activity yet.
            </p>
          ) : (
            <ul className="space-y-3">
              {items.map((item) => {
                const Icon = KIND_ICON[item.kind];
                const styles = TIER_STYLE[item.tier];
                return (
                  <li
                    key={item.id}
                    className="flex items-start gap-3 border-b border-line pb-3 last:border-0 last:pb-0"
                  >
                    <span
                      className={`mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full ${styles.chip} ${styles.text}`}
                    >
                      <Icon className="h-4 w-4" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <p className="truncate text-sm text-ink">{item.title}</p>
                        <span className="shrink-0 rounded-full border border-line px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-ink-faint">
                          {KIND_LABEL[item.kind]}
                        </span>
                      </div>
                      <p className="truncate text-xs text-ink-dim">{item.detail}</p>
                    </div>
                    <span className="shrink-0 text-xs text-ink-faint">{item.timestamp}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
