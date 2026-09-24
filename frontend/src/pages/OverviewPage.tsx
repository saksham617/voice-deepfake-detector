import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ContactsIcon,
  MessageIcon,
  PhoneIcon,
  ShieldIcon,
  WaveIcon,
} from "../components/icons";
import { MODEL_NAME } from "../config";
import { ApiError, fetchReports } from "../services/api";
import { reportToActivityItem, type ActivityItem } from "../utils/activity";

const RECENT_ACTIVITY_LIMIT = 4;

interface QuickAction {
  to: string;
  label: string;
  description: string;
  icon: typeof WaveIcon;
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    to: "/voice-check",
    label: "Voice check",
    description: "Upload or record audio to verify a voice.",
    icon: WaveIcon,
  },
  {
    to: "/message-check",
    label: "Message check",
    description: "Scan a text or message for phishing language.",
    icon: MessageIcon,
  },
  {
    to: "/contacts",
    label: "Contacts",
    description: "Enroll a voiceprint or verify against one.",
    icon: ContactsIcon,
  },
  {
    to: "/live-call",
    label: "Live call",
    description: "Monitor an in-progress call in real time.",
    icon: PhoneIcon,
  },
];

interface StatCardProps {
  label: string;
  value: string;
  hint?: string;
}

function StatCard({ label, value, hint }: StatCardProps) {
  return (
    <div className="rounded-2xl border border-line bg-panel p-5">
      <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
        {label}
      </p>
      <p className="mt-2 font-plexMono text-2xl font-semibold text-ink">{value}</p>
      {hint && <p className="mt-1 text-xs text-ink-dim">{hint}</p>}
    </div>
  );
}

const TIER_DOT = {
  safe: "bg-safe",
  medium: "bg-medium",
  high: "bg-high",
} as const;

export function OverviewPage() {
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [activityError, setActivityError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchReports(controller.signal)
      .then((reports) => setActivity(reports.slice(0, RECENT_ACTIVITY_LIMIT).map(reportToActivityItem)))
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setActivityError(err instanceof ApiError ? err.message : "Could not load recent activity.");
      });
    return () => controller.abort();
  }, []);

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h2 className="text-xl font-semibold text-ink">Overview</h2>
          <p className="mt-1 text-sm text-ink-dim">
            Your voice-fraud detection at a glance.
          </p>
        </header>

        {/* Quick actions */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {QUICK_ACTIONS.map(({ to, label, description, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className="group rounded-2xl border border-line bg-panel p-4 transition-colors hover:border-line-strong hover:bg-panel-raised"
            >
              <span className="grid h-9 w-9 place-items-center rounded-lg bg-safe-dim text-safe">
                <Icon className="h-4.5 w-4.5" />
              </span>
              <p className="mt-3 text-sm font-semibold text-ink">{label}</p>
              <p className="mt-1 text-xs text-ink-dim">{description}</p>
            </Link>
          ))}
        </div>

        {/* Stats */}
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Checks today" value="27" />
          <StatCard label="Flagged" value="4" hint="last 24h" />
          <StatCard label="Reports filed" value="9" />
          <StatCard label="Contacts verified" value="12" />
        </div>

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[1fr_280px]">
          {/* Recent activity */}
          <section className="rounded-2xl border border-line bg-panel p-5">
            <h3 className="mb-4 text-sm font-semibold text-ink">Recent activity</h3>
            {activityError ? (
              <p className="py-4 text-center text-sm text-ink-faint">{activityError}</p>
            ) : activity.length === 0 ? (
              <p className="py-4 text-center text-sm text-ink-faint">No activity yet.</p>
            ) : (
              <ul className="space-y-3">
                {activity.map((item) => (
                  <li
                    key={item.id}
                    className="flex items-start gap-3 border-b border-line pb-3 last:border-0 last:pb-0"
                  >
                    <span
                      className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${TIER_DOT[item.tier]}`}
                    />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm text-ink">{item.title}</p>
                      <p className="truncate text-xs text-ink-dim">{item.detail}</p>
                    </div>
                    <span className="shrink-0 text-xs text-ink-faint">{item.timestamp}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {/* System status */}
          <section className="rounded-2xl border border-line bg-panel p-5">
            <h3 className="mb-4 text-sm font-semibold text-ink">System status</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-safe-dim text-safe">
                  <ShieldIcon className="h-3.5 w-3.5" />
                </span>
                <div>
                  <p className="text-sm text-ink">Detector online</p>
                  <p className="text-xs text-ink-dim">{MODEL_NAME}</p>
                </div>
              </div>
              <div className="flex items-center justify-between border-t border-line pt-3 text-xs">
                <span className="text-ink-dim">Voice check</span>
                <span className="text-safe">Operational</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-ink-dim">Message check</span>
                <span className="text-safe">Operational</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-ink-dim">Live call</span>
                <span className="text-medium">Beta</span>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
