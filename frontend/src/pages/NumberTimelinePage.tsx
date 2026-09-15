import { Link, useParams } from "react-router-dom";
import { AlertIcon, MessageIcon, PhoneIcon, WaveIcon } from "../components/icons";
import { TIER_STYLES, type RiskTier } from "../components/VerdictCard";

type EventType = "voice" | "message" | "call";

interface TimelineEvent {
  id: string;
  date: string; // ISO
  type: EventType;
  tier: RiskTier;
  summary: string;
  detail: string;
}

// TODO(remove-when-backend-ready): replace with a real GET
// /reports/number/:phoneNumber (or similar) endpoint once it exists.
// Deterministic mock, loosely seeded from the phone number so different
// numbers show some variety in the demo.
function buildTimeline(phoneNumber: string): TimelineEvent[] {
  const seed = Array.from(phoneNumber).reduce((sum, ch) => sum + ch.charCodeAt(0), 0);

  return [
    {
      id: "e1",
      date: "2026-09-10T08:15:00.000Z",
      type: "message",
      tier: "high",
      summary: "Phishing SMS received",
      detail: "\"Your account will be suspended, verify now…\" — flagged 92% phishing probability.",
    },
    {
      id: "e2",
      date: "2026-09-10T08:42:00.000Z",
      type: "call",
      tier: "high",
      summary: "Inbound call, 6 min",
      detail:
        "Call placed 27 minutes after the phishing SMS — consistent with a coordinated vishing follow-up.",
    },
    {
      id: "e3",
      date: "2026-09-10T08:44:00.000Z",
      type: "voice",
      tier: seed % 2 === 0 ? "high" : "medium",
      summary: "Voice check on recorded segment",
      detail:
        seed % 2 === 0
          ? "AI-generated voice detected with 88% confidence."
          : "Borderline confidence score; flagged for manual review.",
    },
    {
      id: "e4",
      date: "2026-09-12T14:05:00.000Z",
      type: "message",
      tier: "medium",
      summary: "Follow-up SMS",
      detail: "\"We tried reaching you, please call back…\" — moderate phishing signals.",
    },
    {
      id: "e5",
      date: "2026-09-14T19:30:00.000Z",
      type: "call",
      tier: "safe",
      summary: "Inbound call, 2 min",
      detail: "No suspicious voice signals detected on this call.",
    },
  ];
}

const TYPE_ICON: Record<EventType, typeof WaveIcon> = {
  voice: WaveIcon,
  message: MessageIcon,
  call: PhoneIcon,
};

const TYPE_LABEL: Record<EventType, string> = {
  voice: "Voice check",
  message: "Message check",
  call: "Call",
};

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
  const events = buildTimeline(decoded);
  const highRiskCount = events.filter((e) => e.tier === "high").length;
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

        <div className="relative space-y-4 pl-6">
          <div className="absolute bottom-2 left-[11px] top-2 w-px bg-line" />
          {events.map((event) => {
            const Icon = TYPE_ICON[event.type];
            const styles = TIER_STYLES[event.tier];
            return (
              <div key={event.id} className="relative">
                <span
                  className={`absolute -left-6 grid h-6 w-6 place-items-center rounded-full ${styles.chip} ${styles.text}`}
                >
                  <Icon className="h-3.5 w-3.5" />
                </span>
                <div className="rounded-2xl border border-line bg-panel p-4">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs font-medium uppercase tracking-wider text-ink-faint">
                      {TYPE_LABEL[event.type]}
                    </span>
                    <span className="text-xs text-ink-faint">{formatDateTime(event.date)}</span>
                  </div>
                  <p className="mt-1.5 text-sm font-medium text-ink">{event.summary}</p>
                  <p className="mt-1 text-xs text-ink-dim">{event.detail}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
