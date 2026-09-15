import { useEffect, useState } from "react";
import { CircularGauge } from "../components/CircularGauge";
import { PhoneIcon } from "../components/icons";
import { riskTier, TIER_STYLES } from "../components/VerdictCard";

const HISTORY_LENGTH = 24;
const RISK_TICK_MS = 1200;
const STATUS_TICK_MS = 1400;

const RISK_LABEL = {
  safe: "Low risk",
  medium: "Elevated risk",
  high: "High risk",
} as const;

export function LiveCallPage() {
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<"listening" | "analyzing">("listening");
  const [risk, setRisk] = useState(12);
  const [history, setHistory] = useState<number[]>([]);

  // Alternate the status badge while a call is "in progress" — purely cosmetic,
  // there's no real streaming pipeline behind this yet.
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      setStatus((prev) => (prev === "listening" ? "analyzing" : "listening"));
    }, STATUS_TICK_MS);
    return () => window.clearInterval(id);
  }, [running]);

  // Drift the risk score randomly to simulate a live scoring feed.
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      setRisk((prev) => {
        const drift = (Math.random() - 0.45) * 22;
        return Math.max(2, Math.min(97, Math.round(prev + drift)));
      });
    }, RISK_TICK_MS);
    return () => window.clearInterval(id);
  }, [running]);

  // Append every risk change to the rolling timeline while a call is active.
  useEffect(() => {
    if (!running) return;
    setHistory((h) => [...h, risk].slice(-HISTORY_LENGTH));
  }, [risk, running]);

  const handleStart = () => {
    setRisk(12);
    setHistory([12]);
    setStatus("listening");
    setRunning(true);
  };

  const handleStop = () => setRunning(false);

  const tier = riskTier(risk);
  const styles = TIER_STYLES[tier];
  const statusLabel = !running
    ? history.length > 0
      ? "Call ended"
      : "Ready"
    : status === "listening"
      ? "Listening"
      : "Analyzing";

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h2 className="flex items-center gap-2 text-xl font-semibold text-ink">
            Live call
            <span className="rounded-full border border-line bg-panel px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-medium">
              Beta
            </span>
          </h2>
          <p className="mt-1 text-sm text-ink-dim">
            Real-time risk scoring for an in-progress call. Simulated for now —
            live audio streaming isn't wired up yet.
          </p>
        </header>

        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[360px_1fr]">
          {/* Left column: call control */}
          <section className="rounded-2xl border border-line bg-panel p-5">
            <h3 className="mb-4 text-sm font-semibold text-ink">Call status</h3>

            <div className="flex flex-col items-center gap-4 rounded-xl border border-line bg-panel-raised px-6 py-8 text-center">
              <span className="grid h-12 w-12 place-items-center rounded-full bg-safe-dim text-safe">
                <PhoneIcon className="h-5 w-5" />
              </span>
              <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-ink-dim">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    running ? "bg-high animate-glowpulse" : "bg-ink-faint"
                  }`}
                />
                {statusLabel}
              </span>
              <p className="text-xs text-ink-dim">
                {running
                  ? "Monitoring live audio for synthetic-voice risk signals."
                  : "Start a simulated call to see live risk scoring."}
              </p>
            </div>

            {running ? (
              <button
                type="button"
                onClick={handleStop}
                className="mt-4 w-full rounded-xl bg-high px-4 py-3 text-sm font-semibold text-canvas transition-opacity hover:opacity-90"
              >
                End call
              </button>
            ) : (
              <button
                type="button"
                onClick={handleStart}
                className="mt-4 w-full rounded-xl bg-safe px-4 py-3 text-sm font-semibold text-canvas transition-opacity hover:opacity-90"
              >
                Start simulated call
              </button>
            )}
          </section>

          {/* Right column: risk gauge + timeline */}
          <section className="space-y-4">
            <div className="flex items-center gap-6 rounded-2xl border border-line bg-panel p-6">
              <CircularGauge value={risk} strokeColorClass={styles.stroke} />
              <div className="min-w-0">
                <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
                  Current risk
                </p>
                <p className={`mt-1 text-xl font-semibold ${styles.text}`}>{RISK_LABEL[tier]}</p>
                <p className="mt-1 text-sm text-ink-dim">
                  Synthetic-voice score:{" "}
                  <span className="font-plexMono text-ink">{risk}</span>/100
                </p>
              </div>
            </div>

            <div className="rounded-2xl border border-line bg-panel p-5">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-ink">Risk timeline</h3>
                <span className="text-[11px] text-ink-faint">updates every ~1.2s</span>
              </div>
              <div className="flex h-32 items-end gap-1">
                {history.length === 0 ? (
                  <div className="flex h-full w-full items-center justify-center text-xs text-ink-faint">
                    No call data yet.
                  </div>
                ) : (
                  history.map((v, i) => {
                    const barTier = riskTier(v);
                    const barColor =
                      barTier === "safe" ? "bg-safe" : barTier === "medium" ? "bg-medium" : "bg-high";
                    return (
                      <div
                        key={i}
                        className={`min-w-[4px] flex-1 rounded-t-sm ${barColor}`}
                        style={{ height: `${Math.max(4, v)}%` }}
                        title={`${v}/100`}
                      />
                    );
                  })
                )}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
