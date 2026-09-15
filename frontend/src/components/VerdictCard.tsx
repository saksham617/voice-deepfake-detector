import type { PredictionLabel } from "../types/prediction";
import { CircularGauge } from "./CircularGauge";
import { AlertIcon, CheckIcon } from "./icons";

export type RiskTier = "safe" | "medium" | "high";

/** Score -> tier bands for the fake-probability gauge (0 = real, 100 = synthetic). */
export function riskTier(fakeProbabilityPct: number): RiskTier {
  if (fakeProbabilityPct < 35) return "safe";
  if (fakeProbabilityPct < 65) return "medium";
  return "high";
}

export const TIER_STYLES: Record<RiskTier, { stroke: string; text: string; chip: string }> = {
  safe: { stroke: "stroke-safe", text: "text-safe", chip: "bg-safe-dim" },
  medium: { stroke: "stroke-medium", text: "text-medium", chip: "bg-medium-dim" },
  high: { stroke: "stroke-high", text: "text-high", chip: "bg-high-dim" },
};

interface VerdictCardProps {
  fakeProbabilityPct: number;
  verdict: PredictionLabel;
}

export function VerdictCard({ fakeProbabilityPct, verdict }: VerdictCardProps) {
  const tier = riskTier(fakeProbabilityPct);
  const styles = TIER_STYLES[tier];
  const isSpoof = verdict === "spoof";

  return (
    <div className="flex items-center gap-6 rounded-2xl border border-line bg-panel p-6">
      <CircularGauge value={fakeProbabilityPct} strokeColorClass={styles.stroke} />
      <div className="min-w-0">
        <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
          Verdict
        </p>
        <p className={`mt-1 flex items-center gap-2 text-xl font-semibold ${styles.text}`}>
          <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full ${styles.chip}`}>
            {isSpoof ? (
              <AlertIcon className="h-3.5 w-3.5" />
            ) : (
              <CheckIcon className="h-3.5 w-3.5" />
            )}
          </span>
          {isSpoof ? "Likely AI-generated" : "Likely genuine"}
        </p>
        <p className="mt-1 text-sm text-ink-dim">
          Fake probability:{" "}
          <span className="font-plexMono text-ink">{Math.round(fakeProbabilityPct)}</span>/100
        </p>
      </div>
    </div>
  );
}
