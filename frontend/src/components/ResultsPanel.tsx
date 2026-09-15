import { MODEL_NAME } from "../config";
import type { PredictionResponse } from "../types/prediction";
import { formatMmSs } from "../utils/time";
import { MiniLegend } from "./MiniLegend";
import { QuickMetrics } from "./QuickMetrics";
import { VerdictCard } from "./VerdictCard";

interface ResultsPanelProps {
  result: PredictionResponse;
  /** Audio duration in seconds, or null while it's still being read. */
  duration: number | null;
}

/** Everything shown in the right column once a prediction comes back. */
export function ResultsPanel({ result, duration }: ResultsPanelProps) {
  const fakeProbabilityPct = Math.round(result.spoof_probability * 100);
  const confidencePct = Math.round(result.confidence * 100);

  return (
    <div className="space-y-4">
      <MiniLegend />

      <VerdictCard fakeProbabilityPct={fakeProbabilityPct} verdict={result.prediction} />

      <QuickMetrics
        metrics={[
          { label: "Model", value: MODEL_NAME },
          { label: "Duration", value: duration != null ? formatMmSs(duration) : "—", mono: true },
          { label: "Confidence", value: `${confidencePct}%`, mono: true },
        ]}
      />

      <div className="grid grid-cols-2 gap-3">
        <button
          type="button"
          disabled
          title="Coming soon"
          className="rounded-xl border border-line bg-panel-raised px-4 py-2.5 text-sm font-medium text-ink-dim disabled:cursor-not-allowed disabled:opacity-50"
        >
          Save recording
        </button>
        <button
          type="button"
          disabled
          title="Coming soon, once Contacts is built"
          className="rounded-xl border border-line bg-panel-raised px-4 py-2.5 text-sm font-medium text-ink-dim disabled:cursor-not-allowed disabled:opacity-50"
        >
          Verify against a contact
        </button>
      </div>

      <button
        type="button"
        disabled
        title="Coming soon, once Reports is built"
        className="w-full rounded-xl border border-high/30 bg-high-dim px-4 py-2.5 text-sm font-medium text-high disabled:cursor-not-allowed disabled:opacity-60"
      >
        Report this recording
      </button>
    </div>
  );
}
