import type { PredictionResponse } from "../types/prediction";
import { CheckIcon, AlertIcon } from "./icons";

interface ResultCardProps {
  result: PredictionResponse;
  onReset: () => void;
}

/**
 * Presents the prediction so it is immediately understandable:
 *   ANALYSIS RESULT -> BONAFIDE/SPOOF -> plain-language subtitle -> Confidence.
 *
 * NOTE: "Confidence" is presented as-is. We deliberately do NOT claim it is the
 * probability the model is correct — that meaning isn't defined by the pipeline.
 */
export function ResultCard({ result, onReset }: ResultCardProps) {
  const isSpoof = result.prediction === "spoof";
  const confidencePct = Math.round(result.confidence * 100);

  const label = isSpoof ? "SPOOF" : "BONAFIDE";
  const subtitle = isSpoof ? "AI-generated voice" : "Genuine voice";
  const accentText = isSpoof ? "text-spoof" : "text-bonafide";
  const accentBar = isSpoof ? "bg-spoof" : "bg-bonafide";
  const accentRing = isSpoof
    ? "ring-spoof/30 bg-spoof/10"
    : "ring-bonafide/30 bg-bonafide/10";

  return (
    <div
      className="flex flex-col items-center text-center py-4"
      role="region"
      aria-label="Analysis result"
    >
      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
        Analysis Result
      </p>

      <span
        className={`mt-6 grid place-items-center h-16 w-16 rounded-2xl ring-1 ${accentRing} ${accentText}`}
      >
        {isSpoof ? (
          <AlertIcon className="h-8 w-8" />
        ) : (
          <CheckIcon className="h-8 w-8" />
        )}
      </span>

      <h2
        className={`mt-5 text-4xl sm:text-5xl font-bold tracking-tight ${accentText}`}
      >
        {label}
      </h2>
      <p className="mt-2 text-sm text-slate-300">{subtitle}</p>

      {/* Confidence */}
      <div className="mt-8 w-full max-w-xs">
        <div className="flex items-baseline justify-between">
          <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Confidence
          </span>
          <span className={`text-2xl font-semibold tabular-nums ${accentText}`}>
            {confidencePct}%
          </span>
        </div>
        <div
          className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10"
          role="progressbar"
          aria-valuenow={confidencePct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Confidence"
        >
          <div
            className={`h-full rounded-full ${accentBar} transition-[width] duration-700`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      <button
        type="button"
        onClick={onReset}
        className="mt-9 rounded-lg border border-white/15 px-5 py-2.5 text-sm font-medium text-slate-200 hover:bg-white/5 hover:border-white/25 transition-colors"
      >
        Analyze another file
      </button>
    </div>
  );
}
