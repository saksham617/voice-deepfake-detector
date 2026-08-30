import { WaveIcon } from "./icons";

interface LoadingStateProps {
  onCancel?: () => void;
}

/** Processing indicator shown while analysis is in flight. */
export function LoadingState({ onCancel }: LoadingStateProps) {
  return (
    <div
      className="flex flex-col items-center gap-5 py-10 text-center"
      role="status"
      aria-live="polite"
    >
      <div className="relative grid place-items-center h-16 w-16">
        <span className="absolute inset-0 rounded-full border-2 border-accent/20" />
        <span className="absolute inset-0 rounded-full border-2 border-transparent border-t-accent animate-spin" />
        <WaveIcon className="h-7 w-7 text-accent" />
      </div>
      <div>
        <p className="text-sm font-medium text-white">Analyzing audio…</p>
        <p className="mt-1 text-xs text-slate-400">
          Running the detection pipeline. This usually takes a few seconds.
        </p>
      </div>
      {onCancel && (
        <button
          type="button"
          onClick={onCancel}
          className="text-xs font-medium text-slate-400 hover:text-white underline underline-offset-4 transition-colors"
        >
          Cancel
        </button>
      )}
    </div>
  );
}
