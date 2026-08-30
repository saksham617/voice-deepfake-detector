interface LoadingStateProps {
  onCancel?: () => void;
}

/**
 * Processing caption shown while analysis is in flight. The animated waveform
 * above provides the motion, so this stays focused on the message + cancel.
 */
export function LoadingState({ onCancel }: LoadingStateProps) {
  return (
    <div
      className="flex flex-col items-center gap-3 py-4 text-center"
      role="status"
      aria-live="polite"
    >
      <p className="flex items-center gap-2 text-sm font-medium text-white">
        Analyzing audio
        <span className="inline-flex gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-accent animate-glowpulse [animation-delay:0ms]" />
          <span className="h-1.5 w-1.5 rounded-full bg-accent animate-glowpulse [animation-delay:200ms]" />
          <span className="h-1.5 w-1.5 rounded-full bg-accent animate-glowpulse [animation-delay:400ms]" />
        </span>
      </p>
      <p className="text-xs text-slate-400">
        Running the detection pipeline. This usually takes a few seconds.
      </p>
      {onCancel && (
        <button
          type="button"
          onClick={onCancel}
          className="mt-1 text-xs font-medium text-slate-400 hover:text-white underline underline-offset-4 transition-colors"
        >
          Cancel
        </button>
      )}
    </div>
  );
}
