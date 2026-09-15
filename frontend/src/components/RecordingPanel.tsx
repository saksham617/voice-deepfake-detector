import { formatMmSs } from "../utils/time";
import { AudioLevelMeter } from "./AudioLevelMeter";

interface RecordingPanelProps {
  status: "requesting" | "recording";
  elapsedMs: number;
  analyser: AnalyserNode | null;
  onStop: () => void;
}

/** In-progress microphone recording UI: permission wait, then live meter + timer + stop. */
export function RecordingPanel({ status, elapsedMs, analyser, onStop }: RecordingPanelProps) {
  if (status === "requesting") {
    return (
      <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-line px-6 py-14 text-center">
        <span className="h-2.5 w-2.5 rounded-full bg-medium animate-glowpulse" />
        <p className="text-sm font-medium text-ink">Waiting for microphone access</p>
        <p className="max-w-xs text-xs text-ink-dim">
          Allow microphone access in the browser prompt to start recording.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-4 rounded-2xl border border-line bg-panel-raised px-6 py-10 text-center">
      <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-high">
        <span className="h-1.5 w-1.5 rounded-full bg-high animate-glowpulse" />
        Recording
      </span>
      <span className="font-plexMono text-2xl font-semibold text-ink">
        {formatMmSs(elapsedMs / 1000)}
      </span>
      <div className="w-full max-w-[220px]">
        <AudioLevelMeter analyser={analyser} />
      </div>
      <button
        type="button"
        onClick={onStop}
        className="mt-1 rounded-xl bg-high px-5 py-2.5 text-sm font-semibold text-canvas transition-opacity hover:opacity-90"
      >
        Stop recording
      </button>
    </div>
  );
}
