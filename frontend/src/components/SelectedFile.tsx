import { formatBytes } from "../utils/audioFile";
import { CloseIcon, WaveIcon } from "./icons";

interface SelectedFileProps {
  file: File;
  onClear: () => void;
  disabled?: boolean;
}

/** Shows the currently selected audio file with a way to remove it. */
export function SelectedFile({ file, onClear, disabled }: SelectedFileProps) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-line bg-panel-raised px-4 py-3">
      <span className="grid place-items-center h-10 w-10 shrink-0 rounded-lg bg-safe-dim text-safe">
        <WaveIcon className="h-5 w-5" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-ink" title={file.name}>
          {file.name}
        </p>
        <p className="text-xs text-ink-dim">
          {formatBytes(file.size)}
          {file.type ? ` · ${file.type}` : ""}
        </p>
      </div>
      <button
        type="button"
        onClick={onClear}
        disabled={disabled}
        aria-label="Remove selected file"
        className="grid place-items-center h-8 w-8 rounded-lg text-ink-dim hover:text-ink hover:bg-panel disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        <CloseIcon className="h-4 w-4" />
      </button>
    </div>
  );
}
