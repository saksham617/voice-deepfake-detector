import { useRef, useState, type DragEvent } from "react";
import { ACCEPTED_AUDIO_EXTENSIONS } from "../config";
import { UploadIcon } from "./icons";

interface FileUploadProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

/**
 * Drag-and-drop + click-to-browse audio picker.
 * Validation happens upstream (useAnalyzer); this component only surfaces the
 * chosen file.
 */
export function FileUpload({ onFileSelected, disabled }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const openPicker = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleDrag = (e: DragEvent<HTMLDivElement>, active: boolean) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setDragActive(active);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onFileSelected(file);
  };

  const accept = ACCEPTED_AUDIO_EXTENSIONS.join(",") + ",audio/*";

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Upload an audio file"
      onClick={openPicker}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openPicker();
        }
      }}
      onDragEnter={(e) => handleDrag(e, true)}
      onDragOver={(e) => handleDrag(e, true)}
      onDragLeave={(e) => handleDrag(e, false)}
      onDrop={handleDrop}
      className={[
        "group relative flex flex-col items-center justify-center gap-4",
        "rounded-2xl border-2 border-dashed px-6 py-14 text-center",
        "transition-colors cursor-pointer select-none",
        disabled
          ? "opacity-50 cursor-not-allowed border-white/10"
          : dragActive
            ? "border-accent bg-accent/5"
            : "border-white/15 hover:border-accent/60 hover:bg-white/[0.02]",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelected(file);
          // Reset so selecting the same file again re-fires onChange.
          e.target.value = "";
        }}
      />
      <span
        className={[
          "grid place-items-center h-14 w-14 rounded-xl transition-colors",
          dragActive
            ? "bg-accent/20 text-accent"
            : "bg-white/5 text-slate-300 group-hover:text-accent",
        ].join(" ")}
      >
        <UploadIcon className="h-7 w-7" />
      </span>
      <div>
        <p className="text-sm font-medium text-white">
          {dragActive ? "Drop the audio file" : "Drag & drop an audio file"}
        </p>
        <p className="mt-1 text-xs text-slate-400">
          or{" "}
          <span className="text-accent font-medium">browse your computer</span>
        </p>
      </div>
      <p className="text-[11px] text-slate-500">
        WAV · MP3 · FLAC · OGG · M4A — up to 25 MB
      </p>
    </div>
  );
}
