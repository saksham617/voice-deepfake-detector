import { useState } from "react";
import { ErrorBanner } from "../components/ErrorBanner";
import { FileUpload } from "../components/FileUpload";
import { WaveIcon } from "../components/icons";
import { InputSourceTabs, type InputSource } from "../components/InputSourceTabs";
import { LoadingState } from "../components/LoadingState";
import { ResultCard } from "../components/ResultCard";
import { SelectedFile } from "../components/SelectedFile";
import { WaveVisual } from "../components/WaveVisual";
import { useAnalyzer } from "../hooks/useAnalyzer";

export function AnalyzerPage() {
  const analyzer = useAnalyzer();
  const [source, setSource] = useState<InputSource>("upload");

  const { status, file, result, error } = analyzer;
  const isAnalyzing = status === "analyzing";
  const showResult = status === "result" && result !== null;

  return (
    <div className="mx-auto max-w-2xl px-4 sm:px-6 py-10 sm:py-14">
      {/* Intro */}
      <section className="text-center mb-8">
        <span className="inline-flex items-center gap-2 rounded-full border border-accent/25 bg-accent/10 px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-accent">
          <span className="h-1.5 w-1.5 rounded-full bg-accent animate-glowpulse" />
          Voice authenticity analysis
        </span>
        <h2 className="mt-4 text-2xl sm:text-3xl font-bold tracking-tight text-white">
          Is this voice real or AI-generated?
        </h2>
        <p className="mx-auto mt-3 max-w-lg text-sm text-slate-400">
          Upload an audio recording and VoiceGuard analyzes it to predict whether
          the speech is genuine (bonafide) or synthetic/cloned (spoof).
        </p>
      </section>

      {/* Main card */}
      <section className="overflow-hidden rounded-3xl border border-white/10 bg-surface-900/70 shadow-2xl shadow-black/40">
        {showResult ? (
          <div className="p-5 sm:p-8">
            <ResultCard result={result} onReset={analyzer.reset} />
          </div>
        ) : (
          <>
            {/* Glowing waveform hero */}
            <div className="relative">
              <WaveVisual
                active={isAnalyzing}
                className="h-32 w-full sm:h-40"
              />
              <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-b from-transparent to-surface-900/70" />
            </div>

            <div className="space-y-5 p-5 sm:p-8 pt-2">
              {isAnalyzing ? (
                <LoadingState onCancel={analyzer.cancel} />
              ) : (
                <>
                  <InputSourceTabs active={source} onChange={setSource} />

                  {source === "upload" ? (
                    file ? (
                      <SelectedFile file={file} onClear={analyzer.reset} />
                    ) : (
                      <FileUpload onFileSelected={analyzer.selectFile} />
                    )
                  ) : (
                    <RecordPlaceholder />
                  )}

                  {error && (
                    <ErrorBanner message={error} onDismiss={analyzer.reset} />
                  )}

                  <button
                    type="button"
                    onClick={analyzer.analyze}
                    disabled={!file}
                    className="w-full rounded-2xl bg-accent px-4 py-3.5 text-sm font-semibold text-surface-950 transition-all hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40 enabled:glow-accent"
                  >
                    Analyze Audio
                  </button>
                </>
              )}
            </div>
          </>
        )}
      </section>

      <p className="mt-4 text-center text-xs text-slate-500">
        Results are model predictions and may not be perfectly accurate.
      </p>
    </div>
  );
}

/** Placeholder shown under the (disabled) Record tab. */
function RecordPlaceholder() {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-white/15 px-6 py-12 text-center">
      <span className="grid place-items-center h-12 w-12 rounded-xl bg-accent/10 text-accent">
        <WaveIcon className="h-6 w-6" />
      </span>
      <p className="text-sm font-medium text-white">
        Microphone recording is coming soon
      </p>
      <p className="max-w-xs text-xs text-slate-400">
        For now, please use the Upload tab to analyze an audio file.
      </p>
    </div>
  );
}
