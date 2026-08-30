import { useState } from "react";
import { ErrorBanner } from "../components/ErrorBanner";
import { FileUpload } from "../components/FileUpload";
import { WaveIcon } from "../components/icons";
import { InputSourceTabs, type InputSource } from "../components/InputSourceTabs";
import { LoadingState } from "../components/LoadingState";
import { ResultCard } from "../components/ResultCard";
import { SelectedFile } from "../components/SelectedFile";
import { useAnalyzer } from "../hooks/useAnalyzer";

export function AnalyzerPage() {
  const analyzer = useAnalyzer();
  const [source, setSource] = useState<InputSource>("upload");

  const { status, file, result, error } = analyzer;
  const isAnalyzing = status === "analyzing";

  return (
    <div className="mx-auto max-w-2xl px-4 sm:px-6 py-10 sm:py-14">
      {/* Intro */}
      <section className="text-center mb-8">
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
          Is this voice real or AI-generated?
        </h2>
        <p className="mx-auto mt-3 max-w-lg text-sm text-slate-400">
          Upload an audio recording and VoiceGuard analyzes it to predict whether
          the speech is genuine (bonafide) or synthetic/cloned (spoof).
        </p>
      </section>

      {/* Main card */}
      <section className="rounded-2xl border border-white/10 bg-surface-900/70 p-5 sm:p-8 shadow-xl shadow-black/30">
        {status === "result" && result ? (
          <ResultCard result={result} onReset={analyzer.reset} />
        ) : isAnalyzing ? (
          <LoadingState onCancel={analyzer.cancel} />
        ) : (
          <div className="space-y-5">
            <InputSourceTabs
              active={source}
              onChange={setSource}
              disabled={isAnalyzing}
            />

            {source === "upload" ? (
              file ? (
                <SelectedFile
                  file={file}
                  onClear={analyzer.reset}
                  disabled={isAnalyzing}
                />
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
              disabled={!file || isAnalyzing}
              className="w-full rounded-xl bg-accent px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
            >
              Analyze Audio
            </button>
          </div>
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
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-white/15 px-6 py-14 text-center">
      <span className="grid place-items-center h-12 w-12 rounded-xl bg-white/5 text-slate-400">
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
