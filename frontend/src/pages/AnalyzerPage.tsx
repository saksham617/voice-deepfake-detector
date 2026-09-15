import { useEffect, useState } from "react";
import { ErrorBanner } from "../components/ErrorBanner";
import { FileUpload } from "../components/FileUpload";
import { MicIcon } from "../components/icons";
import { LoadingState } from "../components/LoadingState";
import { RecordingPanel } from "../components/RecordingPanel";
import { ResultsPanel } from "../components/ResultsPanel";
import { SelectedFile } from "../components/SelectedFile";
import { useAnalyzer } from "../hooks/useAnalyzer";
import { useMicRecorder } from "../hooks/useMicRecorder";
import { getAudioDuration } from "../utils/audioFile";

export function AnalyzerPage() {
  const analyzer = useAnalyzer();
  const { status, file, result, error } = analyzer;
  const isAnalyzing = status === "analyzing";
  const showResult = status === "result" && result !== null;

  const recorder = useMicRecorder({ onRecorded: analyzer.selectFile });

  // Read the audio duration client-side (the API doesn't return one) for the
  // "Duration" quick metric. Deliberately kept outside useAnalyzer, which
  // owns only the predict/error/status state machine.
  const [duration, setDuration] = useState<number | null>(null);
  useEffect(() => {
    if (!file) {
      setDuration(null);
      return;
    }
    let cancelled = false;
    getAudioDuration(file)
      .then((seconds) => {
        if (!cancelled) setDuration(seconds);
      })
      .catch(() => {
        if (!cancelled) setDuration(null);
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h2 className="text-xl font-semibold text-ink">Voice check</h2>
          <p className="mt-1 text-sm text-ink-dim">
            Upload or record audio to check whether the voice is genuine or AI-generated.
          </p>
        </header>

        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[360px_1fr]">
          {/* Left column: Submit audio */}
          <section className="rounded-2xl border border-line bg-panel p-5">
            <h3 className="mb-4 text-sm font-semibold text-ink">Submit audio</h3>

            {isAnalyzing ? (
              <LoadingState onCancel={analyzer.cancel} />
            ) : file ? (
              <SelectedFile file={file} onClear={analyzer.reset} />
            ) : recorder.status !== "idle" ? (
              <RecordingPanel
                status={recorder.status}
                elapsedMs={recorder.elapsedMs}
                analyser={recorder.analyser}
                onStop={recorder.stop}
              />
            ) : (
              <>
                <FileUpload onFileSelected={analyzer.selectFile} />

                <div className="my-4 flex items-center gap-3">
                  <div className="h-px flex-1 bg-line" />
                  <span className="text-xs text-ink-faint">or</span>
                  <div className="h-px flex-1 bg-line" />
                </div>

                <button
                  type="button"
                  onClick={recorder.start}
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-line bg-panel-raised px-4 py-3 text-sm font-medium text-ink transition-colors hover:border-line-strong"
                >
                  <MicIcon className="h-4 w-4" />
                  Record directly
                </button>
              </>
            )}

            {recorder.error && (
              <div className="mt-3">
                <ErrorBanner message={recorder.error} onDismiss={recorder.dismissError} />
              </div>
            )}
            {error && (
              <div className="mt-3">
                <ErrorBanner message={error} onDismiss={analyzer.reset} />
              </div>
            )}

            <button
              type="button"
              onClick={analyzer.analyze}
              disabled={!file || isAnalyzing}
              className="mt-4 w-full rounded-xl bg-safe px-4 py-3 text-sm font-semibold text-canvas transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Analyze voice
            </button>
          </section>

          {/* Right column: Results */}
          <section>
            {showResult && result ? (
              <ResultsPanel result={result} duration={duration} />
            ) : (
              <div className="flex min-h-[320px] items-center justify-center rounded-2xl border border-dashed border-line text-sm text-ink-faint">
                {isAnalyzing
                  ? "Analyzing…"
                  : "Results will appear here once you analyze an audio clip."}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
