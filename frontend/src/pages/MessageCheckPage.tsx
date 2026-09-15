import { useState } from "react";
import { ErrorBanner } from "../components/ErrorBanner";
import { CircularGauge } from "../components/CircularGauge";
import { LoadingState } from "../components/LoadingState";
import { riskTier, TIER_STYLES } from "../components/VerdictCard";
import { AlertIcon, CheckIcon } from "../components/icons";
import { useMessageCheck } from "../hooks/useMessageCheck";

const TIER_LABEL = {
  safe: "Looks safe",
  medium: "Some suspicious signals",
  high: "Likely phishing",
} as const;

/** Escapes regex-special characters so a flagged phrase can be used literally. */
function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Renders `text` with any of `phrases` wrapped in a highlighted <mark>. */
function highlightPhrases(text: string, phrases: string[]) {
  if (phrases.length === 0) return text;
  const pattern = new RegExp(`(${phrases.map(escapeRegExp).join("|")})`, "gi");
  const lowerPhrases = phrases.map((p) => p.toLowerCase());
  return text.split(pattern).map((part, i) =>
    lowerPhrases.includes(part.toLowerCase()) ? (
      <mark key={i} className="rounded bg-high-dim px-1 text-high">
        {part}
      </mark>
    ) : (
      <span key={i}>{part}</span>
    ),
  );
}

export function MessageCheckPage() {
  const [text, setText] = useState("");
  const check = useMessageCheck();
  const isChecking = check.status === "checking";
  const showResult = check.status === "result" && check.result !== null;

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h2 className="text-xl font-semibold text-ink">Message check</h2>
          <p className="mt-1 text-sm text-ink-dim">
            Paste a text or message to scan it for phishing language.
          </p>
        </header>

        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[360px_1fr]">
          {/* Left column: submit message */}
          <section className="rounded-2xl border border-line bg-panel p-5">
            <h3 className="mb-4 text-sm font-semibold text-ink">Message text</h3>

            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={isChecking}
              placeholder="Paste the message you want to check…"
              rows={10}
              className="w-full resize-none rounded-xl border border-line bg-panel-raised px-4 py-3 text-sm text-ink placeholder:text-ink-faint focus:border-line-strong focus:outline-none disabled:opacity-60"
            />

            {check.error && (
              <div className="mt-3">
                <ErrorBanner message={check.error} onDismiss={check.reset} />
              </div>
            )}

            {isChecking ? (
              <div className="mt-2">
                <LoadingState onCancel={check.cancel} />
              </div>
            ) : (
              <button
                type="button"
                onClick={() => check.check(text)}
                disabled={!text.trim()}
                className="mt-4 w-full rounded-xl bg-safe px-4 py-3 text-sm font-semibold text-canvas transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Check message
              </button>
            )}
          </section>

          {/* Right column: results */}
          <section>
            {showResult && check.result ? (
              <MessageResults
                submittedText={check.submittedText ?? ""}
                phishingProbability={check.result.phishing_probability}
                flaggedPhrases={check.result.flagged_phrases}
              />
            ) : (
              <div className="flex min-h-[320px] items-center justify-center rounded-2xl border border-dashed border-line text-sm text-ink-faint">
                {isChecking
                  ? "Checking…"
                  : "Results will appear here once you check a message."}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

interface MessageResultsProps {
  submittedText: string;
  phishingProbability: number;
  flaggedPhrases: string[];
}

function MessageResults({ submittedText, phishingProbability, flaggedPhrases }: MessageResultsProps) {
  const pct = Math.round(phishingProbability * 100);
  const tier = riskTier(pct);
  const styles = TIER_STYLES[tier];

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-line bg-panel-raised px-4 py-3 text-xs leading-relaxed text-ink-dim">
        <span className="font-medium text-ink">Phishing probability</span> shows
        how confident the model is that this message is a phishing/scam attempt
        — 0 means safe, 100 means highly suspicious.
      </div>

      <div className="flex items-center gap-6 rounded-2xl border border-line bg-panel p-6">
        <CircularGauge value={pct} strokeColorClass={styles.stroke} />
        <div className="min-w-0">
          <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
            Verdict
          </p>
          <p className={`mt-1 flex items-center gap-2 text-xl font-semibold ${styles.text}`}>
            <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full ${styles.chip}`}>
              {tier === "safe" ? (
                <CheckIcon className="h-3.5 w-3.5" />
              ) : (
                <AlertIcon className="h-3.5 w-3.5" />
              )}
            </span>
            {TIER_LABEL[tier]}
          </p>
          <p className="mt-1 text-sm text-ink-dim">
            Phishing probability:{" "}
            <span className="font-plexMono text-ink">{pct}</span>/100
          </p>
        </div>
      </div>

      <div className="rounded-2xl border border-line bg-panel p-5">
        <h3 className="mb-3 text-sm font-semibold text-ink">Flagged phrases</h3>
        {flaggedPhrases.length > 0 ? (
          <div className="mb-4 flex flex-wrap gap-2">
            {flaggedPhrases.map((phrase) => (
              <span
                key={phrase}
                className="rounded-full bg-high-dim px-2.5 py-1 text-xs font-medium text-high"
              >
                {phrase}
              </span>
            ))}
          </div>
        ) : (
          <p className="mb-4 text-xs text-ink-dim">No known phishing phrases detected.</p>
        )}
        <div className="whitespace-pre-wrap rounded-xl border border-line bg-panel-raised px-4 py-3 text-sm leading-relaxed text-ink">
          {highlightPhrases(submittedText, flaggedPhrases)}
        </div>
      </div>
    </div>
  );
}
