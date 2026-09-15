import { useState } from "react";
import { ErrorBanner } from "../components/ErrorBanner";
import { FileUpload } from "../components/FileUpload";
import { LoadingState } from "../components/LoadingState";
import { RecordingPanel } from "../components/RecordingPanel";
import { SelectedFile } from "../components/SelectedFile";
import { CircularGauge } from "../components/CircularGauge";
import { AlertIcon, CheckIcon, ContactsIcon, MicIcon, ReportIcon } from "../components/icons";
import { useContacts } from "../hooks/useContacts";
import { useMicRecorder } from "../hooks/useMicRecorder";
import { validateAudioFile } from "../utils/audioFile";

type Tab = "enroll" | "verify";

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export function ContactsPage() {
  const contacts = useContacts();
  const [tab, setTab] = useState<Tab>("enroll");
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [selectedContactId, setSelectedContactId] = useState("");
  const [reported, setReported] = useState(false);

  const selectFile = (candidate: File) => {
    const validation = validateAudioFile(candidate);
    if (!validation.ok) {
      setFile(null);
      setFileError(validation.error ?? "That file can't be used.");
      return;
    }
    setFile(candidate);
    setFileError(null);
  };

  const recorder = useMicRecorder({ onRecorded: selectFile });

  const switchTab = (next: Tab) => {
    setTab(next);
    setFile(null);
    setFileError(null);
    setReported(false);
    contacts.resetEnroll();
    contacts.resetVerify();
  };

  const isEnrolling = contacts.enrollStatus === "enrolling";
  const isVerifying = contacts.verifyStatus === "verifying";
  const busy = isEnrolling || isVerifying;

  const canSubmitEnroll = tab === "enroll" && !!file && name.trim().length > 0 && !busy;
  const canSubmitVerify = tab === "verify" && !!file && !!selectedContactId && !busy;

  const handleSubmit = () => {
    if (!file) return;
    if (tab === "enroll" && name.trim()) {
      void contacts.enroll(name.trim(), file);
    } else if (tab === "verify" && selectedContactId) {
      setReported(false);
      void contacts.verify(selectedContactId, file);
    }
  };

  const selectedContactName = contacts.contacts.find((c) => c.id === selectedContactId)?.name;

  return (
    <div className="min-h-full bg-canvas px-4 py-8 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6">
          <h2 className="text-xl font-semibold text-ink">Contacts</h2>
          <p className="mt-1 text-sm text-ink-dim">
            Enroll a known speaker's voiceprint, or verify a sample against one.
          </p>
        </header>

        {/* Tabs */}
        <div className="mb-6 inline-flex rounded-xl border border-line bg-panel p-1">
          {(["enroll", "verify"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => switchTab(t)}
              className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                tab === t ? "bg-panel-raised text-ink" : "text-ink-dim hover:text-ink"
              }`}
            >
              {t === "enroll" ? "Enroll contact" : "Verify voice"}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[360px_1fr]">
          {/* Left column: form */}
          <section className="rounded-2xl border border-line bg-panel p-5">
            {tab === "enroll" ? (
              <>
                <h3 className="mb-4 text-sm font-semibold text-ink">New contact</h3>
                <label className="mb-1.5 block text-xs font-medium text-ink-dim">
                  Contact name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={busy}
                  placeholder="e.g. Aditi Rao"
                  className="mb-4 w-full rounded-xl border border-line bg-panel-raised px-4 py-2.5 text-sm text-ink placeholder:text-ink-faint focus:border-line-strong focus:outline-none disabled:opacity-60"
                />
              </>
            ) : (
              <>
                <h3 className="mb-4 text-sm font-semibold text-ink">Sample to verify</h3>
                <label className="mb-1.5 block text-xs font-medium text-ink-dim">
                  Verify against
                </label>
                <select
                  value={selectedContactId}
                  onChange={(e) => setSelectedContactId(e.target.value)}
                  disabled={busy}
                  className="mb-4 w-full rounded-xl border border-line bg-panel-raised px-4 py-2.5 text-sm text-ink focus:border-line-strong focus:outline-none disabled:opacity-60"
                >
                  <option value="">Select a contact…</option>
                  {contacts.contacts.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </>
            )}

            {busy ? (
              <LoadingState />
            ) : file ? (
              <SelectedFile file={file} onClear={() => setFile(null)} />
            ) : recorder.status !== "idle" ? (
              <RecordingPanel
                status={recorder.status}
                elapsedMs={recorder.elapsedMs}
                analyser={recorder.analyser}
                onStop={recorder.stop}
              />
            ) : (
              <>
                <FileUpload onFileSelected={selectFile} />

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
            {fileError && (
              <div className="mt-3">
                <ErrorBanner message={fileError} onDismiss={() => setFileError(null)} />
              </div>
            )}
            {tab === "enroll" && contacts.enrollError && (
              <div className="mt-3">
                <ErrorBanner message={contacts.enrollError} onDismiss={contacts.resetEnroll} />
              </div>
            )}
            {tab === "verify" && contacts.verifyError && (
              <div className="mt-3">
                <ErrorBanner message={contacts.verifyError} onDismiss={contacts.resetVerify} />
              </div>
            )}

            <button
              type="button"
              onClick={handleSubmit}
              disabled={tab === "enroll" ? !canSubmitEnroll : !canSubmitVerify}
              className="mt-4 w-full rounded-xl bg-safe px-4 py-3 text-sm font-semibold text-canvas transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {tab === "enroll" ? "Enroll contact" : "Verify voice"}
            </button>
          </section>

          {/* Right column: results */}
          <section className="space-y-4">
            {tab === "enroll" ? (
              <>
                {contacts.enrollStatus === "success" && (
                  <div className="flex items-center gap-3 rounded-2xl border border-safe/30 bg-safe-dim px-5 py-4">
                    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-safe/20 text-safe">
                      <CheckIcon className="h-4.5 w-4.5" />
                    </span>
                    <div>
                      <p className="text-sm font-semibold text-ink">Contact enrolled</p>
                      <p className="text-xs text-ink-dim">
                        The voiceprint has been saved and can now be used for verification.
                      </p>
                    </div>
                  </div>
                )}

                <div className="rounded-2xl border border-line bg-panel p-5">
                  <h3 className="mb-4 text-sm font-semibold text-ink">Enrolled contacts</h3>
                  {contacts.contacts.length === 0 ? (
                    <p className="text-sm text-ink-faint">No contacts enrolled yet.</p>
                  ) : (
                    <ul className="space-y-2">
                      {contacts.contacts.map((c) => (
                        <li
                          key={c.id}
                          className="flex items-center gap-3 rounded-xl border border-line bg-panel-raised px-4 py-3"
                        >
                          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-safe-dim text-safe">
                            <ContactsIcon className="h-4 w-4" />
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-medium text-ink">{c.name}</p>
                            <p className="text-xs text-ink-dim">Enrolled {formatDate(c.enrolledAt)}</p>
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </>
            ) : contacts.verifyStatus === "result" && contacts.verifyResult ? (
              <VerifyResultCard
                match={contacts.verifyResult.match}
                similarity={contacts.verifyResult.similarity}
                contactName={selectedContactName ?? "this contact"}
                reported={reported}
                onReport={() => setReported(true)}
              />
            ) : (
              <div className="flex min-h-[320px] items-center justify-center rounded-2xl border border-dashed border-line text-sm text-ink-faint">
                {isVerifying
                  ? "Verifying…"
                  : "Select a contact and submit a sample to verify a voice."}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

interface VerifyResultCardProps {
  match: boolean;
  similarity: number;
  contactName: string;
  reported: boolean;
  onReport: () => void;
}

function VerifyResultCard({ match, similarity, contactName, reported, onReport }: VerifyResultCardProps) {
  const pct = Math.round(similarity * 100);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-6 rounded-2xl border border-line bg-panel p-6">
        <CircularGauge value={pct} strokeColorClass={match ? "stroke-safe" : "stroke-high"} />
        <div className="min-w-0">
          <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
            Verdict
          </p>
          <p className={`mt-1 flex items-center gap-2 text-xl font-semibold ${match ? "text-safe" : "text-high"}`}>
            <span
              className={`grid h-6 w-6 shrink-0 place-items-center rounded-full ${
                match ? "bg-safe-dim" : "bg-high-dim"
              }`}
            >
              {match ? <CheckIcon className="h-3.5 w-3.5" /> : <AlertIcon className="h-3.5 w-3.5" />}
            </span>
            {match ? "Voice matches" : "Voice mismatch"}
          </p>
          <p className="mt-1 text-sm text-ink-dim">
            {match
              ? `This sample matches ${contactName}'s enrolled voiceprint.`
              : `This sample does not match ${contactName}'s enrolled voiceprint.`}
          </p>
          <p className="mt-1 text-sm text-ink-dim">
            Similarity: <span className="font-plexMono text-ink">{pct}</span>/100
          </p>
        </div>
      </div>

      {!match && (
        <button
          type="button"
          onClick={onReport}
          disabled={reported}
          className="flex w-full items-center justify-center gap-2 rounded-xl border border-high/30 bg-high-dim px-4 py-2.5 text-sm font-medium text-high disabled:cursor-not-allowed disabled:opacity-60"
        >
          <ReportIcon className="h-4 w-4" />
          {reported ? "Reported — thank you" : "Report possible impersonation"}
        </button>
      )}
    </div>
  );
}
