import { AlertIcon } from "./icons";

interface ErrorBannerProps {
  message: string;
  onDismiss?: () => void;
}

/** Inline, user-friendly error message. */
export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-xl border border-spoof/30 bg-spoof/10 px-4 py-3"
    >
      <AlertIcon className="mt-0.5 h-5 w-5 shrink-0 text-spoof" />
      <p className="flex-1 text-sm text-slate-200">{message}</p>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="text-xs font-medium text-slate-400 hover:text-white transition-colors"
        >
          Dismiss
        </button>
      )}
    </div>
  );
}
