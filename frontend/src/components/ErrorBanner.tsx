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
      className="flex items-start gap-3 rounded-xl border border-high/30 bg-high-dim px-4 py-3"
    >
      <AlertIcon className="mt-0.5 h-5 w-5 shrink-0 text-high" />
      <p className="flex-1 text-sm text-ink">{message}</p>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="text-xs font-medium text-ink-dim hover:text-ink transition-colors"
        >
          Dismiss
        </button>
      )}
    </div>
  );
}
