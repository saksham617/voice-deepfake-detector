/** Formats a duration in seconds as "m:ss". Used for both audio duration and
 * the live recording timer, so both read the same way. */
export function formatMmSs(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return "—";
  const total = Math.round(totalSeconds);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

const RELATIVE_UNITS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["year", 60 * 60 * 24 * 365],
  ["month", 60 * 60 * 24 * 30],
  ["day", 60 * 60 * 24],
  ["hour", 60 * 60],
  ["minute", 60],
];

const relativeFormatter = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

/** Formats an ISO 8601 timestamp as "4 min ago" / "3 hr ago" style relative
 * time, falling back to "just now" under a minute. Used anywhere a report's
 * server timestamp is shown as an activity feed entry. */
export function formatRelativeTime(isoTimestamp: string): string {
  const then = new Date(isoTimestamp).getTime();
  if (!Number.isFinite(then)) return "";

  const diffSeconds = (then - Date.now()) / 1000;
  for (const [unit, secondsInUnit] of RELATIVE_UNITS) {
    if (Math.abs(diffSeconds) >= secondsInUnit) {
      return relativeFormatter.format(Math.round(diffSeconds / secondsInUnit), unit);
    }
  }
  return "just now";
}
