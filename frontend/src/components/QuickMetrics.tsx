interface Metric {
  label: string;
  value: string;
  /** Use IBM Plex Mono for numeric/technical values. */
  mono?: boolean;
}

/** Row of small metric tiles (model, duration, confidence, ...). */
export function QuickMetrics({ metrics }: { metrics: Metric[] }) {
  return (
    <div className="grid grid-cols-3 gap-3">
      {metrics.map((metric) => (
        <div
          key={metric.label}
          className="rounded-xl border border-line bg-panel-raised px-4 py-3"
        >
          <p className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">
            {metric.label}
          </p>
          <p
            className={`mt-1 truncate text-sm font-semibold text-ink ${
              metric.mono ? "font-plexMono" : ""
            }`}
            title={metric.value}
          >
            {metric.value}
          </p>
        </div>
      ))}
    </div>
  );
}
