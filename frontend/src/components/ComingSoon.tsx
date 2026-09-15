interface ComingSoonProps {
  title: string;
  description?: string;
}

/** Placeholder shown on routes that aren't built out yet. */
export function ComingSoon({ title, description }: ComingSoonProps) {
  return (
    <div className="min-h-full bg-canvas flex items-center justify-center px-4 sm:px-6 py-16 font-plexSans">
      <div className="mx-auto max-w-2xl text-center">
        <span className="inline-flex items-center gap-2 rounded-full border border-line bg-panel px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-ink-dim">
          Coming soon
        </span>
        <h2 className="mt-4 text-2xl sm:text-3xl font-semibold tracking-tight text-ink">
          {title}
        </h2>
        <p className="mx-auto mt-3 max-w-lg text-sm text-ink-dim">
          {description ?? "This screen hasn't been built yet."}
        </p>
      </div>
    </div>
  );
}
