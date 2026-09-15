/** Explains what the fake-probability gauge means, before showing the score itself. */
export function MiniLegend() {
  return (
    <div className="rounded-xl border border-line bg-panel-raised px-4 py-3 text-xs leading-relaxed text-ink-dim">
      <span className="font-medium text-ink">Fake probability</span> shows how
      confident the model is that this audio is AI-generated — 0 means
      definitely real, 100 means definitely synthetic.
    </div>
  );
}
