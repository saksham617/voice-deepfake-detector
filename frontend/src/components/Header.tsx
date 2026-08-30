import { ShieldIcon } from "./icons";

export function Header() {
  return (
    <header className="border-b border-white/5">
      <div className="mx-auto max-w-5xl px-4 sm:px-6 py-4 flex items-center gap-3">
        <span className="grid place-items-center h-9 w-9 rounded-lg bg-accent/15 text-accent">
          <ShieldIcon className="h-5 w-5" />
        </span>
        <div className="leading-tight">
          <h1 className="text-base font-semibold text-white tracking-tight">
            VoiceGuard
          </h1>
          <p className="text-xs text-slate-400">Voice Deepfake Detection</p>
        </div>
        <span className="ml-auto text-[11px] font-medium uppercase tracking-wider text-slate-500 border border-white/10 rounded-full px-2.5 py-1">
          SIH Demo
        </span>
      </div>
    </header>
  );
}
