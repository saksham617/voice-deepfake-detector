import { MicIcon, UploadIcon } from "./icons";

/**
 * The audio input source. Adding microphone recording later means implementing
 * the "record" panel and enabling this tab — the rest of the flow (analyze,
 * loading, result) already works on any File, however it was produced.
 */
export type InputSource = "upload" | "record";

interface InputSourceTabsProps {
  active: InputSource;
  onChange: (source: InputSource) => void;
  disabled?: boolean;
}

export function InputSourceTabs({
  active,
  onChange,
  disabled,
}: InputSourceTabsProps) {
  const tabs: {
    id: InputSource;
    label: string;
    icon: typeof UploadIcon;
    comingSoon?: boolean;
  }[] = [
    { id: "upload", label: "Upload file", icon: UploadIcon },
    { id: "record", label: "Record", icon: MicIcon, comingSoon: true },
  ];

  return (
    <div
      role="tablist"
      aria-label="Audio input source"
      className="grid grid-cols-2 gap-1 rounded-xl bg-surface-800/60 p-1"
    >
      {tabs.map((tab) => {
        const isActive = active === tab.id;
        const isDisabled = disabled || tab.comingSoon;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            disabled={isDisabled}
            onClick={() => onChange(tab.id)}
            className={[
              "relative flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              isActive
                ? "bg-surface-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200",
              isDisabled ? "cursor-not-allowed opacity-60" : "",
            ].join(" ")}
          >
            <tab.icon className="h-4 w-4" />
            {tab.label}
            {tab.comingSoon && (
              <span className="ml-1 rounded-full bg-white/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-slate-300">
                Soon
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
