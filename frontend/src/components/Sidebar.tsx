import { NavLink } from "react-router-dom";
import {
  ActivityIcon,
  ContactsIcon,
  HomeIcon,
  MessageIcon,
  PhoneIcon,
  ReportIcon,
  SettingsIcon,
  ShieldIcon,
  WaveIcon,
} from "./icons";

const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: HomeIcon, end: true },
  { to: "/voice-check", label: "Voice check", icon: WaveIcon, end: false },
  { to: "/message-check", label: "Message check", icon: MessageIcon, end: false },
  { to: "/contacts", label: "Contacts", icon: ContactsIcon, end: false },
  { to: "/live-call", label: "Live call", icon: PhoneIcon, end: false },
  { to: "/reports", label: "Reports", icon: ReportIcon, end: false },
  { to: "/activity", label: "Activity", icon: ActivityIcon, end: false },
] as const;

function navLinkClass({ isActive }: { isActive: boolean }) {
  return [
    "flex items-center gap-2.5 border-l-2 px-3 py-2 text-[13.5px] transition-colors",
    isActive
      ? "border-safe bg-safe-dim text-ink"
      : "border-transparent text-ink-dim hover:bg-panel-raised hover:text-ink",
  ].join(" ");
}

/** Persistent left-hand navigation. Stays mounted across every route. */
export function Sidebar() {
  return (
    <aside className="w-[220px] shrink-0 bg-panel border-r border-line flex flex-col font-plexSans">
      <div className="flex items-center gap-2.5 px-4 py-4 border-b border-line">
        <span className="grid place-items-center h-7 w-7 rounded-md bg-safe-dim text-safe">
          <ShieldIcon className="h-4 w-4" />
        </span>
        <span className="text-[14.5px] font-semibold text-ink tracking-tight">
          VoiceGuard
        </span>
      </div>

      <nav className="flex-1 px-2 py-3 space-y-0.5">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={navLinkClass}>
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-line px-2 py-2">
        <NavLink to="/settings" end={false} className={navLinkClass}>
          <SettingsIcon className="h-4 w-4" />
          Settings
        </NavLink>
      </div>

      <div className="flex items-center gap-2 px-4 py-4 border-t border-line">
        <span className="h-1.5 w-1.5 rounded-full bg-safe animate-glowpulse" />
        <span className="text-[12px] text-ink-dim">Detector online</span>
      </div>
    </aside>
  );
}
