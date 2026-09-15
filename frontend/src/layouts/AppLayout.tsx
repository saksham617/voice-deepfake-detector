import { Outlet } from "react-router-dom";
import { Sidebar } from "../components/Sidebar";

/** Persistent shell: sidebar stays mounted, only the routed page changes. */
export function AppLayout() {
  return (
    <div className="app-backdrop min-h-screen flex">
      <Sidebar />
      <main className="flex-1 min-w-0 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
