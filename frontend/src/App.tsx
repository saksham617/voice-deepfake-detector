import { Route, Routes } from "react-router-dom";
import { AppLayout } from "./layouts/AppLayout";
import { ActivityPage } from "./pages/ActivityPage";
import { AnalyzerPage } from "./pages/AnalyzerPage";
import { ContactsPage } from "./pages/ContactsPage";
import { LiveCallPage } from "./pages/LiveCallPage";
import { MessageCheckPage } from "./pages/MessageCheckPage";
import { NumberTimelinePage } from "./pages/NumberTimelinePage";
import { OverviewPage } from "./pages/OverviewPage";
import { ReportsPage } from "./pages/ReportsPage";
import { SettingsPage } from "./pages/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<OverviewPage />} />
        <Route path="voice-check" element={<AnalyzerPage />} />
        <Route path="message-check" element={<MessageCheckPage />} />
        <Route path="contacts" element={<ContactsPage />} />
        <Route path="live-call" element={<LiveCallPage />} />
        <Route path="reports" element={<ReportsPage />} />
        <Route path="reports/number/:phoneNumber" element={<NumberTimelinePage />} />
        <Route path="activity" element={<ActivityPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
