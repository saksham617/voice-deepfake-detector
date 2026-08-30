import { Header } from "./components/Header";
import { AnalyzerPage } from "./pages/AnalyzerPage";

export default function App() {
  return (
    <div className="app-backdrop min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <AnalyzerPage />
      </main>
      <footer className="py-6 text-center text-xs text-slate-500">
        VoiceGuard · Voice Deepfake Detection · SIH Demonstration
      </footer>
    </div>
  );
}
