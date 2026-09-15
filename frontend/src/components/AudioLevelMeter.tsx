import { useEffect, useRef } from "react";

interface AudioLevelMeterProps {
  analyser: AnalyserNode | null;
}

/**
 * Live microphone level bar. Writes directly to the DOM inside the
 * requestAnimationFrame loop instead of React state, so ~60fps updates don't
 * churn re-renders.
 */
export function AudioLevelMeter({ analyser }: AudioLevelMeterProps) {
  const barRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    let raf = 0;

    const tick = () => {
      analyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i];
      const average = sum / data.length; // 0..255
      const pct = Math.min(100, (average / 160) * 100);
      if (barRef.current) barRef.current.style.width = `${pct}%`;
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => cancelAnimationFrame(raf);
  }, [analyser]);

  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-panel">
      <div
        ref={barRef}
        className="h-full rounded-full bg-high transition-[width] duration-75"
        style={{ width: "0%" }}
      />
    </div>
  );
}
