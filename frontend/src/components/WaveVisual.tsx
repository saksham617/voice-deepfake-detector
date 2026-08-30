/**
 * WaveVisual — the glowing, flowing green waveform used as the hero/loading
 * visual. Pure SVG + CSS animation (no dependencies).
 *
 * Each layer is a sine wave drawn across 2x the viewBox width so translating it
 * by exactly one base-width loops seamlessly. Layers use different amplitudes,
 * speeds and directions to read as an organic, living waveform.
 */

const VIEW_W = 800;
const VIEW_H = 220;
const MID = VIEW_H / 2;

/** Build a seamlessly-tileable sine path across 2 * VIEW_W. */
function wavePath(amplitude: number, periodsOverView: number): string {
  const totalW = VIEW_W * 2;
  // periodsOverView must be an integer so the wave tiles at translateX(-VIEW_W).
  const k = (Math.PI * 2 * periodsOverView) / VIEW_W;
  let d = `M 0 ${MID}`;
  for (let x = 0; x <= totalW; x += 8) {
    const y = MID + Math.sin(x * k) * amplitude;
    d += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return d;
}

interface WaveVisualProps {
  /** Boosts amplitude/opacity while audio is being analyzed. */
  active?: boolean;
  className?: string;
}

export function WaveVisual({ active = false, className }: WaveVisualProps) {
  const amp = active ? 1.35 : 1;
  const layers = [
    { d: wavePath(46 * amp, 2), stroke: "url(#waveGrad)", width: 2.5, anim: "animate-flow-1", opacity: 0.95 },
    { d: wavePath(30 * amp, 3), stroke: "url(#waveGrad)", width: 2, anim: "animate-flow-2", opacity: 0.6 },
    { d: wavePath(60 * amp, 1), stroke: "url(#waveGrad2)", width: 1.5, anim: "animate-flow-3", opacity: 0.4 },
  ];

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      preserveAspectRatio="none"
      className={className}
      role="img"
      aria-label={active ? "Audio being analyzed" : "Audio waveform"}
    >
      <defs>
        <linearGradient id="waveGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#0e6b4c" />
          <stop offset="35%" stopColor="#2fe3a2" />
          <stop offset="65%" stopColor="#7ef7c8" />
          <stop offset="100%" stopColor="#0e6b4c" />
        </linearGradient>
        <linearGradient id="waveGrad2" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#2fe3a2" stopOpacity="0.2" />
          <stop offset="50%" stopColor="#2fe3a2" />
          <stop offset="100%" stopColor="#2fe3a2" stopOpacity="0.2" />
        </linearGradient>
        <filter id="waveGlow" x="-20%" y="-60%" width="140%" height="220%">
          <feGaussianBlur stdDeviation={active ? 5 : 3} result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <g filter="url(#waveGlow)">
        {layers.map((layer, i) => (
          <path
            key={i}
            d={layer.d}
            fill="none"
            stroke={layer.stroke}
            strokeWidth={layer.width}
            strokeLinecap="round"
            opacity={layer.opacity}
            className={layer.anim}
            style={{ willChange: "transform" }}
          />
        ))}
      </g>
    </svg>
  );
}
