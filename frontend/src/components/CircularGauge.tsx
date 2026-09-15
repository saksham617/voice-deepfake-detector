interface CircularGaugeProps {
  /** 0-100 */
  value: number;
  /** Literal Tailwind stroke-color class, e.g. "stroke-safe". */
  strokeColorClass: string;
  size?: number;
}

/** SVG ring gauge showing a 0-100 score, e.g. fake probability. */
export function CircularGauge({ value, strokeColorClass, size = 96 }: CircularGaugeProps) {
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, value));
  const dash = (clamped / 100) * circumference;

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          className="stroke-line"
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          className={strokeColorClass}
          fill="none"
          strokeDasharray={`${dash} ${circumference - dash}`}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center">
        <span className="font-plexMono text-lg font-semibold text-ink">
          {Math.round(clamped)}
        </span>
      </div>
    </div>
  );
}
