/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Deep teal-green "voice product" surface palette.
        surface: {
          950: "#050f0b",
          900: "#081712",
          800: "#0d211a",
          700: "#123026",
          600: "#1a4234",
          500: "#255747",
        },
        // Neon-mint accent with glow.
        accent: {
          DEFAULT: "#2fe3a2",
          hover: "#25c98b",
          soft: "#7ef7c8",
        },
        bonafide: "#2fe3a2",
        spoof: "#ff5b6e",

        // New design-system palette, backed by the CSS variables in index.css.
        // Namespaced separately from `surface`/`accent` above so pages keep
        // rendering with the old tokens until they're migrated one at a time
        // (Sidebar/ComingSoon use these now; Voice Check still uses the old
        // ones until its own redesign pass). Note: the spec's own `--accent`
        // CSS variable (light blue, for links/info) is exposed here as `info`
        // to avoid clashing with the existing teal `accent` above.
        canvas: "var(--bg)",
        panel: {
          DEFAULT: "var(--panel)",
          raised: "var(--panel-raised)",
        },
        line: {
          DEFAULT: "var(--border)",
          strong: "var(--border-strong)",
        },
        ink: {
          DEFAULT: "var(--text)",
          dim: "var(--text-dim)",
          faint: "var(--text-faint)",
        },
        safe: {
          DEFAULT: "var(--safe)",
          dim: "var(--safe-dim)",
        },
        medium: {
          DEFAULT: "var(--medium)",
          dim: "var(--medium-dim)",
        },
        high: {
          DEFAULT: "var(--high)",
          dim: "var(--high-dim)",
        },
        info: "var(--accent)",
      },
      fontFamily: {
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],

        // New design-system typefaces (Sidebar/ComingSoon and pages migrated
        // after them). Loaded via Google Fonts in index.html.
        plexSans: [
          "IBM Plex Sans",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        plexMono: [
          "IBM Plex Mono",
          "ui-monospace",
          "SFMono-Regular",
          "monospace",
        ],
      },
      keyframes: {
        flow1: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-800px)" },
        },
        flow2: {
          "0%": { transform: "translateX(-800px)" },
          "100%": { transform: "translateX(0)" },
        },
        glowpulse: {
          "0%, 100%": { opacity: "0.55" },
          "50%": { opacity: "1" },
        },
        floaty: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-4px)" },
        },
      },
      animation: {
        "flow-1": "flow1 11s linear infinite",
        "flow-2": "flow2 17s linear infinite",
        "flow-3": "flow1 7s linear infinite",
        glowpulse: "glowpulse 2.6s ease-in-out infinite",
        floaty: "floaty 4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
