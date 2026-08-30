/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Neutral "security product" surface palette
        surface: {
          950: "#0a0d14",
          900: "#0f131c",
          800: "#161b26",
          700: "#1f2633",
          600: "#2a3342",
          500: "#3a4556",
        },
        accent: {
          DEFAULT: "#4f8cff",
          hover: "#3d7bf5",
        },
        bonafide: "#22c55e",
        spoof: "#ef4444",
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
      },
    },
  },
  plugins: [],
};
