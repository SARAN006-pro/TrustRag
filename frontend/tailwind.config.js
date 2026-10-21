/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gov: {
          dark: '#0f172a',
          navy: '#1e293b',
          border: '#334155',
          saffron: '#f97316',
          saffronLight: '#ffedd5',
          ashoka: '#2563eb',
          ashokaLight: '#dbeafe',
          emerald: '#10b981',
          emeraldLight: '#d1fae5',
        }
      }
    },
  },
  plugins: [],
}
