import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ask': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/providers': 'http://localhost:8000',
    }
  }
})
