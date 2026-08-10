import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Pinned so the backend's CORS allowlist always matches. `strictPort` makes
    // a port conflict fail loudly instead of silently moving to 5174, which
    // would then be blocked by CORS.
    port: 5173,
    strictPort: true,
  },
})
