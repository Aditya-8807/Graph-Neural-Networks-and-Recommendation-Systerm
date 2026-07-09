import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// API_BASE in App.jsx is '' (relative paths like /api/...), so the dev server
// proxies those to the FastAPI backend instead of hitting the Vite origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
