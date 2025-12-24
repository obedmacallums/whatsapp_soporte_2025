import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../docs', // Cambia la salida a /docs
    emptyOutDir: true
  },
  server: {
    port: 5173,
    open: true
  },
  base: './'
})
