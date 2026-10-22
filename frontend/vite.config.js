import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@locales': path.resolve(__dirname, '../locales')
    }
  },
  server: {
    port: 3000,
    open: true,
    // The ngrok tunnel used to share the app (password-protected by its ngrok traffic policy).
    allowedHosts: ['nectar-unrivaled-spindle.ngrok-free.dev'],
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
