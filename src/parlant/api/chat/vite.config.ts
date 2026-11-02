import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  // Relative so the built page works wherever it is mounted, including behind a
  // path prefix such as host.com/{projectId}/chat.
  base: './',
  test: {
    globals: true,
    environment: 'jsdom',
    includeSource: ['app/**/*.{jsx,tsx}'],
    setupFiles: ['./setupTests.ts']
  },
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 8002,
    host: '127.0.0.1'
  }
});
