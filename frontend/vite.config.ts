import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies /api to the FastAPI backend so the browser never needs CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
