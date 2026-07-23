import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/** FastAPI origin used by the dev server proxy. */
const API_ORIGIN = process.env.VITE_API_ORIGIN ?? "http://127.0.0.1:8000";

/** Paths owned by the FastAPI service — proxied verbatim during development. */
const PROXIED_PREFIXES = ["/v1", "/api", "/healthz", "/metrics"];

export default defineConfig({
  // The production bundle is served by FastAPI from server/static/app.
  base: "/static/app/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      PROXIED_PREFIXES.map((prefix) => [
        prefix,
        { target: API_ORIGIN, changeOrigin: true },
      ]),
    ),
  },
  build: {
    outDir: path.resolve(import.meta.dirname, "../server/static/app"),
    emptyOutDir: true,
    sourcemap: false,
    // Split vendor code so route chunks stay small and cache well across deploys.
    rollupOptions: {
      output: {
        manualChunks(id) {
          const path = id.replaceAll("\\", "/");
          if (!path.includes("/node_modules/")) return;

          // Charts are only reachable from the lazily-loaded Performance route.
          if (/\/node_modules\/(recharts|d3-|victory-vendor|decimal\.js)/.test(path)) {
            return "charts";
          }
          if (/\/node_modules\/(framer-motion|motion-dom|motion-utils)\//.test(path)) {
            return "motion";
          }
          if (/\/node_modules\/(react-router|react-router-dom|@remix-run)\//.test(path)) {
            return "router";
          }
          if (path.includes("/node_modules/@tanstack/")) return "query";
          // Keep scheduler with React — splitting them creates a chunk cycle.
          if (/\/node_modules\/(react|react-dom|scheduler)\//.test(path)) return "react";
          return "vendor";
        },
      },
    },
  },
});
