import { defineConfig, devices } from "@playwright/test";

// End-to-end tests run against the real backend with the SYNTHETIC cohort.
// Prepare it first with `make e2e-prepare` (seed, index, run the smoke suite).
const backendPort = Number(process.env.E2E_BACKEND_PORT ?? 8010);
const frontendPort = Number(process.env.E2E_FRONTEND_PORT ?? 5183);
const python = process.env.E2E_PYTHON ?? "python";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${frontendPort}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `${python} -m app.cli serve --port ${backendPort}`,
      cwd: "../backend",
      url: `http://127.0.0.1:${backendPort}/api/v1/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: `npm run dev -- --port ${frontendPort} --strictPort --host 127.0.0.1`,
      url: `http://127.0.0.1:${frontendPort}`,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      env: {
        VITE_BACKEND_URL: `http://127.0.0.1:${backendPort}`,
        VITE_API_KEY: process.env.E2E_API_KEY ?? "dev-research-key",
      },
    },
  ],
});
