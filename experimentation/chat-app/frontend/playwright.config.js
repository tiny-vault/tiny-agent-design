import { defineConfig, devices } from "@playwright/test";

// Two projects:
//   mocked  Fast and deterministic. Runs the CURRENT source on a Vite dev server
//           that Playwright starts itself, with the API stubbed in the browser.
//           Needs no Docker. Covers model-switching gating and chat edge cases.
//   live    Drives the REAL stack through the gateway (node ../../../execute.mjs up):
//           real router, real models. Slow, because switching loads models.
// Run one with:  npx playwright test --project=mocked
const DEV_PORT = 5174;
const gateway = process.env.E2E_BASE_URL || "http://localhost:3200";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  workers: 1,
  reporter: [["list"]],
  use: { trace: "retain-on-failure", screenshot: "only-on-failure" },
  webServer: {
    command: `npm run dev -- --port ${DEV_PORT} --strictPort`,
    url: `http://localhost:${DEV_PORT}`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
  projects: [
    {
      name: "mocked",
      testMatch: /(gating|edge)\.spec\.js/,
      use: { ...devices["Desktop Chrome"], baseURL: `http://localhost:${DEV_PORT}` },
    },
    {
      name: "live",
      testMatch: /live\.spec\.js/,
      timeout: 600_000,
      use: { ...devices["Desktop Chrome"], baseURL: gateway },
    },
  ],
});
