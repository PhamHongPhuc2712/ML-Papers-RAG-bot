import { defineConfig, devices } from "@playwright/test";

const MOCK = "http://127.0.0.1:8799";

export default defineConfig({
  testDir: "e2e",
  testIgnore: process.env.REAL_STACK ? [] : ["real-stack.spec.ts"],
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: process.env.REAL_STACK ? (process.env.REAL_BASE_URL ?? "http://127.0.0.1:3000") : "http://127.0.0.1:3100", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: process.env.REAL_STACK
    ? undefined
    : [
        { command: "node e2e/mock-backend.mjs", url: `${MOCK}/v1/corpus/coverage`, reuseExistingServer: !process.env.CI },
        {
          command: "npx next dev --port 3100",
          url: "http://127.0.0.1:3100/login",
          env: { BACKEND_URL: MOCK },
          timeout: 120_000,
          reuseExistingServer: !process.env.CI,
        },
      ],
});
