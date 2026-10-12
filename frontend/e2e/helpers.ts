import AxeBuilder from "@axe-core/playwright";
import { expect, type Page } from "@playwright/test";

export const PASSWORD = "correct horse battery staple";
export const MOCK = "http://127.0.0.1:8799";

export async function signIn(page: Page, next = "/chat") {
  await page.goto(`/login?next=${encodeURIComponent(next)}`);
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL(`**${next}`);
}

export async function noSeriousViolations(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious.map((v) => v.id)).toEqual([]);
}
