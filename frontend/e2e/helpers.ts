import AxeBuilder from "@axe-core/playwright";
import { expect, type Page } from "@playwright/test";

export const PASSWORD = "correct horse battery staple";
export const MOCK = "http://127.0.0.1:8799";

export async function signIn(page: Page, next = "/chat") {
  await page.goto(`/login?next=${encodeURIComponent(next)}`);
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Log in" }).click();
  await page.waitForURL(`**${next}`);
}

/** Sign out and the account page live in the sidebar's account menu (UI redesign, D1). */
export async function openAccountMenu(page: Page) {
  await page.getByRole("button", { name: /^Account menu/ }).click();
}

export async function noSeriousViolations(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious.map((v) => v.id)).toEqual([]);
}

/** Axe in the light theme, then in the dark one (the system setting), then light again. */
export async function noSeriousViolationsInBothThemes(page: Page) {
  await page.emulateMedia({ colorScheme: "light" });
  await noSeriousViolations(page);
  await page.emulateMedia({ colorScheme: "dark" });
  await expect.poll(() => page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe("rgb(26, 28, 34)");
  await noSeriousViolations(page);
  await page.emulateMedia({ colorScheme: "light" });
}
