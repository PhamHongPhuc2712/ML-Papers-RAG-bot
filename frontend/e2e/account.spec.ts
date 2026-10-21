import { expect, test } from "@playwright/test";
import { CODE, MOCK, noSeriousViolationsInBothThemes, openAccountMenu, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("the settings page needs a session", async ({ page }) => {
  await page.goto("/account");
  await expect(page).toHaveURL(/\/login\?next=%2Faccount/);
});

test("settings show the account, today's usage, and keep a chosen theme", async ({ page, request }) => {
  await request.post(`${MOCK}/__control/answered?n=7`);
  await signIn(page, "/account");
  await expect(page.getByRole("heading", { name: "Settings", level: 1 })).toBeVisible();
  await expect(page.getByRole("region", { name: "Account" })).toContainText("alice@example.org");
  await expect(page.getByRole("region", { name: "Usage" })).toContainText("7 of 20 answers used today, 7 this month.");
  await noSeriousViolationsInBothThemes(page);
  await page.getByRole("region", { name: "Theme" }).getByRole("button", { name: "Dark" }).click();
  await page.reload();
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe("rgb(26, 28, 34)");
  await expect(page.getByRole("region", { name: "Theme" }).getByRole("button", { name: "Dark" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("region", { name: "Account" }).getByRole("button", { name: "Log out" }).click();
  await page.waitForURL("**/login");
});

test("a wrong code keeps the account and the session, and Cancel closes the form", async ({ page }) => {
  await signIn(page, "/account");
  await page.getByRole("button", { name: "Delete account" }).click();
  await expect(page.getByLabel("Code")).toBeFocused();
  await page.getByLabel("Code").fill("135790");
  await page.getByRole("button", { name: "Delete my account" }).click();
  await expect(page.locator("#confirm-me-error")).toHaveText("That code is not right. Check it and try again.");
  await noSeriousViolationsInBothThemes(page);
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByRole("button", { name: "Delete account" })).toBeFocused();
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/chat$/);
});

test("deleting all chats with the emailed code empties the list and keeps the session", async ({ page, request }) => {
  await request.post(`${MOCK}/__control/seed-history`);
  await signIn(page, "/account");
  await expect(page.getByRole("navigation", { name: "Chats" }).getByRole("link")).not.toHaveCount(0);
  await page.getByRole("button", { name: "Delete all chats" }).click();
  await page.getByLabel("Code").fill(CODE);
  await page.getByRole("button", { name: "Delete all chats" }).click();
  await expect(page.getByRole("status").filter({ hasText: "All of your chats are deleted." })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Chats" }).getByRole("link")).toHaveCount(0);
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/chat$/);
});

test("deleting the account with the emailed code signs out and the session is gone", async ({ page }) => {
  await signIn(page, "/chat");
  await openAccountMenu(page);
  await page.getByRole("link", { name: "Settings" }).click();
  await page.waitForURL("**/account");
  await page.getByRole("button", { name: "Delete account" }).click();
  await page.getByLabel("Code").fill(CODE);
  await page.getByRole("button", { name: "Delete my account" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  await expect(page.getByRole("link", { name: "Log in", exact: true })).toBeVisible();
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/login\?next=%2Fchat/);
});
