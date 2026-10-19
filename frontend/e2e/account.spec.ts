import { expect, test } from "@playwright/test";
import { MOCK, PASSWORD, noSeriousViolationsInBothThemes, openAccountMenu, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("the settings page needs a session", async ({ page }) => {
  await page.goto("/account");
  await expect(page).toHaveURL(/\/login\?next=%2Faccount/);
});

test("settings show the account and keep a chosen theme", async ({ page }) => {
  await signIn(page, "/account");
  await expect(page.getByRole("heading", { name: "Settings", level: 1 })).toBeVisible();
  await expect(page.getByRole("region", { name: "Account" })).toContainText("alice@example.org");
  await noSeriousViolationsInBothThemes(page);
  await page.getByRole("region", { name: "Theme" }).getByRole("button", { name: "Dark" }).click();
  await page.reload();
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe("rgb(26, 28, 34)");
  await expect(page.getByRole("region", { name: "Theme" }).getByRole("button", { name: "Dark" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("region", { name: "Account" }).getByRole("button", { name: "Log out" }).click();
  await page.waitForURL("**/login");
});

test("a wrong password keeps the account and the session, and Cancel closes the form", async ({ page }) => {
  await signIn(page, "/account");
  await page.getByRole("button", { name: "Delete account" }).click();
  await expect(page.getByLabel("Password")).toBeFocused();
  await page.getByLabel("Password").fill("not the password at all");
  await page.getByRole("button", { name: "Delete my account" }).click();
  await expect(page.locator("#delete-error")).toHaveText("The password is incorrect.");
  await noSeriousViolationsInBothThemes(page);
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByRole("button", { name: "Delete account" })).toBeFocused();
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/chat$/);
});

test("deleting the account signs out and the session is gone", async ({ page }) => {
  await signIn(page, "/chat");
  await openAccountMenu(page);
  await page.getByRole("link", { name: "Settings" }).click();
  await page.waitForURL("**/account");
  await page.getByRole("button", { name: "Delete account" }).click();
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Delete my account" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  await expect(page.getByRole("link", { name: "Log in", exact: true })).toBeVisible();
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/login\?next=%2Fchat/);
});
