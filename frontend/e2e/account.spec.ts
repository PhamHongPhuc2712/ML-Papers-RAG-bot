import { expect, test } from "@playwright/test";
import { MOCK, PASSWORD, noSeriousViolations, openAccountMenu, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("the account page needs a session", async ({ page }) => {
  await page.goto("/account");
  await expect(page).toHaveURL(/\/login\?next=%2Faccount/);
});

test("a wrong password keeps the account and the session", async ({ page }) => {
  await signIn(page, "/account");
  await expect(page.getByRole("heading", { name: "Account", level: 1 })).toBeVisible();
  await noSeriousViolations(page);
  await page.getByLabel("Password").fill("not the password at all");
  await page.getByRole("button", { name: "Delete my account" }).click();
  await expect(page.locator("#delete-error")).toHaveText("The password is incorrect.");
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/chat$/);
});

test("deleting the account signs out and the session is gone", async ({ page }) => {
  await signIn(page, "/chat");
  await openAccountMenu(page);
  await page.getByRole("link", { name: "Account" }).click();
  await page.waitForURL("**/account");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Delete my account" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  await expect(page.getByRole("link", { name: "Log in", exact: true })).toBeVisible();
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/login\?next=%2Fchat/);
});
