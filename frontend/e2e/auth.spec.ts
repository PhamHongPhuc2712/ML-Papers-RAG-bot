import { expect, test } from "@playwright/test";
import { MOCK, PASSWORD, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("a wrong password says so and stays on the page", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByLabel("Password").fill("not the password at all");
  await page.getByRole("button", { name: "Sign in" }).click();
  // Next renders its own route announcer with role="alert"; this is the form's.
  await expect(page.getByRole("alert").and(page.locator("#login-error"))).toHaveText(
    "Email or password is incorrect.",
  );
  await expect(page).toHaveURL(/\/login/);
});

test("signing in keeps the token out of page scripts", async ({ page, context }) => {
  await signIn(page);
  const cookie = (await context.cookies()).find((c) => c.name === "copilot_session");
  expect(cookie?.httpOnly).toBe(true);
  expect(await page.evaluate(() => document.cookie)).not.toContain("copilot_session");
  const leak = await page.evaluate(async () => (await fetch("/api/v1/auth/token", { method: "POST" })).status);
  expect(leak).toBe(404);
});

test("chat needs a session and signing out ends it", async ({ page }) => {
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/login\?next=%2Fchat/);
  await signIn(page);
  await page.getByRole("button", { name: "Sign out" }).click();
  await page.waitForURL("**/login");
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/login/);
});

test("an open redirect is refused", async ({ page }) => {
  await signIn(page, "/chat");
  await page.goto("/login?next=//evil.example.org");
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL((url) => url.hostname === "127.0.0.1" && url.pathname === "/");
});
