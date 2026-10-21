import { expect, test } from "@playwright/test";
import { CODE, MOCK, noSeriousViolationsInBothThemes, openAccountMenu, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

const formError = (page: import("@playwright/test").Page) =>
  // Next renders its own route announcer with role="alert"; this is the form's.
  page.getByRole("alert").and(page.locator("#login-error"));

test("a new email signs up with the code sent to it, and the account is its own", async ({ page }) => {
  await page.goto("/login?next=%2Fchat");
  await expect(page.getByRole("heading", { name: "Log in or sign up" })).toBeVisible();
  await noSeriousViolationsInBothThemes(page);
  await page.getByLabel("Email").fill("New.Person@Example.org");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByText("We sent a 6-digit code to")).toContainText("new.person@example.org");
  await expect(page.getByRole("button", { name: /Send a new code in \d+ s/ })).toBeDisabled();
  await noSeriousViolationsInBothThemes(page);
  await page.getByLabel("Code").fill(CODE);
  await page.getByRole("button", { name: "Log in" }).click();
  await page.waitForURL("**/chat");
  await openAccountMenu(page);
  await expect(page.getByRole("button", { name: /^Account menu/ })).toContainText("new.person@example.org");
});

test("a wrong, a used-up and an expired code each say what to do next", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByRole("button", { name: "Continue" }).click();
  const cases: [string, string][] = [
    ["135790", "That code is not right. Check it and try again."],
    ["111111", "Too many wrong tries. Ask for a new code."],
    ["000000", "That code has expired or was already used. Ask for a new code."],
  ];
  for (const [code, message] of cases) {
    await page.getByLabel("Code").fill(code);
    await page.getByRole("button", { name: "Log in" }).click();
    await expect(formError(page)).toHaveText(message);
  }
  await expect(page).toHaveURL(/\/login/);
});

test("asking again within a minute keeps the code step and says how long to wait", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByRole("link", { name: "Use another email" }).click();
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(formError(page)).toContainText("A code was sent less than a minute ago");
  await expect(page.getByLabel("Code")).toBeVisible();
});

test("a malformed email is refused on the email step", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill("not-an-email@x");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(formError(page)).toHaveText("Enter a valid email address.");
});

test("the signed-out sidebar offers Sign up and Log in, both to the same page", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Sign up" }).click();
  await expect(page.getByRole("heading", { name: "Log in or sign up" })).toBeVisible();
  await page.goto("/");
  await page.getByRole("link", { name: "Log in", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Log in or sign up" })).toBeVisible();
});

test("signing in keeps the token out of page scripts", async ({ page, context }) => {
  await signIn(page);
  const cookie = (await context.cookies()).find((c) => c.name === "copilot_session");
  expect(cookie?.httpOnly).toBe(true);
  expect((await context.cookies()).find((c) => c.name === "copilot_login_email")).toBeUndefined();
  expect(await page.evaluate(() => document.cookie)).not.toContain("copilot_session");
  const leak = await page.evaluate(async () => (await fetch("/api/v1/auth/token", { method: "POST" })).status);
  expect(leak).toBe(404);
});

test("chat needs a session and signing out ends it", async ({ page }) => {
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/login\?next=%2Fchat/);
  await signIn(page);
  await openAccountMenu(page);
  await page.getByRole("button", { name: "Sign out" }).click();
  await page.waitForURL("**/login");
  await page.goto("/chat");
  await expect(page).toHaveURL(/\/login/);
});

test.describe("before the page's script runs", () => {
  test.use({ javaScriptEnabled: false });

  test("both steps post their forms, the code never enters a URL, and it lands on next", async ({ page }) => {
    await page.goto("/login?next=%2Fhistory");
    await page.getByLabel("Email").fill("alice@example.org");
    await page.getByRole("button", { name: "Continue" }).click();
    await page.waitForURL(/\/login\?next=%2Fhistory&step=code$/);
    await expect(page.getByText("We sent a 6-digit code to")).toContainText("alice@example.org");
    await page.getByLabel("Code").fill(CODE);
    await page.getByRole("button", { name: "Log in" }).click();
    await page.waitForURL("**/history");
    expect(page.url()).not.toContain(CODE);
  });

  test("a wrong code comes back as the form's error, without the code", async ({ page }) => {
    await page.goto("/login?next=%2Fhistory");
    await page.getByLabel("Email").fill("alice@example.org");
    await page.getByRole("button", { name: "Continue" }).click();
    await page.getByLabel("Code").fill("135790");
    await page.getByRole("button", { name: "Log in" }).click();
    await page.waitForURL(/\/login\?next=%2Fhistory&step=code&error=invalid_code$/);
    await expect(page.locator("#login-error")).toHaveText("That code is not right. Check it and try again.");
    expect(page.url()).not.toContain("135790");
  });
});

test("an open redirect is refused", async ({ page }) => {
  await signIn(page, "/chat");
  await page.goto("/login?next=//evil.example.org");
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByLabel("Code").fill(CODE);
  await page.getByRole("button", { name: "Log in" }).click();
  await page.waitForURL((url) => url.hostname === "127.0.0.1" && url.pathname === "/");
});
