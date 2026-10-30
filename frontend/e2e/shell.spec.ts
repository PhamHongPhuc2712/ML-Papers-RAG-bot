import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolations, openAccountMenu, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("the sidebar groups chats by day, collapses to a rail and remembers it", async ({ page }) => {
  await signIn(page, "/");
  await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  const chats = page.getByRole("navigation", { name: "Chats" });
  await expect(chats.getByRole("region", { name: "Today" }).getByRole("link")).toHaveText(["How are sentence pairs formed?"]);
  await noSeriousViolations(page);

  await page.getByRole("button", { name: "Collapse sidebar" }).click();
  await expect(chats).toBeHidden();
  await page.reload();
  await expect(page.getByRole("button", { name: "Expand sidebar" })).toBeVisible();
  await expect(chats).toBeHidden();
  // The rail keeps New chat reachable by its name.
  await expect(page.getByRole("link", { name: "New chat" })).toBeVisible();
  await page.getByRole("button", { name: "Expand sidebar" }).click();
  await expect(chats).toBeVisible();
});

test("the theme follows the system, and a chosen theme is kept", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await signIn(page, "/");
  const background = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(await background()).toBe("rgb(26, 28, 34)");
  await openAccountMenu(page);
  await page.getByRole("group", { name: "Theme" }).getByRole("button", { name: "Light" }).click();
  expect(await background()).toBe("rgb(255, 255, 255)");
  await page.reload();
  expect(await background()).toBe("rgb(255, 255, 255)");
  await openAccountMenu(page);
  await expect(page.getByRole("button", { name: "Light" })).toHaveAttribute("aria-pressed", "true");
  await noSeriousViolations(page);
  await page.getByRole("button", { name: "System" }).click();
  expect(await background()).toBe("rgb(26, 28, 34)");
  await noSeriousViolations(page);
});

test("the account menu closes with Escape and returns focus", async ({ page }) => {
  await signIn(page, "/");
  await openAccountMenu(page);
  await expect(page.getByRole("link", { name: "Settings" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("link", { name: "Settings" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /^Account menu/ })).toBeFocused();
});

test.describe("on a phone", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("the sidebar is a drawer that opens, closes with Escape, and closes on navigation", async ({ page }) => {
    await signIn(page, "/");
    await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
    await page.keyboard.press("Enter");
    await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
    const chats = page.getByRole("navigation", { name: "Chats" });
    await expect(chats).toBeHidden();
    await page.getByRole("button", { name: "Open menu" }).click();
    await expect(chats).toBeVisible();
    await noSeriousViolations(page);
    await page.keyboard.press("Escape");
    await expect(chats).toBeHidden();
    await expect(page.getByRole("button", { name: "Open menu" })).toBeFocused();
    await page.getByRole("button", { name: "Open menu" }).click();
    await page.getByRole("link", { name: "New chat" }).click();
    await expect(page).toHaveURL(/\/$/);
    await expect(chats).toBeHidden();
  });
});
