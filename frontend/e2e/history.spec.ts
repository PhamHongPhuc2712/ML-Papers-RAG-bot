import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolationsInBothThemes, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("History needs a session", async ({ page }) => {
  await page.goto("/history");
  await expect(page).toHaveURL(/\/login\?next=%2Fhistory/);
});

test("History groups chats by day with their age, and searches their titles", async ({ page, request }) => {
  await request.post(`${MOCK}/__control/seed-history`);
  await signIn(page, "/");
  await page.getByRole("link", { name: "History" }).click();
  await page.waitForURL("**/history");
  await expect(page.getByRole("link", { name: "History" })).toHaveAttribute("aria-current", "page");
  const main = page.getByRole("main");
  await expect(main.getByRole("heading", { level: 2 })).toHaveText(["Today", "Previous 7 days", "Older"]);
  const today = main.getByRole("region", { name: "Today" });
  await expect(today.getByRole("link")).toHaveText(["What loss does SimCSE use?"]);
  await expect(today).toContainText("1 hour ago");
  await expect(main.getByRole("region", { name: "Previous 7 days" })).toContainText("3 days ago");
  await noSeriousViolationsInBothThemes(page);

  const search = page.getByLabel("Search your chats");
  await search.fill("graph MOLECULES");
  await expect(main.getByRole("listitem").getByRole("link")).toHaveText(["Graph transformers for molecules"]);
  await search.fill("protein folding");
  await expect(main).toContainText("No chat titles match “protein folding”.");
  await search.fill("");
  await main.getByRole("link", { name: "Speculative decoding speedups" }).click();
  await page.waitForURL(/\/chat\/[0-9a-f-]{36}$/);
  await expect(page.getByRole("list", { name: "Conversation" })).toContainText("Speculative decoding speedups");
});

test("a chat deleted on History leaves the page and the sidebar", async ({ page, request }) => {
  await request.post(`${MOCK}/__control/seed-history`);
  await signIn(page, "/history");
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("main").getByRole("button", { name: "Delete chat What loss does SimCSE use?" }).click();
  await expect(page.getByRole("main").getByRole("region", { name: "Today" })).toHaveCount(0);
  await expect(page.getByRole("navigation", { name: "Chats" }).getByRole("link")).toHaveCount(2);
});

test("no chats yet, and an outage, are each said", async ({ page, request }) => {
  await signIn(page, "/history");
  await expect(page.getByRole("main")).toContainText("Your chats appear here after your first question.");
  await page.getByRole("main").getByRole("link", { name: "Ask a question" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  await request.post(`${MOCK}/__control/database?down=1`);
  await page.goto("/history");
  const alert = page.getByRole("main").getByRole("alert");
  await expect(alert).toContainText("The service can’t be reached right now. Your chats are safe.");
  await request.post(`${MOCK}/__control/database?down=0`);
  await alert.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByRole("main")).toContainText("Your chats appear here after your first question.");
});
