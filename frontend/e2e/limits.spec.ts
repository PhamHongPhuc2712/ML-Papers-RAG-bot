import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolationsInBothThemes, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("the ask box counts down the last answers of the day", async ({ page, request }) => {
  await request.post(`${MOCK}/__control/answered?n=17`);
  await signIn(page, "/");
  await expect(page.getByRole("status").filter({ hasText: "answers left today" })).toHaveText("3 answers left today");
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("How are sentence pairs used?");
  await box.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  await expect(page.getByText("2 answers left today")).toBeVisible();
});

test("with no answers left, asking says so, keeps the question, and stores nothing", async ({ page, request }) => {
  await request.post(`${MOCK}/__control/answered?n=20`);
  await signIn(page, "/");
  await expect(page.getByText("No answers left today")).toBeVisible();
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("How are sentence pairs used?");
  await box.press("Enter");
  const banner = page.getByRole("status").filter({ hasText: "Daily limit reached." });
  await expect(banner).toContainText("You have used today's 20 answers. You can ask again after 00:00 UTC");
  await expect(box).toHaveValue("How are sentence pairs used?");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(0);
  await noSeriousViolationsInBothThemes(page);
});
