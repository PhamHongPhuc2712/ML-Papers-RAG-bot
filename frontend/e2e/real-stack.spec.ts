// Run by hand against the real local stack (REAL_STACK=1). It signs in with REAL_EMAIL and
// REAL_PASSWORD, asks two turns about a paper in our corpus, and saves screenshots. It spends
// about $0.01 on the hosted model and needs an idle GPU.
import { expect, test } from "@playwright/test";

test("search, a paper, and a two-turn chat on the real stack", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/?q=contrastive+learning+of+sentence+embeddings");
  await expect(page.getByRole("article").first()).toBeVisible({ timeout: 30_000 });
  await page.screenshot({ path: "test-results/real-search.png", fullPage: true });
  await page.getByRole("article").first().getByRole("link").first().click();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.goto("/login?next=%2Fchat");
  await page.getByLabel("Email").fill(process.env.REAL_EMAIL ?? "");
  await page.getByLabel("Password").fill(process.env.REAL_PASSWORD ?? "");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL("**/chat");
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("What loss does SimCSE use to train sentence embeddings?");
  await box.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1, { timeout: 90_000 });
  await box.fill("How large was its batch size?");
  await box.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(2, { timeout: 90_000 });
  // Open the first answer's first citation. With condensing off (C7's rule, configs/chat.yaml)
  // the follow-up is searched as typed and may come back insufficient, with nothing to cite.
  await page.getByRole("article", { name: "Answer" }).first().getByRole("button").first().click();
  await page.screenshot({ path: "test-results/real-chat.png", fullPage: true });
});
