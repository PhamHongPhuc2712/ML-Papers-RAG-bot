// Run by hand against the real local stack (REAL_STACK=1). It signs in with REAL_EMAIL and
// REAL_PASSWORD, asks for papers on a topic, opens one of its References, then asks a question and
// a follow-up, and saves screenshots. It spends about $0.02 on the hosted model and needs an idle GPU.
import { expect, test } from "@playwright/test";

test("a paper-finding request, a question and a follow-up on the real stack", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/login?next=%2F");
  await page.getByLabel("Email").fill(process.env.REAL_EMAIL ?? "");
  await page.getByLabel("Password").fill(process.env.REAL_PASSWORD ?? "");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  const box = page.getByLabel("Ask about ML papers");
  const answers = page.getByRole("article", { name: "Answer" });
  await box.fill("Find papers on contrastive learning of sentence embeddings");
  await box.press("Enter");
  await expect(answers).toHaveCount(1, { timeout: 90_000 });
  await page.getByRole("region", { name: "References" }).first().getByText(/^References/).click();
  await page.screenshot({ path: "test-results/real-papers.png", fullPage: true });
  await page.getByRole("region", { name: "References" }).first().getByRole("link").first().click();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.goto("/");
  await box.fill("What loss does SimCSE use to train sentence embeddings?");
  await box.press("Enter");
  await expect(answers).toHaveCount(1, { timeout: 90_000 });
  await box.fill("How large was its batch size?");
  await box.press("Enter");
  await expect(answers).toHaveCount(2, { timeout: 90_000 });
  // Open the first answer's first citation: the follow-up may still come back insufficient
  // (condensing rewrites it into a standalone query since 2026-10-14, configs/chat.yaml).
  await answers.first().getByRole("button").first().click();
  await page.screenshot({ path: "test-results/real-chat.png", fullPage: true });
});
