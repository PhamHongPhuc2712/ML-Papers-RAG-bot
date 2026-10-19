// Run by hand against the real local stack (REAL_STACK=1). It signs in with REAL_EMAIL and
// REAL_PASSWORD, asks for papers on a topic and a follow-up, opens a citation, the cited paper's page,
// History and Settings, and saves screenshots in light and dark (UI redesign, D4). Two turns: about
// $0.01 on the hosted model. Needs an idle GPU.
import { expect, test, type Page } from "@playwright/test";

const shot = async (page: Page, name: string) => {
  for (const scheme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme: scheme });
    await page.screenshot({ path: `test-results/real-${name}-${scheme}.png` });
  }
  await page.emulateMedia({ colorScheme: "light" });
};

test("an answer, its panel, a follow-up, the paper, History and Settings on the real stack", async ({ page }) => {
  test.setTimeout(240_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/login?next=%2F");
  await page.getByLabel("Email").fill(process.env.REAL_EMAIL ?? "");
  await page.getByLabel("Password").fill(process.env.REAL_PASSWORD ?? "");
  await page.getByRole("button", { name: "Log in" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  await shot(page, "home");

  const box = page.getByLabel("Ask about ML papers");
  const answers = page.getByRole("article", { name: "Answer" });
  await box.fill("Find papers on contrastive learning of sentence embeddings");
  await box.press("Enter");
  await expect(answers).toHaveCount(1, { timeout: 90_000 });
  await answers.first().getByRole("button", { name: /^References/ }).click();
  await expect(page.getByRole("complementary", { name: "References" })).toBeVisible();
  await shot(page, "references");

  await answers.first().getByRole("button", { name: "Source 1" }).click();
  const panel = page.getByRole("complementary", { name: "Paper" });
  await expect(panel.locator("mark")).toBeVisible();
  await shot(page, "citation");

  await box.fill("Which of them uses dropout as augmentation?");
  await box.press("Enter");
  await expect(answers).toHaveCount(2, { timeout: 90_000 });
  await shot(page, "follow-up");

  await page.getByRole("complementary").getByRole("link", { name: "Paper page" }).click();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.getByText(/^Asking about:/)).toBeVisible();
  await shot(page, "paper");

  await page.getByRole("link", { name: "History" }).click();
  await expect(page.getByRole("main").getByRole("listitem")).toHaveCount(1);
  await shot(page, "history");

  await page.goto("/account");
  await expect(page.getByRole("heading", { name: "Settings", level: 1 })).toBeVisible();
  await shot(page, "settings");
});
