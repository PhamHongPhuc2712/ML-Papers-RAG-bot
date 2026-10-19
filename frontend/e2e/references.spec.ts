import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolations, signIn } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("signed out, the home page asks first and keeps the question across log in; there is no search page", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("What do you want to know about ML research?");
  await expect(page.getByText("Searches the main-conference papers of ACL and ICLR from 2024 to 2025.")).toBeVisible();
  await expect(page.getByText("in 3 conference papers")).toBeVisible();
  await expect(page.getByRole("searchbox")).toHaveCount(0);
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link")).toHaveText(["ML Research Copilot"]);
  // Sign up appears with the sign-up plan.
  await expect(page.getByRole("link", { name: "Sign up" })).toHaveCount(0);
  await noSeriousViolations(page);
  // A filter chosen before asking is written into the question.
  await page.getByRole("button", { name: "Venues and years" }).click();
  await page.getByLabel("Venue").selectOption("ACL");
  await page.getByRole("button", { name: "Done" }).click();
  await expect(page.getByRole("button", { name: "ACL" })).toBeVisible();
  await page.getByLabel("Ask about ML papers").fill("Find papers on LoRA");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/login\?next=/);
  await expect(page.getByText("Your question is kept")).toBeVisible();
  await page.getByLabel("Email").fill("alice@example.org");
  await page.getByLabel("Password").fill("correct horse battery staple");
  await page.getByRole("button", { name: "Log in" }).click();
  await expect(page.getByLabel("Ask about ML papers")).toHaveValue("Find papers on LoRA\n\nOnly use papers from ACL.");
  expect(new URL(page.url()).pathname).toBe("/");
});

test("a suggested question is asked at once", async ({ page }) => {
  await signIn(page, "/");
  await page.getByRole("button", { name: /Find papers on contrastive learning/ }).click();
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  await expect(page).toHaveURL(/\/chat\/[0-9a-f-]{36}$/);
});

test("an answer says what it searched and lists its References, filterable", async ({ page }) => {
  await signIn(page, "/");
  const box = page.getByLabel("Ask about ML papers");
  // "slow" holds the turn open long enough to see the search step before the answer.
  await box.fill("Find papers on sentence embeddings, slowly");
  await box.press("Enter");
  await expect(page.getByRole("status")).toContainText("Searched for “Find papers on sentence embeddings, slowly” · reading 2 papers");
  const answer = page.getByRole("article", { name: "Answer" });
  await expect(answer).toContainText("Searched for “Find papers on sentence embeddings, slowly” · read 2 papers");
  await expect(page).toHaveURL(/\/chat\/[0-9a-f-]{36}$/);

  const references = page.getByRole("region", { name: "References" });
  await references.getByText("References (3)").click();
  const items = references.getByRole("listitem");
  await expect(items).toHaveCount(3);
  await expect(items.nth(0)).toContainText("ACL 2024 · read for this answer · cited [1]");
  await expect(items.nth(1)).toContainText("ICLR 2025 · read for this answer");
  await expect(items.nth(2)).toHaveText(/Sentence embeddings from language model prompts\s+ACL 2025$/);
  await expect(items.nth(0).getByRole("link")).toHaveAttribute("href", "/papers/aaaaaaaa-0000-4000-8000-000000000001");
  await noSeriousViolations(page);

  await references.getByLabel("Venue").selectOption("ACL");
  await expect(items).toHaveCount(2);
  await references.getByLabel("Year").selectOption("2025");
  await expect(items).toHaveCount(1);
  await expect(references).toContainText("Showing 1 of 3 papers.");
  await references.getByLabel("Venue").selectOption("");
  await references.getByLabel("Year").selectOption("");
  await expect(items).toHaveCount(3);

  // The References are stored with the answer.
  await page.reload();
  await page.getByRole("region", { name: "References" }).getByText("References (3)").click();
  await expect(page.getByRole("region", { name: "References" }).getByRole("listitem")).toHaveCount(3);
});

test("New chat goes back to the empty ask box", async ({ page }) => {
  await signIn(page, "/");
  await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  await page.getByRole("link", { name: "New chat" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(0);
});
