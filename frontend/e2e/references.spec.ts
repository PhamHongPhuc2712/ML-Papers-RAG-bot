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

test("an answer shows its search live, then its References in the panel, filterable", async ({ page }) => {
  await signIn(page, "/");
  const box = page.getByLabel("Ask about ML papers");
  // "slow" holds the turn open long enough to see the search step before the answer.
  await box.fill("Find papers on sentence embeddings, slowly");
  await box.press("Enter");
  await expect(page.getByRole("status")).toContainText("Searched for “Find papers on sentence embeddings, slowly”, 3 papers ranked");
  await expect(page.getByRole("status")).toContainText("Reading the top 2 papers");
  const answer = page.getByRole("article", { name: "Answer" });
  await expect(answer).toContainText("Searched for “Find papers on sentence embeddings, slowly” and read 2 of 3 papers");
  await expect(page).toHaveURL(/\/chat\/[0-9a-f-]{36}$/);

  await answer.getByRole("button", { name: "References (3)" }).click();
  const references = page.getByRole("complementary", { name: "References" });
  const items = references.getByRole("listitem");
  await expect(items).toHaveCount(3);
  await expect(items.nth(0)).toContainText("ACL 2024Cited 1Read");
  await expect(items.nth(1)).toContainText("ICLR 2025Read");
  await expect(items.nth(2)).toContainText("Sentence embeddings from language model promptsACL 2025Ranked, not read");
  await noSeriousViolations(page);

  await references.getByLabel("Venue").selectOption("ACL");
  await expect(items).toHaveCount(2);
  await references.getByLabel("Year").selectOption("2025");
  await expect(items).toHaveCount(1);
  await expect(references).toContainText("Showing 1 of 3 papers");
  await references.getByLabel("Venue").selectOption("");
  await references.getByLabel("Year").selectOption("");
  await references.getByLabel("Cited only").check();
  await expect(items).toHaveCount(1);

  // The header button toggles the same list, and the References are stored with the answer.
  await page.getByRole("button", { name: "References (3)" }).first().click();
  await expect(references).toHaveCount(0);
  await page.reload();
  await page.getByRole("button", { name: "References (3)" }).first().click();
  await expect(page.getByRole("complementary", { name: "References" }).getByRole("listitem")).toHaveCount(3);
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
