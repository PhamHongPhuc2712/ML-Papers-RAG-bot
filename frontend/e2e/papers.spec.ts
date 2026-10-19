import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolationsInBothThemes, signIn } from "./helpers";

const PAPER = "/papers/aaaaaaaa-0000-4000-8000-000000000001";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("a paper page shows its venue, authors, links, abstract and related papers", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto(PAPER);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(/Contrastive sentence embeddings/);
  await expect(page.getByRole("main")).toContainText("ACL 2024");
  await expect(page.getByRole("main")).toContainText("Ada Lovelace, Alan Turing, Grace Hopper, Edsger Dijkstra");
  await expect(page.getByText("We train sentence encoders")).toBeVisible();
  await expect(page.getByText("Full text indexed: answers can quote any section")).toBeVisible();
  await expect(page.getByRole("link", { name: "PDF" })).toHaveAttribute("href", "https://example.org/a.pdf");
  await expect(page.getByRole("link", { name: "Graph transformers at scale" })).toBeVisible();
  await page.getByRole("button", { name: "Copy citation" }).click();
  await expect(page.getByRole("button", { name: "Copied" })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
    "Ada Lovelace, Alan Turing, Grace Hopper and Edsger Dijkstra. Contrastive sentence embeddings with in-batch negatives. ACL 2024. https://example.org/a.pdf",
  );
  await noSeriousViolationsInBothThemes(page);
});

test("a paper with only its abstract says so", async ({ page }) => {
  await page.goto("/papers/aaaaaaaa-0000-4000-8000-000000000002");
  await expect(page.getByText("Only the abstract of this paper is indexed")).toBeVisible();
});

test("asking about a paper, signed in, asks at once with the paper named", async ({ page }) => {
  await signIn(page, PAPER);
  await expect(page.getByText("Asking about: Contrastive sentence embeddings with in-batch negatives (ACL 2024)")).toBeVisible();
  const box = page.getByLabel("Ask about this paper");
  await box.fill("What are the negatives?");
  await box.press("Enter");
  await page.waitForURL(/\/chat\/[0-9a-f-]{36}$/);
  await expect(page.getByRole("list", { name: "Conversation" }).locator(".bubble")).toHaveText(
    "What are the negatives?\n\nAbout the paper “Contrastive sentence embeddings with in-batch negatives (ACL 2024)”.",
  );
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  // Asked once: Back returns to the paper, not to a page that would ask again.
  await page.goBack();
  await expect(page).toHaveURL(new RegExp(`${PAPER}$`));
});

test("asking about a paper, signed out, keeps the question across log in", async ({ page }) => {
  await page.goto(PAPER);
  await page.getByLabel("Ask about this paper").fill("What are the negatives?");
  await page.getByRole("button", { name: "Send" }).click();
  await page.waitForURL(/\/login\?next=/);
  await expect(page.getByText("Your question is kept")).toBeVisible();
});

test("an unknown paper is a not-found page", async ({ page }) => {
  const response = await page.goto("/papers/aaaaaaaa-0000-4000-8000-00000000dead");
  expect(response?.status()).toBe(404);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("This page doesn’t exist");
  await expect(page.getByRole("link", { name: "Go to home" })).toHaveAttribute("href", "/");
  await noSeriousViolationsInBothThemes(page);
});
