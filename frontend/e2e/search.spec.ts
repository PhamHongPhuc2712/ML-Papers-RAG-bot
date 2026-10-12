import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolations } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
});

test("a search lists papers, pages, and keeps the query in the URL", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(/2 papers from 2 venue-years/)).toBeVisible();
  await page.getByLabel("Search papers").fill("contrastive sentence embeddings");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\?q=contrastive/);
  const first = page.getByRole("article").first();
  await expect(first.getByRole("link", { name: /Contrastive sentence embeddings/ })).toBeVisible();
  await expect(first).toContainText("ACL 2024");
  await expect(first).toContainText("Ada Lovelace, Alan Turing, Grace Hopper et al.");
  await page.getByRole("button", { name: "More results" }).click();
  await expect(page.getByRole("article")).toHaveCount(2);
  await noSeriousViolations(page);
});

test("a degraded ordering is named, and no results says so", async ({ page }) => {
  await page.goto("/?q=degraded");
  await expect(page.getByRole("status")).toContainText("may be incomplete");
  await page.goto("/?q=nothing+matches");
  await expect(page.getByText("No papers matched")).toBeVisible();
});

test("a paper page shows its abstract, source and related papers", async ({ page }) => {
  await page.goto("/papers/aaaaaaaa-0000-4000-8000-000000000001");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(/Contrastive sentence embeddings/);
  await expect(page.getByText("We train sentence encoders")).toBeVisible();
  await expect(page.getByRole("link", { name: "PDF" })).toHaveAttribute("href", "https://example.org/a.pdf");
  await expect(page.getByRole("link", { name: /Graph transformers at scale/ })).toBeVisible();
  await expect(page.getByRole("link", { name: "Ask about this paper" })).toHaveAttribute(
    "href",
    "/chat?about=aaaaaaaa-0000-4000-8000-000000000001",
  );
  await noSeriousViolations(page);
});

test("an unknown paper is a not-found page", async ({ page }) => {
  const response = await page.goto("/papers/aaaaaaaa-0000-4000-8000-00000000dead");
  expect(response?.status()).toBe(404);
});
