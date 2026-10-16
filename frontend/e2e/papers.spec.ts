import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolations } from "./helpers";

test.beforeEach(async ({ request }) => {
  await request.post(`${MOCK}/__control/reset`);
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
