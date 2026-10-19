import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolations, signIn } from "./helpers";

const chats = (page: import("@playwright/test").Page) =>
  page.getByRole("navigation", { name: "Chats" }).getByRole("listitem");

test.beforeEach(async ({ request, page }) => {
  await request.post(`${MOCK}/__control/reset`);
  await signIn(page);
});

test("a question streams its steps, then an answer whose citation opens its passage", async ({ page }) => {
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("How are sentence pairs formed?");
  await box.press("Enter");
  await expect(page.getByRole("status")).toContainText(/Searching the papers|Reading the top|Writing the answer/);
  await expect(page.getByRole("list", { name: "Conversation" }).getByText("How are sentence pairs formed?")).toBeVisible();
  const answer = page.getByRole("article", { name: "Answer" });
  await expect(answer).toContainText("Sentence pairs are formed");
  await expect(answer).toContainText("Searched for “How are sentence pairs formed?” and read 2 of 3 papers");
  await expect(page).toHaveURL(/\/chat\/[0-9a-f-]{36}$/);
  await answer.getByRole("button", { name: "Source 1" }).click();
  // The claim the citation supports is marked, and the panel shows its paper and passage.
  await expect(answer.locator("mark")).toHaveText("Sentence pairs are formed from a sentence and a noised copy of it.");
  await expect(answer.getByRole("button", { name: "Source 1" })).toHaveAttribute("aria-pressed", "true");
  const panel = page.getByRole("complementary", { name: "Paper" });
  await expect(panel.getByRole("heading")).toHaveText("Contrastive sentence embeddings with in-batch negatives");
  await expect(panel).toContainText("Method, page 3");
  await expect(panel).toContainText("Ada Lovelace, Alan Turing, Grace Hopper and 1 other. ACL 2024");
  await expect(panel.getByRole("link", { name: "Paper page" })).toHaveAttribute("href", "/papers/aaaaaaaa-0000-4000-8000-000000000001");
  await expect(panel.getByRole("link", { name: /PDF/ })).toHaveAttribute("href", "https://example.org/a.pdf");
  await expect(chats(page)).toHaveCount(1);
  await noSeriousViolations(page);
  // A paper the answer read but did not cite keeps its passages under "more passages read".
  await panel.getByRole("button", { name: "All references" }).click();
  await page.getByRole("complementary", { name: "References" }).getByRole("button", { name: "Graph transformers at scale" }).click();
  const other = page.getByRole("complementary", { name: "Paper" });
  await expect(other).toContainText("cited none of its passages");
  await other.getByText("1 more passage read from this paper").click();
  await expect(other).toContainText("Abstract, abstract only");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("complementary", { name: "Paper" })).toHaveCount(0);
});

test("Copy puts the answer and its numbered sources on the clipboard", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
  await page.keyboard.press("Enter");
  await page.getByRole("article", { name: "Answer" }).getByRole("button", { name: "Copy" }).click();
  await expect(page.getByRole("button", { name: "Copied" })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
    "Sentence pairs are formed from a sentence and a noised copy of it. [1]\n\n[1] Contrastive sentence embeddings with in-batch negatives, ACL 2024",
  );
});

test("an answer whose search ran out of time says it may have missed papers", async ({ page }) => {
  await page.getByLabel("Ask about ML papers").fill("A partial search, please");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toContainText("Part of the search ran out of time");
});

test("a follow-up shows what was searched, and history survives a reload", async ({ page }) => {
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("How are sentence pairs formed?");
  await box.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  await box.fill("And the negatives?");
  await box.press("Enter");
  await expect(page.getByText("Searched for “And the negatives? (contrastive sentence embeddings)” and read 2 of 3 papers")).toBeVisible();
  await page.reload();
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(2);
});

test("insufficient evidence is said plainly", async ({ page }) => {
  await page.getByLabel("Ask about ML papers").fill("How do proteins fold?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("The indexed papers don't contain enough evidence")).toBeVisible();
});

test("the spend cap is explained without a retry button, and the composer stays usable", async ({ page }) => {
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("Will the budget hold?");
  await box.press("Enter");
  // Next renders its own route announcer with role="alert"; the app's alerts are banners.
  const alert = page.getByRole("alert").and(page.locator(".banner"));
  await expect(alert).toContainText("resets at 00:00 UTC");
  await expect(alert.getByRole("button", { name: "Retry" })).toHaveCount(0);
  // The message stays in the conversation (the sidebar also lists it as the chat's title).
  await expect(page.getByRole("list", { name: "Conversation" }).getByText("Will the budget hold?")).toBeVisible();
  await expect(box).toBeEnabled();
});

test("one click is one request: send becomes Stop while a turn runs", async ({ page }) => {
  const box = page.getByLabel("Ask about ML papers");
  // "slow" makes the scripted backend pause longer, so the turn is still running below.
  await box.fill("Explain slowly how sentence pairs are formed");
  await page.getByRole("button", { name: "Send" }).click();
  await box.fill("A second question typed meanwhile");
  await expect(page.getByRole("button", { name: "Stop" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Send" })).toHaveCount(0);
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  await expect(page.getByRole("button", { name: "Send" })).toBeEnabled();
});

test("Stop ends a running turn and the box is ready again", async ({ page }) => {
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("Explain slowly how sentence pairs are formed");
  await box.press("Enter");
  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(0);
  await expect(box).toBeEnabled();
  // The stopped question can be asked again, as one question, not two.
  await page.getByRole("button", { name: "Ask again" }).click();
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  await expect(page.getByRole("list", { name: "Conversation" }).getByText("Explain slowly how sentence pairs are formed", { exact: true })).toHaveCount(1);
});

test("the composer counts characters and refuses more than 2,000", async ({ page }) => {
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("x".repeat(1800));
  await expect(page.getByText(/\/ 2000/)).toHaveCount(0);
  await box.fill("x".repeat(2001));
  await expect(page.getByText("2001 / 2000")).toBeVisible();
  await expect(page.getByRole("button", { name: "Send" })).toBeDisabled();
});

test("a chat can be deleted from the sidebar", async ({ page }) => {
  await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: /Delete chat/ }).click();
  await expect(chats(page)).toHaveCount(0);
  await expect(page).toHaveURL(/\/$/);
});

test("an expired session sends you to sign in and back", async ({ page, request }) => {
  await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  const here = new URL(page.url()).pathname;
  await request.post("http://127.0.0.1:8799/__control/expire");
  await page.reload();
  await expect(page).toHaveURL(/\/login\?next=.*&expired=1/);
  expect(new URL(page.url()).searchParams.get("next")).toBe(here);
  await expect(page.getByRole("status")).toContainText("Your session expired");
});

test("a database outage is said, with a retry, never shown as an empty history", async ({ page, request }) => {
  await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  const chatUrl = page.url();
  await request.post(`${MOCK}/__control/database?down=1`);
  await page.goto(chatUrl);
  await expect(page.getByRole("alert").filter({ hasText: "Your chats could not be loaded." })).toBeVisible();
  await expect(page.getByRole("alert").filter({ hasText: "This chat could not be loaded." })).toBeVisible();
  await request.post(`${MOCK}/__control/database?down=0`);
  await page.getByRole("alert").filter({ hasText: "Your chats could not be loaded." }).getByRole("button", { name: "Retry" }).click();
  await page.getByRole("alert").filter({ hasText: "This chat could not be loaded." }).getByRole("button", { name: "Retry" }).click();
  await expect(chats(page)).toHaveCount(1);
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  await expect(page.getByText("could not be loaded")).toHaveCount(0);
});
