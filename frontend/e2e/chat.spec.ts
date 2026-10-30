import { expect, test } from "@playwright/test";
import { MOCK, noSeriousViolations, noSeriousViolationsInBothThemes, signIn } from "./helpers";

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
  await expect(
    page.getByText("Searched for “And the negatives? (contrastive sentence embeddings)” and “contrastive learning negatives”, then read 2 of 3 papers"),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(2);
});

/** In the viewport and not under the sticky header or ask box: what is drawn at its centre is it. */
async function uncovered(locator: import("@playwright/test").Locator): Promise<boolean> {
  return locator.evaluate((element) => {
    const box = element.getBoundingClientRect();
    const x = box.left + box.width / 2;
    const y = box.top + box.height / 2;
    if (y < 0 || y > window.innerHeight) return false;
    const hit = document.elementFromPoint(x, y);
    return hit !== null && element.contains(hit);
  });
}

test.describe("on a short screen", () => {
  test.use({ viewport: { width: 1280, height: 600 } });

  test("a new question scrolls to the top, so its answer is read from the start", async ({ page }) => {
    const box = page.getByLabel("Ask about ML papers");
    const answers = page.getByRole("article", { name: "Answer" });
    await box.fill("How are sentence pairs formed?");
    await box.press("Enter");
    await expect(answers).toHaveCount(1);
    await box.fill("And the negatives?");
    await box.press("Enter");
    await expect(answers).toHaveCount(2);
    await expect.poll(() => uncovered(page.locator(".question .bubble").last())).toBe(true);
    await expect.poll(() => uncovered(answers.last().getByRole("button", { name: "Copy" }))).toBe(true);
    // A chat opened later starts at its latest question too.
    await page.reload();
    await expect(answers).toHaveCount(2);
    await expect.poll(() => uncovered(answers.last().getByRole("button", { name: "Copy" }))).toBe(true);
  });
});

test("insufficient evidence is said plainly, and its References stay", async ({ page }) => {
  await page.getByLabel("Ask about ML papers").fill("How do proteins fold?");
  await page.getByRole("button", { name: "Send" }).click();
  const answer = page.getByRole("article", { name: "Answer" });
  await expect(answer).toContainText("The papers this search found don’t say enough to answer that.");
  await expect(answer).toContainText("The 3 papers it found are still in References.");
  await expect(answer.getByRole("button", { name: "References (3)" })).toBeVisible();
  await noSeriousViolationsInBothThemes(page);
});

test("the spend cap is explained without a retry button, and the composer stays usable", async ({ page }) => {
  const box = page.getByLabel("Ask about ML papers");
  await box.fill("Will the budget hold?");
  await box.press("Enter");
  // A spent budget is a notice, not a fault, and offers no retry.
  const notice = page.getByRole("status").and(page.locator(".banner"));
  await expect(notice).toContainText("resets at 00:00 UTC");
  await expect(notice.getByRole("button")).toHaveCount(0);
  // The message stays in the conversation (the sidebar also lists it as the chat's title).
  await expect(page.getByRole("list", { name: "Conversation" }).getByText("Will the budget hold?")).toBeVisible();
  await expect(box).toBeEnabled();
});

test("a search that timed out says so, and Try again asks the question again", async ({ page }) => {
  await page.getByLabel("Ask about ML papers").fill("A timeout, then an answer");
  await page.keyboard.press("Enter");
  const alert = page.getByRole("alert").and(page.locator(".banner"));
  await expect(alert).toContainText("The answer could not be finished because the search timed out. Your question is kept.");
  await noSeriousViolationsInBothThemes(page);
  await alert.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByRole("article", { name: "Answer" })).toHaveCount(1);
  await expect(page.locator(".banner.error")).toHaveCount(0);
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
  await expect(page.getByRole("status")).toContainText("Your session ended. Log in again and you return to the same chat.");
  // The shell is signed out too: no chats or account menu left from the ended session.
  await expect(page.getByRole("link", { name: "Log in", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Account menu/ })).toHaveCount(0);
  await noSeriousViolationsInBothThemes(page);
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

test("Helpful / Not helpful rates an answer, changes it, clears it and comes back after a reload", async ({ page }) => {
  await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
  await page.keyboard.press("Enter");
  const answer = page.getByRole("article", { name: "Answer" });
  const helpful = answer.getByRole("button", { name: "Helpful", exact: true });
  const notHelpful = answer.getByRole("button", { name: "Not helpful" });
  // The button shows the choice at once; wait for the server to keep it before the next step,
  // or a reload (or the next click) can overtake the save.
  const rate = async (button: typeof helpful) => {
    const saved = page.waitForResponse((r) => r.url().endsWith("/feedback") && r.request().method() === "PUT");
    await button.click();
    expect((await saved).status()).toBe(204);
  };
  await expect(helpful).toHaveAttribute("aria-pressed", "false");
  await rate(helpful);
  await expect(helpful).toHaveAttribute("aria-pressed", "true");
  await page.reload();
  await expect(page.getByRole("article", { name: "Answer" }).getByRole("button", { name: "Helpful", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await rate(notHelpful);
  await expect(notHelpful).toHaveAttribute("aria-pressed", "true");
  await expect(helpful).toHaveAttribute("aria-pressed", "false");
  // Choosing the same rating again clears it.
  await rate(notHelpful);
  await expect(notHelpful).toHaveAttribute("aria-pressed", "false");
  await page.reload();
  await expect(page.getByRole("article", { name: "Answer" }).getByRole("button", { name: "Not helpful" }))
    .toHaveAttribute("aria-pressed", "false");
  await noSeriousViolationsInBothThemes(page);
});

test("a rating that cannot be saved goes back and says so", async ({ page, request }) => {
  await page.getByLabel("Ask about ML papers").fill("How are sentence pairs formed?");
  await page.keyboard.press("Enter");
  const answer = page.getByRole("article", { name: "Answer" });
  await expect(answer).toContainText("Sentence pairs are formed");
  await request.post(`${MOCK}/__control/database?down=1`);
  await answer.getByRole("button", { name: "Helpful", exact: true }).click();
  await expect(answer.getByRole("alert")).toHaveText("Your rating could not be saved. Try again in a moment.");
  await expect(answer.getByRole("button", { name: "Helpful", exact: true })).toHaveAttribute("aria-pressed", "false");
});
