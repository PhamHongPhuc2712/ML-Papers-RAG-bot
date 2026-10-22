// Run by hand against the real local stack (REAL_STACK=1). It signs up (or in) as REAL_EMAIL with the
// code the API emails to Mailpit (MAILPIT_URL, default http://127.0.0.1:8025), asks for papers on a topic and a follow-up, opens a citation, the cited paper's page,
// History and Settings, and saves screenshots in light and dark (UI redesign, D4). With REAL_DELETE=1 it
// also rates the answer, checks usage, deletes all chats and then the account, each by emailed code
// (sign-up plan, A6): use a throwaway email. Two turns: about $0.01 on the hosted model. Needs an idle GPU.
import { expect, test, type Page } from "@playwright/test";

const MAILPIT = (process.env.MAILPIT_URL ?? "http://127.0.0.1:8025").replace(/\/+$/, "");

/** The newest sign-in code Mailpit received for an email, waiting up to 20 s for it to arrive. */
async function codeFor(email: string, after: number): Promise<string> {
  for (let tries = 0; tries < 40; tries += 1) {
    const reply = await fetch(`${MAILPIT}/api/v1/search?query=${encodeURIComponent(`to:${email}`)}`);
    const found = (await reply.json()) as { messages: { ID: string; Created: string }[] };
    const fresh = found.messages.find((m) => Date.parse(m.Created) >= after);
    if (fresh) {
      const message = (await (await fetch(`${MAILPIT}/api/v1/message/${fresh.ID}`)).json()) as { Text: string };
      const code = message.Text.match(/\b(\d{6})\b/)?.[1];
      if (code) return code;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`no sign-in code for ${email} in Mailpit`);
}

const shot = async (page: Page, name: string) => {
  for (const scheme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme: scheme });
    await page.screenshot({ path: `test-results/real-${name}-${scheme}.png` });
  }
  await page.emulateMedia({ colorScheme: "light" });
};

test("an answer, its panel, a follow-up, the paper, History and Settings on the real stack", async ({ page }) => {
  test.setTimeout(420_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/login?next=%2F");
  const email = process.env.REAL_EMAIL ?? "";
  const asked = Date.now() - 1000;
  await page.getByLabel("Email").fill(email);
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByLabel("Code").fill(await codeFor(email, asked));
  await page.getByRole("button", { name: "Log in" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  await shot(page, "home");

  const box = page.getByLabel("Ask about ML papers");
  const answers = page.getByRole("article", { name: "Answer" });
  await box.fill("Find papers on contrastive learning of sentence embeddings");
  await box.press("Enter");
  await expect(answers).toHaveCount(1, { timeout: 90_000 });
  const helpful = answers.first().getByRole("button", { name: "Helpful", exact: true });
  await helpful.click();
  await expect(helpful).toHaveAttribute("aria-pressed", "true");
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
  await expect(page.getByText(/2 of 20 answers used today, 2 this month\./)).toBeVisible();
  await shot(page, "settings");
  if (!process.env.REAL_DELETE) return;

  // The rating was kept: reopen the chat from the sidebar.
  await page.getByRole("navigation", { name: "Chats" }).getByRole("link").first().click();
  await expect(page.getByRole("article", { name: "Answer" }).first().getByRole("button", { name: "Helpful", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await shot(page, "rated");

  // Each delete asks for a fresh code; codes are 60 s apart per email.
  await page.goto("/account");
  let sent = Date.now() - 1000;
  await page.getByRole("button", { name: "Delete all chats" }).click();
  await page.getByLabel("Code").fill(await codeFor(email, sent));
  await shot(page, "delete-chats");
  await page.getByRole("button", { name: "Delete all chats" }).click();
  await expect(page.getByRole("status").filter({ hasText: "All of your chats are deleted." })).toBeVisible();
  await page.goto("/history");
  await expect(page.getByText("Your chats appear here after your first question.")).toBeVisible();
  await page.goto("/account");
  await expect(page.getByText(/2 of 20 answers used today/)).toBeVisible();

  await page.waitForTimeout(61_000);
  sent = Date.now() - 1000;
  await page.getByRole("button", { name: "Delete account" }).click();
  await page.getByLabel("Code").fill(await codeFor(email, sent));
  await page.getByRole("button", { name: "Delete my account" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  await expect(page.getByRole("link", { name: "Sign up" })).toBeVisible();
  await shot(page, "deleted");
});
