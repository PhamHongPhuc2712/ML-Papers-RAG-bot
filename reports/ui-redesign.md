# UI redesign: D4's pages, states and checks (2026-10-19)

Plan: [`docs/superpowers/plans/2026-10-16-ui-redesign.md`](../docs/superpowers/plans/2026-10-16-ui-redesign.md),
task D4. The design is the sketch canvas (boards 8–11). No backend or API change.

## What was built

- **History** (`/history`, board 9): every chat (up to the API's 100 per request), grouped Today /
  Previous 7 days / Older, each with its age ("1 hour ago", "3 days ago", or a date after a week)
  and a delete action; a title search in the browser that keeps titles holding every word. The
  board's answer preview and question count are **not built**: `GET /v1/chats` returns ids,
  titles and times only (listed under "Later plans"). The sidebar gains a History link.
- **Paper page** (board 8): venue and year, every author, PDF and source links, Copy citation, and an
  "Ask about this paper" box showing the paper it is about. The search has no way to attach a paper,
  so the question names it in its text ("About the paper “Title (ACL 2024)”."), as the D2 venue
  filter does. Signed in, the question is asked at once on a new chat and the handover URL
  (`/?q=…&send=1`) is replaced, so Back or a reload does not ask it again. Signed out, it waits
  across log in like the home page's. The earlier `/chat?about=` link, whose draft "In this paper, "
  never named the paper, is gone.
- **Settings** (`/account`, board 10's existing parts): the account email with Log out, the theme
  switch, and Delete account, which asks for the password before anything is deleted. Usage, a
  password change and "delete all chats" wait for the sign-up plan. The account menu's link reads
  Settings.
- **States** (board 11): page not found; insufficient evidence in the board's words, saying the
  papers found are still in References; a failed answer with Try again as its main action and, for
  `candidates_unavailable`, "the search timed out" instead of the API's generic message; the daily
  budget as a notice rather than an error; "Your session ended. Log in again and you return to the
  same chat."; the proxy's unreachable message in the board's words; "no chats yet" and an outage on
  History.

## Two defects found and fixed

1. **An ended session left the signed-in sidebar on the log-in page.** The client-side redirect
   kept the root layout from the old session (chats, account menu). Every expiry redirect now
   refreshes the server render (`sessionEnded` in `lib/api.ts`). Found on a screenshot; the
   expired-session test now checks the shell is signed out, and fails without the fix.
2. **A log-in submitted before the page's script ran put the password in the URL.** The form had
   no `method`, so a click before hydration was a native GET to
   `/login?email=…&password=…`, into the address bar, the history and any server log. One full-suite
   run caught it (`shell.spec.ts`, signing in). The form now posts to `/api/session`, which accepts
   a form body, sets the cookie and redirects with 303 (to `next`, or back to log in with
   `error=credentials|unreachable`). Two tests with JavaScript disabled cover it. The same-site
   check on `next` is now one function (`lib/next-path.ts`) used by the page and the route, and
   also refuses `/\host`, which browsers read as `//host`.

## Checks

| Check | Result |
|---|---|
| `npm run lint`, `npm run typecheck` | clean |
| `npm test` (vitest) | 29 passed (21 before D4) |
| `npm run e2e` (Playwright, scripted backend) | **40 passed** (29 before D4): History (4), paper page (5), Settings (4), the no-script log-in (2), the timed-out search with Try again, the insufficient-evidence wording, the ended session; axe in light and dark with no serious or critical violation on History, the paper page, not found, Settings, the delete form, the log-in page, an insufficient answer and a failed turn |
| `npm run build` | clean, 10 routes |
| Screenshots at 1440 px, light and dark, against the scripted backend | History, the paper page signed in and out, Settings with the delete form open, not found, an insufficient answer, the spend-cap notice, the ended session; compared by eye with boards 8–11, differences only where the API lacks the data (above) |

## The real-stack walk

`frontend/e2e/real-stack.spec.ts`, rewritten for the redesign: log in, "Find papers on contrastive
learning of sentence embeddings", its References panel, citation 1, the follow-up "Which of them
uses dropout as augmentation?", the cited paper's page, History and Settings, each screenshotted in
light and dark. `playwright.config.ts` takes `REAL_BASE_URL`; the walk ran against `next start` on
port 3200 (this commit's build) and the running API (release `m2-20260924T095724Z`), each attempt
with a throwaway account made by `users create` and deleted after by `users delete`.

| Attempt | UTC | Outcome | Stored turns |
|---|---|---|---|
| 1 | 08:44 | Failed, **set aside** | the first turn failed `candidates_unavailable` (both search branches past their deadlines) before any model call, $0; GPU 29–40% busy with 4 GB held by a Windows-side process WSL cannot see |
| — | 08:47–09:17 | No run | the GPU never stayed under 5% for 60 s in 30 minutes; the developer then asked for the walk to run anyway |
| 2 | 09:34 | Passed (31.8 s), **set aside** | follow-up degraded (`dense_timeout`, its answer showing the partial-search note); first turn retrieve 12.4 s; $0.00061 |
| 3 | 09:35 | **Passed** (8.7 s) | no warnings; retrieve 1.69 s and 2.12 s; the follow-up condensed to "Which of the papers on contrastive learning of sentence embeddings uses dropout as augmentation?" and answered with one claim citing two passages; $0.00054 |

**The GPU was not idle** during attempts 2 and 3: the same Windows-side workload ran beside them
(attempt 3: 5 samples at 2 s, median 17%, peak 99%). Nothing here is a timing measurement; the
walk checks that the pages work against the real stack. The first turn cost $0 in both passing
attempts because the hosted model's answer to that exact question was already in the response
cache (U4's walk asked it), and the follow-up's condensing hit the cache in attempt 3. Spent this
task: **$0.0012**.

Screenshots of attempt 3, light and dark, under [`ui-redesign/`](ui-redesign/): `real-home`,
`real-references` (the answer with its 20 References), `real-citation` (citation 1 opened: the
claim and passage marked), `real-follow-up`, `real-paper`, `real-history`, `real-settings`. They
show our corpus's own titles and passages.

What they show, beyond the boards being met:
- **The thread does not scroll to a new answer.** After the follow-up, its answer's actions sit
  under the ask box at 1440×900 until the reader scrolls. Not fixed here; a small follow-up.
- **Citation 1 is the paper's front matter** (the proceedings header with the author list), cited
  for "WhitenedCSE … is a 2023 paper by …". The passage supports the claim, but a paper-finding
  answer that cites front matter shows the reader nothing of the method: a retrieval and
  generation observation, recorded and not acted on.
- In attempt 2's light citation screenshot the panel's author line was not yet loaded (it comes
  from `GET /v1/papers/{id}` after the panel opens); attempt 3's has it.

## G7

Boards 1 and 4–11 built in light and dark and matching the canvas (D1–D4); every behaviour of
today's app kept and tested (Playwright 40); no serious or critical axe violation in either theme;
one recorded real-stack walk (attempt 3). **Remaining: CI green on a pushed commit.**
