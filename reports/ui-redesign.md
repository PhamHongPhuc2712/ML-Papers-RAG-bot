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

## The real-stack walk: not yet run

`frontend/e2e/real-stack.spec.ts` was rewritten for the redesign: log in, a paper-finding request,
its References panel, a citation, a follow-up, the cited paper's page, History and Settings, each
screenshotted in light and dark (two turns, about $0.01). `playwright.config.ts` takes
`REAL_BASE_URL` so it can run against `next start` on any port.

| Attempt | UTC | Outcome |
|---|---|---|
| 1 | 08:44 | **Set aside.** The first turn failed with `candidates_unavailable` (both search branches past their deadlines) before any model call ($0). The GPU was 29–40% busy with 4 GB held by a Windows-side process WSL cannot see. |

The GPU then stayed busy for 30 minutes (sampled every 3 s, never 60 s under 5%), so by the
idle-GPU policy the walk waits. The throwaway account was deleted (`users delete`). Still to do: the
walk on an idle GPU, its screenshots under `reports/ui-redesign/`, and then G7.
