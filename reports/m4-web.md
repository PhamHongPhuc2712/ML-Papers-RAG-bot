# M4 web app: browser tests and a walkthrough on the real stack (W1–W4, 2026-10-12)

The Next.js app in `frontend/`: search (`/`), paper pages (`/papers/[id]`), sign-in
(`/login`) and the chat (`/chat`, `/chat/[id]`). The browser talks only to the Next server,
which keeps the access token in an `httpOnly` cookie and proxies `/api/v1/*` to the backend,
chat events included.

## Versions

Node 22.23.2; next 16.3.6, react 19.3.0, typescript 5.9.3 (not 6.0.3: openapi-typescript
7.13.0 needs TypeScript 5), vitest 5.0.2, @playwright/test 1.63.0 with Chromium headless shell
build 1243. The lockfile was written with npm 11.20.0 through `npx`, because Node 22's npm
10.9.8 crashes resolving vitest's peers; npm 10's `npm ci` installs it cleanly. Backend at
`2145193` plus W4's files; `configs/chat.yaml` as decided by C4 and C7 (`StructuredGenerator`,
`condense: false`).

## Scripted backend (no GPU, database or money)

`frontend/e2e/mock-backend.mjs` serves invented data on port 8799. On 2026-10-12, against
`next dev`: **16 of 16 browser tests pass** (auth 4, search 4, chat 8), including the axe
checks (no `serious` or `critical` violation on the pages they visit). There are 5 unit tests:
the SSE parser and citation numbering. Every Review Focus state has a test:

| Review Focus | Test |
|---|---|
| Session expires while a page is open | `chat.spec.ts` "an expired session sends you to sign in and back" |
| Answer budget used up | `chat.spec.ts` "the spend cap is explained without a retry button…" |
| Navigate away or double send | `chat.spec.ts` "one click is one request: send is disabled while a turn runs" |
| Degraded or empty search | `search.spec.ts` "a degraded ordering is named, and no results says so" |
| Long input | `chat.spec.ts` "the composer counts characters and refuses more than 2,000" |

The CI job `frontend` (`.github/workflows/checks.yml`) runs `npm ci`, regenerates
`lib/api-types.ts` and fails on a diff, then runs lint, typecheck, unit tests and the browser
tests. Its first run happens on the next push. This report does not claim it is green.

## The real stack (spends money; idle GPU)

```text
uv run --env-file .env --project backend uvicorn --factory copilot.app:create_default_app --port 8000   # behind the idle-GPU gate
cd frontend && npm run build && BACKEND_URL=http://127.0.0.1:8000 npm run start
REAL_STACK=1 REAL_EMAIL=chat-smoke@example.org REAL_PASSWORD=<…> npx playwright test real-stack.spec.ts
```

GPU: the idle gate passed before the API loaded its models. `nvidia-smi` read 2% and
4,288 MiB before the first attempt, and 2% and 5,087 MiB before the third. The only compute
entry was the Windows-side `[Not Found]` process the host always shows.

Four attempts, all recorded:

| # | UTC | Outcome | What the stored trace says |
|---|---|---|---|
| 1 | 06:39 | **Failed**: the first search returned 503, and the page showed "search is unavailable" with Retry | The cold-start gap (`reports/m3-chat.md`): a new query's first chunk lookup exceeded the candidate deadline |
| 2 | 06:40 | **Failed** at turn 2 | Turn 1 complete (retrieve 19.84 s cold, generate 3.02 s); turn 2 `failed` with `candidates_unavailable`, cold again for the follow-up's words |
| 3 | 06:43 | **Failed** on the script's last step | Turn 1 complete (retrieve 2.42 s); turn 2 `insufficient` (retrieve 1.85 s, generate 3.05 s), so there was no citation to click |
| 4 | 06:46 | **Passed** (6.1 s) | Turn 1 complete (retrieve 2.62 s); turn 2 `insufficient` (retrieve 1.56 s). Both answers were served from the LLM response cache (generate < 1 ms) |

**Why turn 2 was insufficient.** C7's rule turned condensing off, so "How large was its batch
size?" is searched as typed. The search never sees "SimCSE" and finds no evidence for it. The
app says so plainly ("Searched for: How large was its batch size?"; screenshot below). This is
the product doing what `configs/chat.yaml` says, and it is the cost of that decision on the
plan's own example follow-up. With condensing on, C6's live smoke answered the same follow-up
("SimCSE's batch size was 64"). The walkthrough script, written assuming condensing, clicked
the last answer's citation; after attempt 3 it clicks the first answer's citation, still
requiring both answers.

**Spend.** Purpose `chat`, 2026-10-12: 2 calls (attempts 2 and 3 generated uncached), $0.00112.

**Screenshots** (`reports/m4-web/`): `real-search.png`, the search page for "contrastive
learning of sentence embeddings"; `real-chat.png`, turn 1's answer with two cited sources open,
then turn 2's insufficient-evidence reply. They show our corpus's own abstracts, not ORB.

## What degraded, and what it means

- **Cold start** failed the first two attempts. This is the same open item as C6. A broader
  startup warm-up, or a longer candidate deadline for chat, remains R3's to decide.
- **Follow-ups without condensing** fail whenever the follow-up depends on the previous turn
  for its subject. C7 measured this on 43 conversations: `raw` 0.674 against `condensed` 0.791,
  with gold-paper retrieval 0.81 against 1.00. The rule required p < 0.05 and saw 0.227. This
  walkthrough is one more concrete instance. Whether to revisit the setting is the
  developer's decision, not this report's.
