# Sign-up by emailed code: the real-stack walk (sign-up plan, A6)

Date: 2026-10-22. Plan: `docs/superpowers/plans/2026-10-21-sign-up.md` (A1–A6, gate G8).

## What was built (A1–A5)

| Task | Commit | What |
|---|---|---|
| A1 | `2b90f37` | 6-digit codes (HMAC only, 10 min, once, dead on the 5th wrong try, a new code ends the old), the SMTP sender, Mailpit `v1.31.3` in Compose, `POST /v1/auth/code` |
| A2 | `2b8c84b` | One "Log in or sign up" page: email, then code; `POST /v1/auth/token {email, code}`; no passwords at sign-in |
| A3 | `3586d42` | 20 answers a day per account, checked under a lock on the account before anything is spent; `429 daily_limit`; "N answers left today" |
| A4 | `af42019` | Settings: usage, delete all chats and delete the account, each by a code from `POST /v1/me/code`; the count lives in `answer_usage` |
| A5 | `de50ed3` | Helpful / Not helpful under each finished answer, `PUT /v1/chats/{chat}/messages/{message}/feedback`, counted by `chat stats`, backed up |

## The walk

`e2e/real-stack.spec.ts` with `REAL_STACK=1 REAL_DELETE=1`, Playwright 1.63.0, Chromium, against
the API on `copilot_v2` (migrated to `0013_message_feedback` for this walk), `npm run start` and
Mailpit on this machine. A new throwaway email each attempt; no account existed beforehand.

Steps: sign up through Mailpit (the code read from its API) → ask "Find papers on contrastive
learning of sentence embeddings" → rate it Helpful → open References and a citation → a follow-up
("Which of them uses dropout as augmentation?") → the cited paper's page → History → Settings shows
"2 of 20 answers used today, 2 this month" → reopen the chat, the rating still Helpful → delete all
chats by a fresh code (History then empty, usage still 2) → after 61 s, delete the account by a
fresh code → signed out, Sign up shown.

| Attempt | Start (UTC) | Outcome |
|---|---|---|
| 1 | 03:25:39 | Set aside. First turn `candidates_unavailable` after 21 s: the first turns after the API restart, Qdrant cold. Nothing to rate |
| 2 | 03:31:44 | Set aside. First answer complete (retrieve 21.6 s) and rated (`PUT` 204); the follow-up `candidates_unavailable` |
| 3 | 03:35:06 | Both answers complete (retrieve 4.3 s, 4.6 s), rated. **Failed at delete all chats: a product bug, fixed below** |
| 4 | 03:41:16 | **Passed** (2.1 min) after search was warmed with 12 queries, all clean |

Attempts 1–3's accounts were removed with `users delete`; attempt 4's deleted itself. No
`walk-a6-*` account is left. Spend for the day, all four attempts: **$0.0025**, 6 `chat` calls
(`llm spend --day 2026-10-22`). `chat stats` for the day reads 0 turns: every walk account was
deleted, and its messages with it.

GPU: sampled each second through attempt 4, mean 22.6%, median 18%, 51 of 120 samples over 20%.
That includes this API's own embedding and reranking, which `nvidia-smi` cannot tell apart from
the Windows-side workload; before the walk, idle, it read 9.6% mean. The walk is a functional
record; its timings are not evidence.

### Found and fixed: a delete code refused right after sign-in

Attempt 3 reached Settings 24 s after signing in. Codes are rate-limited to one a minute per
email, counted from the newest code, which was the sign-in code, already used. `POST
/v1/me/code` answered 429 and the page said "A code was sent less than a minute ago. Use it",
pointing at a dead code. The minute is meant to stop resending while a code waits to be used, so
it now applies only while the newest code is unused (`identity/codes.py`); the 5-an-hour cap
still bounds the mail. Test first: `test_a_used_code_does_not_make_the_next_one_wait`, which
failed with 429 before the change.

### Seen, not fixed

- **Cold search after an API restart.** Attempts 1 and 2 lost turns to `candidates_unavailable`
  although startup warm-up passed (3 searches). The machine was short of memory: 17 of 24 GB
  used and 6.4 GB in swap, the API holding 6.0 GB and Qdrant 4.9 GB, with other projects' and
  editor processes beside them. Twenty searches later every search was clean in 1.4–2.8 s. This
  is the known cold-start item (`reports/m5-drills.md`), worse under memory pressure.
- **Attempt 4's first turn took 48 s and was degraded** ("Part of the search ran out of time"),
  while three `GET /v1/corpus/coverage` calls (the home page's paper count) took 33–44 s each
  beside it. Earlier the same call took 4.5 s. Recorded for the deployment plan.

## Screenshots (`reports/sign-up/`, light and dark)

`real-settings-*`: usage and both deletes; `real-rated-*`: the reopened chat, the first answer
still marked Helpful; `real-delete-chats-*`: the code form; `real-deleted-*`: signed out after
deleting the account. The walk's other screenshots (home, references, citation, follow-up,
paper, History) match `reports/ui-redesign/`'s and are not copied again.

## Checks

Backend full suite 804 passed (`.env.test`; a first run had 804 passed and 1 setup error in the offline `test_vectors.py::test_both_branches_share_one_filter_and_never_serve_deleted_points`, which passed alone, in the unit directory (499) and in a full re-run, under 8 GB of swap: not reproduced), offline set, Ruff, mypy strict; frontend lint,
typecheck, vitest 35, Playwright 50 with axe in both themes, `next build`. CI on the pushed
commit: see the progress board.
