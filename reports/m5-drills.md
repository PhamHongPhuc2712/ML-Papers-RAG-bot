# Failure drills on the local stack (R3, 2026-10-14)

**Result: 7 of 8 drills pass; drill 6 could not be run as specified.** One drill (2) failed its
expectation on the first run. The chat page hid a database outage behind an empty history. It
was fixed with a browser regression test and passed on the re-run. The cold start is still
open. It is not a drill expectation, but it disturbed drills 1 and 3. It is measured below, and
the remedy is left to the developer.

## Setup

| | |
|---|---|
| Code | `8cd95df` + this task's fix |
| API | `uv run --env-file .env --project backend uvicorn --factory copilot.app:create_default_app --port 8000`, restarted per drill with the drill's variable in front (`restart_api.sh VAR=value`) |
| Web app | `npm run build && BACKEND_URL=http://127.0.0.1:8000 npm run start` (Next 16, Node v22.23.2) |
| Services | Compose `core` Postgres (`copilot_v2`) and Qdrant (release `m2-20260924T095724Z`) |
| Account | `drill-r3@example.org`, random password kept outside the repository; deleted at the end |
| Clients | A Python client through the web app's `/api` proxy (cookie session, like the browser) and Playwright's Chromium for the pages. Both record status codes, error codes, SSE frame names and page banners, never question text |
| GPU | 30 s before starting: median 19%, max 69% (gate ≤ 20%); no Linux compute process but ours |
| Spend | One answered chat turn (the baseline): 1 ledger row, $0.000592. Every other drill turn failed before reaching the model |

## Drills

| # | Do | Expected | Observed | Result |
|---|---|---|---|---|
| 1 | `docker compose stop qdrant`; search in the web app; send a chat message | Search: retryable banner. Chat: `error` frame, stored `failed`. `/health/ready` 503 `vector_unavailable` | `/health/ready` 503 `vector_unavailable`. Search API 503 `candidates_unavailable`; the page shows "search is unavailable" with Retry, no stack trace ([screenshot](m5-drills/d1-qdrant-down-search.png)). Chat frames `chat, status, error`, error `candidates_unavailable` (retryable), stored `failed / candidates_unavailable`. A query repeated from before the outage was served from the 10-minute ordering cache (200 in 0.03 s), as designed | **Pass**. The error code is the search service's typed `candidates_unavailable` (both branches failed), not `search_unavailable`/`vector_unavailable` |
| 2 | `docker compose stop postgres`; load `/chat` | Retryable error on the page, no stack trace; `/health/ready` 503 `database_unavailable` | **First run (04:31 UTC):** `/health/ready` 503 `database_unavailable`. The API answered `GET /v1/chats` with a typed 503, and search showed "database is unavailable" with Retry. But `/chat` rendered an **empty history and no message** ([screenshot](m5-drills/d2-postgres-down-chat-before.png)): `ChatSidebar` acted only on a 401, and `ChatView` only on 401 and 404. **Re-run after the fix (04:47 UTC):** "Your chats could not be loaded." with Retry ([screenshot](m5-drills/d2-postgres-down-chat-after.png)). After `docker compose start postgres`, `/health/ready`, `/v1/me`, `/v1/chats` and search recovered without restarting the API | **Fail → fixed → Pass** |
| 3 | Restart the API with `LLM_DAILY_SPEND_CAP_USD=0.0001` (today's ledger already held $0.000592); send a chat message | `llm_spend_cap` message naming the UTC reset; turn `failed`; no ledger row over the cap | The first turn after the restart failed earlier, at retrieval (`candidates_unavailable`, cold start). After two warming searches: frames `chat, status, evidence, status, error`, error `llm_spend_cap`, not retryable, message "Today's answer budget is used up. It resets at 00:00 UTC.", stored `failed / llm_spend_cap`. Ledger unchanged: 1 row, $0.000592 | **Pass** |
| 4 | Restart the API with `OPENAI_API_KEY=` (empty) | Startup logs `chat unavailable: llm_unconfigured`; `POST /v1/chats/stream` 503 `chat_unavailable`; search and history work | Startup log `chat unavailable: llm_unconfigured` (followed by its traceback); readiness 200. Chat 503 `chat_unavailable`; `/v1/chats` 200, `/v1/me` 200, search 200 | **Pass** |
| 5 | `timeouts_seconds.turn: 5` (stale after 10 s); start a chat turn; `kill -9` the API 3 s in; restart | The turn reads `interrupted` after the restart | The client got `RemoteProtocolError` (incomplete chunked read); the row stayed `pending`. After the restart it read `interrupted / turn_interrupted`. `configs/chat.yaml` was restored to `turn: 120` straight afterwards (`git diff` empty) | **Pass** |
| 6 | Run a GPU-heavy process from another project, or record that none was running | Search results carry the degraded banner rather than failing | Not run as specified. No other project's GPU job was running, and starting one is outside this project's policy (never start or stop other workloads). Under the Windows-side load seen earlier in the session (GPU 25–60%), searches degraded and did not fail: 2 returned 200 with `rerank_timeout` and `degraded: true`. While ten more searches ran, 17 one-second samples had a median of 12% (max 99%, our own reranker), and all ten were clean. The degraded banner itself is covered by the browser test `a degraded ordering is named, and no results says so` | **Not possible**; incidental evidence only |
| 7 | Restart with `ACCESS_TOKEN_TTL_SECONDS=300`; sign in; wait past 300 s; click a chat | Redirect to `/login?expired=1&next=…`; signing in returns to the chat | Signed in 04:54:25 UTC; clicked at 04:59:35. Redirected to `/login?next=%2Fchat%2F<id>&expired=1` with "Your session expired. Sign in again to continue." ([screenshot](m5-drills/d7-session-expired.png)); signing in returned to the same chat | **Pass**. Minor: the header on that page still shows "Account / Sign out", since the server-rendered header is not refreshed by the client-side redirect |
| 8 | `backup create`; `users delete` the drill account; `backup restore --replace` into a scratch database at head | The account and its chats are back in the scratch database; the live database is untouched | Live before: 2 users, 12 chats, 34 messages, 78 evidence rows (drill account 7 chats). Backup `20261014T045953Z` under `${DATA_DIR}/backups/`, same counts. Live after deletion: 1 / 5 / 20 / 67, no drill rows. Scratch `copilot_drill_r3` (migrated to `0009`), after the restore: 2 / 12 / 34 / 78, drill account and its 7 chats present. Live after the restore: unchanged at 1 / 5 / 20 / 67. Scratch dropped | **Pass** |

## The fix (drill 2)

- `frontend/components/ChatSidebar.tsx`: any failure other than a 401 now shows "Your chats could
  not be loaded." with Retry, never an empty history.
- `frontend/components/ChatView.tsx`: the same for one chat's messages ("This chat could not be
  loaded.").
- Regression test: `frontend/e2e/chat.spec.ts`, `a database outage is said, with a retry, never
  shown as an empty history`. The scripted backend gains `/__control/database?down=1`, which
  makes `GET /v1/chats*` return 503 `database_unavailable`. The test failed with both components
  reverted (the alert was not found), and passes with them. The whole chat spec passes, 9 of 9.

## Cold start (open, measured)

The first queries after the API or Qdrant restarts can miss the 1 s candidate deadline while
Qdrant's page cache warms. The search then fails with 503 `candidates_unavailable`.

| When | From cold | Failed |
|---|---|---|
| First API start after the host restart | 2 searches | 1 (`candidates_unavailable`); the next was degraded (`rerank_timeout`) |
| After the drill 3 restart | 1 chat turn | 1 (`candidates_unavailable`, before the LLM was reached) |
| After the final normal restart (ready in 37 s, GPU 0%) | 12 fresh searches | 1 (the 7th) |

W4 recorded the same gap: two of four real-stack walkthrough attempts failed on it
(`reports/m4-web.md`). Remedies, none taken here because none is a drill expectation:
- Retry the candidate stage once when its deadline is missed.
- Give chat, whose turns already take seconds, a longer candidate deadline.
- Warm the chunk collection at startup with more than the single warm-up query.

## Other observations

- **Request log.** R1's request log is printed under uvicorn, one JSON line per request. A
  streamed chat turn logs `request_id: null`, because the stream route never calls
  `request_id(request)`.
- **Startup sweep log.** The sweep's line `interrupted N chat turns left pending` is INFO on the
  `copilot.chat.api` logger, which has no handler under uvicorn, so it does not print. Only
  `copilot.requests` was given one in R1.
- **Pending turns after a crash.** With the shipped `turn: 120`, a turn orphaned by a crash stays
  `pending` for up to 240 s. Until then, a new message in that chat is refused with
  `chat_busy`.

## Cold start: the candidate retry (2026-10-14, after the drills)

The developer chose remedy 1. `SearchService.rank` now runs the timed-out branches once more when
**every** branch failed and at least one only timed out. The retry gets what is left of the
total, at most `deadlines_seconds.branch`. Errors other than a timeout, such as Qdrant down, are
not retried. One branch answering is still served degraded, without a retry. A successful retry
is not marked degraded, and the trace records it (`stages.<branch>.retried`). Tests:
`test_candidates_that_all_time_out_are_retried_once` and
`test_the_candidate_retry_happens_once_and_only_for_timeouts`
(`backend/tests/integration/test_search_service.py`). Full suite 763 passed; `eval smoke`
reproduces its frozen rankings unchanged.

Measured on the real stack, GPU median 0%:

| After | Probe | Result |
|---|---|---|
| Qdrant and API restarted | 12 fresh searches | 0 failed (1 of 12 before the retry; too few to call it a gain) |
| Qdrant and API restarted | 2 chat turns, first thing | **Both failed** `candidates_unavailable`, 13.5 s and 14.4 s |
| Qdrant restarted, the chat's search service timed in-process | 4 `rank()` calls | First 3 failed (`lexical:timeout,dense:timeout` after the retry). The 4th was served, with lexical rescued by the retry and dense still timing out (`dense_timeout`, `rerank_timeout`) |

**The retry does not fix a cold Qdrant.** In the in-process probe, capturing the release, a plain
Postgres read on the request thread outside every search deadline, took 3.6–20 s, and `rank()`
overran its 3 s total by seconds. The whole process is congested while Qdrant reads its
segments from disk, not only the search stages. Calls abandoned at their deadline keep running
on the lanes, and a retry adds to them. The retry helps once the backlog drains: the 4th probe
was served. The likely remedy is the third one: readiness that waits for the chunk collection
to be warm before it reports `ok`, rather than more retries.

## Cold start: the startup warm-up (2026-10-14)

The developer chose to try remedy 3. Startup (`SearchApi.start`) now finishes only after
`SearchService.warm_until_ready` has run `warmup.queries` from `configs/search.yaml` through the
full `hybrid_rerank` ranking with production deadlines, until `consecutive: 3` in a row are
served with no candidate branch failing or timing out (cap `max_seconds: 180`). Rules:
- **What does not count as cold:** a reranker timeout is GPU contention, not a cold index, and
  does not hold startup.
- **Qdrant down:** a failure that is not a timeout stops the warm-up at once, and readiness
  reports it.
- **No requests while warming:** startup has not finished, so uvicorn accepts no request, and
  the web app shows its existing "not reachable, try again" message.

The 24 queries are generic ML topics written for this, not corpus text. The
served app now also prints its own `copilot.*` INFO lines on stderr
(`send_app_logs_to_stderr`). This also fixes the earlier observation that the stale-turn sweep's
line was dropped. Tests:
- four in `test_search_service.py`: cold then warm, Qdrant down, gives up at the cap, and a
  rerank timeout accepted with an empty list skipped;
- `test_the_shipped_warm_up_is_configured`;
- `test_the_served_app_prints_its_own_info_lines_once_on_stderr`.

The full suite passed (769), and `eval smoke` is unchanged.

Measured on the real stack, GPU median 0%:

| Start | Startup until ready | Warm-up report | Then |
|---|---|---|---|
| Qdrant just restarted | **138 s** | 3 queries, 4.1 s, no failures | 2 chat turns **complete** (retrieve 20.3 s, then 2.5 s); 6 fresh searches 200, none degraded |
| Qdrant warm (API-only restart) | 38 s | 3 queries, 2.5 s | — |
| Qdrant stopped | 38 s | stopped after 1 query in 0.1 s (`candidates_unavailable`) | `/health/ready` 503 `vector_unavailable`, as in drill 1 |

The extra 100 s on a cold start went into the existing one-off `warm()` call. It has no
deadline, and it waited until Qdrant had loaded the collection back into memory. The cold
start came down to that: the API took requests before Qdrant had finished loading.
Before the warm-up, the same sequence failed both chat turns. That is one run each way, so it
shows the mechanism, not a rate.

Remaining limit: the warm-up runs only when the API starts. If Qdrant restarts while the API
keeps running, the first searches meet a cold Qdrant again; the candidate retry and
readiness's typed 503s are what is left for that case. A host restart starts both.
