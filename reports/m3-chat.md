# M3 chat: the streaming endpoint on the real stack (C6, 2026-10-11)

`POST /v1/chats/stream` runs one chat turn: condense → retrieve → generate, as a LangGraph
graph (C5) over the live corpus. The validated answer is stored before it is sent, then
streamed as server-sent events: `chat`, `status`, `evidence`, then `final` or `error`. This is a
smoke run on the developer's stack, not a benchmark. The questions are about SimCSE, and none
of the text comes from ORB.

## Setup

- Commit `d21035b` plus the C6 working tree. langgraph 1.2.12, langchain-core 1.6.5,
  langchain-openai 1.6.6, openai 3.28.0, fastapi 0.116.1, torch 2.14.0 (CUDA 13.0).
- `copilot_v2` migrated from `0006_search_ordering_mode` to `0009_llm_call_purpose_chat`
  (users, chats and the `chat` ledger purpose). The active release is `m2-20260924T095724Z`.
- `configs/chat.yaml`: `openai-gpt-6-luna`, `generator: structured` (C4's decision),
  `condense: true`. The first stage reranks on title and chunk with a 2.5 s rerank deadline.
- GPU: the idle gate (30 one-second samples, median ≤ 20%, no Linux compute process) passed
  before the API loaded its models. `nvidia-smi` read 1% and 4,514 MiB before the second
  attempt.

```text
uv run --env-file .env --project backend alembic -c backend/alembic.ini upgrade head
uv run --env-file .env --project backend python -m copilot.cli users create --email chat-smoke@example.org < <password file>
uv run --env-file .env --project backend uvicorn --factory copilot.app:create_default_app --port 8000
POST /v1/auth/token, then POST /v1/chats/stream twice in one chat:
  "What loss does SimCSE use to train sentence embeddings?"
  "How large was its batch size?"
```

## Outcome

**First attempt, right after startup: both turns failed with `candidates_unavailable`.** It
failed honestly. The stream sent `chat`, `status {searching}`, then `error` (retryable) after
7.6 s and 6.6 s, and both assistant messages were stored `failed` with that code. The cause is
the open cold-start item: the startup warm-up warms one query, and a new query's first BM25
lookup in the 3.4M-point chunk collection took 7.2 s when reproduced outside the API against
0.01 s once cached. That is past the candidate deadline, so both branches timed out. The
search page's own `POST /v1/search` succeeded at the same moment only because its query was
the warm-up query.

**Second attempt, a new chat, same questions: both turns completed.**

| Turn | Frames, in order (seconds since request) | Stored trace (s) | Answer |
|---|---|---|---|
| 1 | `chat` 0.02, `status searching` 0.02, `evidence` (9 items, no warnings) 8.84, `status writing` 8.84, `final` 11.52 | condense 0.00, retrieve 8.81, generate 2.66 | 1 claim, citing E5 and E7 |
| 2 | `chat` 0.02, `status condensing` 0.02, `status searching` with query "How large was SimCSE’s batch size?" 0.90, `evidence` (9 items) 3.52, `status writing` 3.52, `final` 5.60 | condense 0.87, retrieve 2.62, generate 2.06 | 1 claim, citing E1 and E2 |

The follow-up was condensed into a standalone query that names SimCSE, as expected. Both
answers passed citation validation, with no warnings.

**Ledger.** Three calls with purpose `chat` on `openai-gpt-6-luna` (served `gpt-6-luna`), all
succeeded: one condense and two generations, $0.001229 in total, mean latency 1.8 s. The
failed turns made no model call: they failed in search, before generation.

## Findings

- The turn works end to end on the live stack and costs about $0.0004–0.0008 per turn.
- Cold start is the chat's main failure. Until the chunk collection's pages are cached, a new
  query can lose both candidate branches, and the turn fails as retryable. The search API has
  the same exposure (recorded 2026-10-06). A broader startup warm-up, or a longer
  candidate deadline for chat, would address it. Neither is measured here; both are left to
  the release plan (R3's failure drills).
- Retrieval dominates turn time (2.6–8.8 s); generation took about 2–2.7 s.
