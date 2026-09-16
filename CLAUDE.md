# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

P1.1 (API + isolated persistence harness) and P1.2 (paper identity + provenance) are implemented under `backend/`. They were restored from the pre-reset implementation at `04f6e3e` and adapted to the 2026-09-09 design revision (`DATA_DIR`, bind-mount Compose, test-root guard, hardening constraints folded into `0001_corpus`). Unit tests, Ruff and mypy pass; check the latest commit messages for whether the integration suite has run on this host yet.

P1.3 (pypdf parser behind a `PageAdapter`, versioned chunker, `chunks` table) and P1.4 (PostgreSQL-leased job queue, source adapters, hardened download, `corpus ingest` / `worker run`) are implemented and green against synthetic fixtures. The next task is P1.5 in `docs/superpowers/plans/2026-09-05-01-corpus-foundation.md`.

**OpenReview gates guest API access** with a browser challenge (HTTP 403 `ChallengeRequiredError`) on both the notes and PDF routes. The adapter logs in with a free account and sends a bearer token; the live pilot needs `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` in `.env`. PDFs come from `api2.openreview.net/attachment?name=pdf&id=…`, not `openreview.net/pdf`.

## Sources of truth, in reading order

1. `docs/superpowers/specs/2026-09-05-ml-research-copilot-design.md` — architecture, schemas (§5), contracts (§6), ranking formulas (§7–8), evidence pipeline (§9), API (§10), evaluation gates (§11), operations (§12). This is the contract source of truth.
2. `docs/superpowers/plans/2026-09-05-0N-*.md` — six milestone plans (M1–M6). Each task section (P1.1 … P6.5) names the files it owns, its interfaces, a regression example, implementation logic, acceptance cases, the suites to run and the exact commit message.
3. `docs/superpowers/README.md` — roadmap, working rhythm, test strategy.

Implement only the task asked for. Do not pull later-task schemas or infrastructure forward without a demonstrated need.

## Git gotchas

- `docs/` and `reports/` are gitignored **and untracked** — the owner keeps the design documents local-only. They were removed from the index on 2026-09-10; the last tracked copy is at `dd9b4eb` (`git show dd9b4eb:docs/superpowers/specs/2026-09-05-ml-research-copilot-design.md`). A fresh clone has no `docs/` until they are copied in. Task reports under `reports/` likewise stay local unless force-added.
- There is currently no progress tracker (`docs/superpowers/progress.md` was deleted). Plan steps that say "update the progress tracker" have no target until one is recreated.
- `.gitattributes` normalizes to LF, but three plan files are CRLF on disk. Byte-level edit scripts must tolerate both.

## Hard constraints (spec §2 Global Constraints, duplicated at the top of every plan)

- **Zero infrastructure budget, permanently.** PostgreSQL and Qdrant run self-hosted in Docker Desktop Linux containers on this Windows machine. No Supabase, Qdrant Cloud, or managed storage unless a future explicitly-recorded decision changes it. Docker Desktop 4.90 and the WSL 2.7 runtime were installed on this host on 2026-09-09 via winget; confirm `docker info` answers before running the integration suite.
- **The one paid resource** is the hosted LLM/multimodal generation API, funded personally by the developer. Its per-request cost logging and operator-configured daily spend cap (P5.2) are required, not optional.
- **All persistent local data lives under `DATA_DIR`** (documented default `C:\ml-copilot-data\`; read from environment, never hardcoded; startup fails in every environment if unset). Postgres and Qdrant bind-mount subdirectories of it; uploads, exports, backups, the model-weight cache and the `test` profile's data all live under it. Deleting that directory deletes the entire dataset — keep that guarantee true.
- **Auth is self-hosted** — Argon2id credentials in local Postgres, application-issued JWTs, `POST /v1/auth/token`. Supabase Auth was explicitly superseded (spec §10 records why).
- **Portfolio scope, not production.** The MVP is M1–M4 plus the free-tier subset of M5. P5.3's staged deployment and RPO/RTO drill and P5.2's public multi-tenant rate limiting are *deferred, not deleted*; account deletion and local backup/restore remain required.
- PowerShell commands from the repository root must work; no Bash-only scripts in the critical development path. Python 3.12, Node.js 22, TypeScript strict, UTC timestamps, locked dependencies, pinned images and model revisions.
- User identity comes only from a verified token — never a client-supplied `user_id`. Public corpus data and private uploads never share a Qdrant collection or storage namespace. Hugging Face is an offline artifact destination and is never queried during a user request.
- Never report tests, benchmarks, corpus coverage or user observations that did not actually run. Planned examples are not measured results.

## Commands

Backend package is `copilot` under `backend/src`, managed by uv 0.12.10 with `backend/.venv` as the only environment; frontend (M3, not built yet) will be Next.js under `frontend/`. `uv` is not on PATH on this machine — `py -3.12 -m uv` is the equivalent prefix for every command below. Frontend commands do not run yet.

```powershell
uv sync --project backend --frozen --group dev
uv run --project backend ruff check backend
uv run --project backend mypy --config-file backend/pyproject.toml backend/src
uv run --project backend pytest backend/tests -m "not integration" -q   # offline CI set
uv run --project backend pytest backend/tests -q                        # full, needs services
uv run --project backend pytest backend/tests/unit/test_foo.py -q       # one file
uv run --project backend pytest backend/tests -k test_name -q           # one test
uv run --project backend alembic -c backend/alembic.ini upgrade head
docker compose --profile test up -d                                     # isolated Postgres + Qdrant
docker compose --profile core up -d postgres qdrant                     # dev services for the pilot
uv run --project backend python -m copilot.cli corpus ingest --manifest configs/corpus.yaml --limit 100
uv run --project backend python -m copilot.cli worker run --worker-id pilot-1   # until nothing is due
npm --prefix frontend run typecheck   # also: lint, build, test:e2e
```

The CLI reads `DATABASE_URL` and `DATA_DIR` from `.env`, but `--staging-dir` defaults from the `DATA_DIR` *environment variable*, so export it (or pass the flag) in a shell that has not loaded `.env`. Provider transports, DNS resolution and the clock are injected; `backend/tests/fixtures/providers/` holds captured synthetic responses.

Compose profiles: `core` (Postgres, Qdrant, API, worker, frontend), `models` (embedding/reranker), `test` (isolated services under `${DATA_DIR}/test/`). Integration tests need the explicit test services; a destructive fixture must refuse any database or Qdrant collection prefix not starting with `test_` and any data root outside the test subdirectory. Missing services are a setup failure, never a silent skip. Normal CI is deterministic and offline; real-model evaluations are explicit, versioned, budgeted runs.

## Architecture in brief (spec §3–§9)

One modular Python application plus one durable worker process, sharing typed domain contracts in `backend/src/copilot/contracts.py` — Pydantic models with `extra="forbid"`, and Protocols for model adapters (`EmbeddingModel.encode`, `Reranker.score`, `Generator.answer`) so providers never leak their own types into the domain. No network boundaries between search, recommendation and RAG initially. GPU work never runs in the ASGI event loop.

- **PostgreSQL** owns canonical paper identity (generated UUID work IDs, an alias table for DOI/arXiv/etc., a redirect table for merges), all application and user state, the job queue (transactional leases via `SKIP LOCKED`, idempotency keys, heartbeats — no Redis/Celery unless measured load justifies it), and the singleton `active_release` pointer.
- **Qdrant** is a rebuildable serving index. Every corpus release gets immutable `paper_abstracts_<release>` / `paper_chunks_<release>` collections with named dense (BGE-M3, 1024-d, cosine) and exact-BM25 sparse vectors; a request reads the active-release pointer once and uses it throughout; the pointer switches transactionally only after count/dimension/checksum/canary validation. Private uploads live in a separate `user_documents_<embedding_revision>` collection, dense-only, always filtered by server-derived `user_id`.
- **Search** (§7): 100 BM25 + 100 dense candidates → RRF `sum(1/(60+rank))` implemented in application code → cross-encoder rerank of at most 50 → 20 results. Degrades explicitly: reranker timeout returns RRF order with a warning; one branch down returns the other with a warning; both down is a retryable 503.
- **Recommendations** (§8): content-based profile from decayed positive feedback plus topic/seed priors; bounded heuristic score; MMR diversification; reasons come from real score components, never from an LLM.
- **Evidence and chat** (§9): research mode retrieves papers, then chunks within them; generation returns structured claims with evidence IDs that are structurally validated against the selected evidence before any answer text streams; SSE over POST with `status/evidence/delta/final/error`; idempotent `request_id`. Research mode never silently searches private documents.
- **Corpus pipeline** (§4): versioned per-venue-year source manifests; DOI/arXiv normalization with contradictory strong-ID collisions held as conflicts; parse → section-aware chunks (450 tokens, cap 600, 60 overlap within a section); Parquet exports carrying redistribution eligibility, with private data never exported.

## How a task is executed (plan convention)

Write the task's regression test first and confirm it fails for the stated reason; implement one behavior at a time using the reference logic; add the listed acceptance cases; run the listed suites, then Ruff, mypy and every relevant integration or migration check; record actual commands, versions and outcomes; inspect `git diff --check` and the full diff; stage only the files the task owns; commit with the message the plan specifies.
