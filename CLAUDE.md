# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

**M1 is complete and G1 passes.** P1.1–P1.5 are implemented under `backend/`: API and
isolated persistence harness, canonical paper identity with provenance, parser and
chunker, durable job queue with source adapters, and immutable corpus snapshots with a
coverage endpoint. `docs/superpowers/progress.md` is the live task and gate board — read
it before assuming anything about status. M2 is under way: P2.1 (metrics, labels, exact
BM25), P2.2 (indexes and release switching), P2.3 (fusion, bounded reranking,
deadlines), P2.4 (the search API) and P2.5 (retrieval ablations) are done, and release
`m2-20260924T095724Z` is **active** in Qdrant — 85,729 paper and 3,426,221 chunk points.
`copilot_v2` is migrated to `0004_search_orderings`. P2.5's pre-registered rule retained
BM25 as the first search configuration (`reports/retrieval/pilot/`); G2 still needs E3
and the locked test split, which has **not** been run. Next is E3.

The GPU is shared: with a Windows-side workload WSL cannot see (`nvidia-smi` lists it as
`[Not Found]`) and with other projects' jobs on this host (`esci-multimodel-ltr` ran GPU
smoke scripts during P2.4). Under that load `hybrid_rerank` degrades — explicitly, within
its deadline — and any latency measured then is contended. Sample `nvidia-smi` beside a
timing run; never stop another workload.

Beyond the plan tasks, the corpus pipeline gained a venue-by-venue runner (`corpus
mirror` / `corpus run`) that mirrors one venue-year, ingests it, parses with N workers,
verifies, then deletes that venue-year's PDFs so peak disk stays at one venue-year
rather than the ~450 GB the whole corpus needs at once.

**Current corpus state (2026-09-23):** the rebuild **finished 2026-09-22**. `copilot_v2`
is the live corpus — all 42 venue-years, **85,729 papers and 3,426,221 chunks**, every one
`paragraph-pack-v1`, parsed from source PDFs. Verified: max token count 1,200 with zero
chunks over the ceiling, ordinals contiguous from 0 in all 81,929 documents, body prose
ending mid-sentence 15.2% against the 14.4% predicted. Evidence in
`reports/m1-rebuild.md`.

**Two venue-years are incomplete and cannot be fixed here.** 3,802 papers have no text;
3,759 of those are zero-byte PDFs *at the source* — NeurIPS 2025 (38.3% of it) and ICLR
2026 (32.4%). The mirror published empty placeholders, the recorded OpenReview URLs return
403, and the accepted decision is to ship without them. Do not treat this as a parse bug
and do not re-run those venue-years expecting different output.

Databases on this host: `copilot_v2` (**live**), `copilot` (previous `fixed-window-v2`
corpus, kept as a fallback), `copilot_pilot_v2` and `copilot_restore_check` (validation
leftovers).

## This host

WSL2 Linux, 12 cores, 23 GB RAM, RTX 3080 Laptop with 16 GB VRAM (spec §14's "is there a
usable GPU" question — yes). `DATA_DIR=/home/phamphucintern/ml-copilot-data` on ext4,
outside the repository. `uv` **is** on PATH; the `py -3.12 -m uv` prefix in older evidence
reports describes the earlier Windows host and does not apply here.

Two gitignored env files at the repository root:

- `.env` — full development configuration, including `HF_TOKEN` for shard downloads.
- `.env.test` — only `DATA_DIR`, `TEST_DATABASE_URL`, `TEST_QDRANT_URL`,
  `TEST_QDRANT_COLLECTION_PREFIX`.

**Never pass `.env` to pytest.** `DATABASE_URL` in the process environment beats the
conftest fixture's keyword arguments (pydantic aliases outrank init keywords) and the
suite then targets the developer database.

## Sources of truth, in reading order

1. `docs/superpowers/specs/2026-09-05-ml-research-copilot-design.md` — architecture,
   schemas (§5), contracts (§6), ranking formulas (§7–8), evidence pipeline (§9), API
   (§10), evaluation gates (§11), operations (§12). The contract source of truth.
2. `docs/superpowers/plans/2026-09-05-0N-*.md` — six milestone plans (M1–M6). Each task
   section names the files it owns, its interfaces, a regression example, acceptance
   cases, the suites to run and the exact commit message.
3. `docs/superpowers/progress.md` — live status, gate board, recorded decisions and the
   session log. Update it when a task or decision changes.
4. `reports/` — measured evidence per task: foundation, identity, parser audit,
   ingestion, corpus/export, chunking comparison, parse failures.
5. `docs/data-card.md` — what the corpus is, where it comes from, what it excludes and
   what may be exported.

Implement only the task asked for. Do not pull later-task schemas or infrastructure
forward without a demonstrated need.

## Corpus facts worth knowing before touching the pipeline

- **Source is a Hugging Face mirror, not OpenReview.** `GenAI4ELab/papercli-papers`
  supplies the registry (`papers.parquet`, 11 columns, pinned at `90a1fbd3…`) and
  `GenAI4ELab/papercli-papers-<venue>` the PDFs. Read the **root registry**, never the
  five-column `browse/<venue>/<year>.parquet` views — those carry no authors, no track
  and no PDF path. OpenReview guest access returns `ChallengeRequiredError` (403) on both
  the notes and attachment routes, which is why the mirror exists.
- **Membership is observed, not asserted.** `classify_track` maps each record's own venue
  label onto a canonical track and a decision. Main is matched **positively**, so an
  unrecognized label is `other` and excluded. Across 2023–2026 that admits 85,732
  accepted main-conference papers and excludes Findings, workshops, industry and demo
  tracks, shared tasks, tutorials, co-located conferences and unaccepted submissions.
  `data/fixtures/tracks/registry-tracks.jsonl` freezes all 262 labels for the test that
  asserts the full mapping.
- **Chunk windows are counted in the pinned BGE-M3 tokenizer**, cached under
  `${DATA_DIR}/models/tokenizers/` and verified by sha256. The worker refuses to start
  without it and never falls back to counting words — counting words put 51.4% of chunks
  over the documented cap.
- **PDFs are deleted after their venue-year is stored**, except for papers that failed to
  parse, whose copy is the only thing a retry could read. Re-parsing therefore means
  re-downloading; settle parser and chunker choices before a bulk run.

## Hard constraints (spec §2, duplicated at the top of every plan)

- **Zero infrastructure budget, permanently.** PostgreSQL and Qdrant are self-hosted in
  Docker containers on this machine. No Supabase, Qdrant Cloud or managed storage unless
  a future explicitly-recorded decision changes it.
- **The one paid resource** is the hosted LLM/multimodal generation API, funded
  personally by the developer. Per-request cost logging and the operator-configured daily
  spend cap (P5.2) are required, not optional.
- **All persistent local data lives under `DATA_DIR`** — read from environment, never
  hardcoded, startup fails in every environment when unset. Deleting that one directory
  deletes the entire dataset; keep that guarantee true.
- **Auth is self-hosted** — Argon2id credentials in local Postgres, application-issued
  JWTs, `POST /v1/auth/token`. Supabase Auth was explicitly superseded (spec §10).
- **Portfolio scope, not production.** The MVP is M1–M4 plus the free-tier subset of M5.
  P5.3's staged deployment and RPO/RTO drill and P5.2's public rate limiting are
  *deferred, not deleted*.
- Python 3.12, Node.js 22, TypeScript strict, UTC timestamps, locked dependencies, pinned
  images and model revisions.
- User identity comes only from a verified token — never a client-supplied `user_id`.
  Public corpus data and private uploads never share a Qdrant collection or storage
  namespace. **Hugging Face is an offline artifact destination** and is never queried
  during a user request. Three offline callers contact it — `corpus mirror` for the
  corpus, `evaluation/litsearch.py` for the benchmark and `search fetch-model` for the
  embedding and reranker weights — and all fetch at a pinned revision. Nothing in the
  serving path may join them.
- Never report tests, benchmarks, corpus coverage or user observations that did not
  actually run. Planned examples are not measured results.

## Commands

Backend package is `copilot` under `backend/src`, managed by uv with `backend/.venv` as
the only environment. Frontend (M3) does not exist yet.

```bash
uv sync --project backend --frozen --group dev
uv sync --project backend --frozen --group dev --group models   # + PyTorch, for real models
uv run --project backend ruff check backend
uv run --project backend mypy --config-file backend/pyproject.toml backend/src
uv run --env-file .env.test --project backend pytest backend/tests -q          # full suite
uv run --project backend pytest backend/tests -m "not integration" -q          # offline CI set
docker compose --profile test up -d postgres-test qdrant-test                  # for integration tests
docker compose --profile core up -d postgres qdrant                            # dev services
```

`mypy` **must** be given `--config-file`: run from the repository root it finds no
configuration and silently drops `strict`, which is how four real errors survived until
2026-09-16.

Corpus pipeline, all reading `DATABASE_URL`, `DATA_DIR` and `HF_TOKEN` from `.env`:

```bash
uv run --env-file .env --project backend python -m copilot.cli corpus mirror --venue ICLR --year 2024
uv run --env-file .env --project backend python -m copilot.cli corpus mirror-index --venue ICLR --year 2024
uv run --env-file .env --project backend python -m copilot.cli corpus run            # the whole venue plan
uv run --env-file .env --project backend python -m copilot.cli corpus run --only JMLR:2025
uv run --env-file .env --project backend python -m copilot.cli corpus rechunk        # re-chunk without PDFs
uv run --env-file .env --project backend python -m copilot.cli corpus export --run SNAP --out SNAP
uv run --env-file .env --project backend python -m copilot.cli corpus validate --manifest SNAP/manifest.json
uv run --env-file .env --project backend python -m copilot.cli corpus restore --manifest SNAP/manifest.json --database-url ...
uv run --env-file .env --project backend python -m copilot.cli corpus coverage
uv run --env-file .env --project backend python -m copilot.cli corpus compare-chunkers --source <pdf dir>
uv run --env-file .env --project backend python -m copilot.cli worker run --worker-id w1
```

Retrieval indexes (P2.2), needing the `models` group and the core Qdrant service:

```bash
uv run --env-file .env --project backend python -m copilot.cli search fetch-model
uv run --env-file .env --project backend python -m copilot.cli search build-index --manifest SNAP/manifest.json --collections papers
uv run --env-file .env --project backend python -m copilot.cli search validate-index --release SNAP
uv run --env-file .env --project backend python -m copilot.cli corpus activate --release SNAP
uv run --env-file .env --project backend python -m copilot.cli search compare-precision --manifest SNAP/manifest.json
```

Search service (P2.3): BM25 and dense candidates for one captured release, RRF in our
code, cross-encoder rerank of at most 50, every stage under a deadline with explicit,
typed fallbacks. Ranking parameters live in `configs/search.yaml`; the reranker is pinned
in `configs/models.yaml` beside the embedder.

```bash
uv run --env-file .env --project backend python -m copilot.cli search pilot --query "contrastive learning for sentence embeddings"
uv run --env-file .env --project backend python -m copilot.cli search pilot --split development --traces NAME
uv run --env-file .env --project backend python -m copilot.cli search compare-precision --model reranker --release SNAP
```

Search API (P2.4): `POST /v1/search`, `GET /v1/papers/{id}`, `GET /v1/papers/{id}/related`.
Orderings are cached in PostgreSQL (`search_orderings`, ten minutes) and paged by
HMAC-signed cursors; a degraded ordering is never reused for a new search. The API serves
only a release whose collections sit in its own `QDRANT_COLLECTION_PREFIX`. Startup loads
both models and warms every stage once. `frontend/openapi.json` is the published contract
— re-export it after any API change, or `test_search_api.py` fails.

Retrieval experiments (P2.5): `configs/experiments/retrieval.yaml` lists the variants and
the decision rule, fixed before results are seen. Development tunes, validation chooses,
and test runs only with `--locked-test` — once, for a release decision.
`reports/retrieval/README.md` explains the outputs. BGE-small lives in its own release,
`m2-20260924T095724Z-bge-small`, paper collection only and never activated.

```bash
uv run --env-file .env --project backend python -m copilot.cli eval retrieval --split validation --out reports/retrieval/pilot
uv run --project backend python -m copilot.cli eval report --out reports/retrieval/pilot
uv run --project backend python -m copilot.cli eval compare --baseline A/validation/metrics.json --candidate B/validation/metrics.json
uv run --project backend python -m copilot.cli eval smoke        # offline, frozen outputs; --update to re-freeze
```

```bash
uv run --env-file .env --project backend alembic -c backend/alembic.ini upgrade head
uv run --env-file .env --project backend uvicorn --factory copilot.app:create_default_app --port 8000
uv run --project backend python -m copilot.cli api export-schema --out frontend/openapi.json
```

`.env`'s `DATABASE_URL` names `copilot_v2`, the live corpus. Building never activates;
activation needs an explicit release id and re-validates the whole collection pair.
PyTorch is in the opt-in `models` group so CI never installs or downloads a model —
unit and integration tests use the deterministic two-dimensional `FixtureEmbedding`.

`corpus run` is resumable: progress lives in `${DATA_DIR}/runs/<plan>-state.json`, keyed
to the plan rather than the invocation, so a restart continues at the next venue-year.
`--state` points it elsewhere and `--fresh` starts the plan over.

Compose profiles: `core` (Postgres, Qdrant, API, worker, frontend), `models`
(embedding/reranker), `test` (isolated services under `${DATA_DIR}/test/`). Integration
tests need the explicit test services; destructive fixtures refuse any database or Qdrant
prefix not starting with `test_` and any data root outside the test subdirectory. Missing
services are a setup failure, never a silent skip.

## Architecture in brief (spec §3–§9)

One modular Python application plus one durable worker process, sharing typed domain
contracts in `backend/src/copilot/contracts.py` — Pydantic models with `extra="forbid"`,
and Protocols for model adapters (`EmbeddingModel.encode`, `Reranker.score`,
`Generator.answer`) so providers never leak their own types into the domain. GPU work
never runs in the ASGI event loop.

- **PostgreSQL** owns canonical paper identity (generated UUID work IDs, an alias table,
  a redirect table for merges), all application state, the job queue (transactional
  leases via `SKIP LOCKED`, idempotency keys, heartbeats) and the singleton
  `active_release` pointer.
- **Qdrant** is a rebuildable serving index. Each corpus release gets immutable
  `paper_abstracts_<release>` / `paper_chunks_<release>` collections with named dense
  (BGE-M3, 1024-d, cosine) and exact-BM25 sparse vectors; the pointer switches
  transactionally only after count, dimension, checksum and canary validation. Private
  uploads live in a separate collection, dense-only, filtered by server-derived `user_id`.
- **Search** (§7): 100 BM25 + 100 dense candidates → RRF `sum(1/(60+rank))` in
  application code → cross-encoder rerank of at most 50 → 20 results. Degrades explicitly.
- **Recommendations** (§8): content-based profile from decayed positive feedback plus
  topic/seed priors; bounded heuristic score; MMR diversification; reasons come from real
  score components, never from an LLM.
- **Evidence and chat** (§9): research mode retrieves papers then chunks within them;
  claims carry evidence IDs validated structurally before any answer text streams.
- **Corpus pipeline** (§4): versioned per-venue-year manifests; identity resolution with
  contradictory strong-ID collisions held as conflicts; parse into heading-aware sections
  with page spans; chunk under a versioned policy; Parquet exports carrying
  redistribution eligibility, with private data never exported.

### Chunking policies

Two exist, selected by `chunker.policy` in `configs/parsing.yaml`. The version name
states the policy, so switching changes every chunk ID.

- **`paragraph-pack-v1`** (current): split a section into paragraphs, pack whole ones to
  800 tokens, close the chunk when the next would pass 900, split a paragraph over 1,200
  on sentence boundaries, and repeat the previous chunk's last 2 sentences as overlap
  (bounded to a quarter of the budget). Over 40 real papers it ends 14.4% of prose chunks
  mid-sentence against the fixed window's 47.2%, with 16% fewer chunks.
- **`fixed-window-v2`**: 450-token windows, 600 cap, 60 overlap, never crossing a section.

Paragraphs must be **inferred**: pypdf's extracted text contains no blank lines, so a
paragraph's last line is detected as one that both ends a sentence and falls short of the
column width. `reports/m1-chunking-comparison.md` has the measured comparison;
`corpus compare-chunkers` reproduces it.

## How a task is executed (plan convention)

Write the task's regression test first and confirm it fails for the stated reason;
implement one behavior at a time; add the listed acceptance cases; run the listed suites,
then Ruff, mypy and every relevant integration check; record actual commands, versions and
outcomes in `reports/` and the progress tracker; inspect `git diff --check` and the full
diff; stage only the files the task owns; commit with the message the plan specifies.
