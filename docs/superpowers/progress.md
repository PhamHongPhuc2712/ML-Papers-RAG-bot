# ML Research Copilot — Progress Tracker

Last updated: 2026-09-14.

## Current state

- Implementation: **4 / 30 tasks complete** (P1.1–P1.4); P1.3's parser audit is measured and closed, P1.4's live pilot is still open.
- MVP implementation: **4 / 25 tasks complete**.
- Passing implementation gates: **0 / 6** (G1's parser audit now passes at 95% usable text; the 100-paper replay and snapshot restore remain).
- Corpus indexed: none yet. Benchmarks observed: none yet. User-study observations: none yet.
- Test evidence on this host (2026-09-14, WSL/Linux, Docker 29.4.3): full suite **120 passed**; Ruff and mypy strict clean. Corpus on disk: 2,260 ICLR 2024 PDFs (13.70 GB) plus metadata, under `${DATA_DIR}/sources/papercli/`.

Status vocabulary: **Not started**, **In progress**, **In review**, **Done**, **Blocked**. No placeholder is evidence; a task is Done only when every named acceptance case ran, its report exists and its commit is recorded.

## What has been implemented

| Area | Summary |
|---|---|
| Configuration | `Settings` validates the environment; `DATA_DIR` is required in every environment and is the single root for all persistent local data (default `C:\ml-copilot-data`). Production refuses mock mode, default secrets and unrestricted CORS. |
| Persistence | PostgreSQL via SQLAlchemy 2 + Alembic. Migrations `0000_foundation` (corpus releases, active-release pointer), `0001_corpus` (papers, identifiers, versions, authors, venues, source records, field provenance, conflicts, quarantine, redirects), `0002_chunks`, `0003_jobs` (jobs, source checkpoints). |
| API | FastAPI composition root with `/health/live` and a bounded `/health/ready` that checks the database, the active corpus release and its Qdrant collections, returning typed 503s without credentials. |
| Local services | `compose.yaml` bind-mounts Postgres and Qdrant under `${DATA_DIR}` (test profile under `${DATA_DIR}/test`) instead of named volumes; `infra/Dockerfile.backend`; CI workflow running Ruff, mypy and the offline test set. |
| Paper identity (P1.2) | DOI/arXiv/title normalization; transactional identity resolution with strong-ID matching, compatible-metadata title matching, review conflicts, quarantine of invalid identifiers, per-field provenance, raw-artifact retention by checksum, manual merge with redirects. |
| Parsing and chunking (P1.3) | pypdf text adapter behind a `PageAdapter` boundary with typed failures (missing, not PDF, oversized, encrypted, corrupt, empty text); heading-aware sections with page spans; versioned fixed-window chunker (450/600/60) that never crosses sections, keeps tables coherent or fragments them with labels, and excludes references from default evidence; UUIDv5 chunk identity; idempotent persistence. |
| Ingestion (P1.4) | PostgreSQL-leased durable jobs (`SKIP LOCKED`, lease tokens, heartbeats, capped exponential backoff honouring Retry-After) and a worker that commits handler output with completion; adapters for OpenReview (with login token), proceedings JSON listings, arXiv Atom and Semantic Scholar enrichment; hardened PDF download; `corpus ingest` deterministic sampling and `worker run`. |
| Test harness | Isolated fixtures that refuse any database or Qdrant prefix not starting with `test_` and any data root outside `${DATA_DIR}/test`; injected provider transports and DNS; synthetic captured provider responses; a repository-owned CC0 fixture PDF. |

## Files implemented

```text
CLAUDE.md, compose.yaml, .env.example, .gitattributes, .gitignore
.github/workflows/checks.yml
infra/Dockerfile.backend
configs/parsing.yaml, configs/corpus.yaml
data/fixtures/metadata.jsonl
data/fixtures/papers/{fixture.pdf, fixture.txt, LICENSE.txt, build_fixture.py}
backend/{pyproject.toml, uv.lock, alembic.ini, README.md}
backend/migrations/env.py
backend/migrations/versions/{0000_foundation, 0001_corpus, 0002_chunks, 0003_jobs}.py
backend/src/copilot/{__init__, app, cli, config, contracts}.py
backend/src/copilot/db/{__init__, models, session}.py
backend/src/copilot/corpus/{__init__, normalize, dedupe, parse, chunk, documents, download, ingest}.py
backend/src/copilot/corpus/sources/{__init__, base, openreview, proceedings, arxiv, semantic_scholar}.py
backend/src/copilot/jobs/{__init__, queue, worker}.py
backend/tests/conftest.py
backend/tests/unit/{test_config, test_contracts, test_identity, test_chunks, test_membership}.py
backend/tests/integration/{test_health, test_migrations, test_deduplication, test_parser, test_ingestion}.py
backend/tests/fixtures/providers/openreview_notes_offset_{0,2}.json
```

## Task board

| Task | Deliverable | Status | Depends on | Evidence / commit |
|---|---|---|---|---|
| P1.1 | Create a runnable API and isolated persistence test harness | Done | Design review | `c9ec8f4`, `f652b9a`; [foundation evidence](../../reports/m1-foundation.md) (local) |
| P1.2 | Normalize publication identity and preserve version provenance | Done | P1.1 | `35c0da1`; [identity evidence](../../reports/m1-identity.md) (local) |
| P1.3 | Parse documents into traceable sections and chunks | Done; 20-paper audit measured 2026-09-14 at 95% usable text, parser retained, and all five measured defects repaired in `pypdf-text-v2` (prose chunk median 182 → 401 tokens) | P1.2 | `9bbef4a`, `4f8a3b7`, `c541429`; [parser evidence](../../reports/m1-parser-audit.md) |
| P1.4 | Ingest an accepted-paper pilot through resumable jobs | In progress — code and synthetic-fixture suite green; live 100-paper pilot blocked on OpenReview credentials | P1.2, P1.3 | `d3fad84`; `reports/m1-ingestion.md` not yet produced |
| P1.5 | Export immutable corpus snapshots and publish coverage | Not started | P1.4 | Not produced |
| P2.1–P2.5 | Retrieval and ranking | Not started | P1.5 | Not produced |
| P3.1–P3.5 | Recommendation workspace | Not started | P2.4 | Not produced |
| P4.1–P4.5 | Evidence assistant | Not started | P2.3, P3.1 | Not produced |
| P5.1–P5.5 | Portfolio hardening (P5.3 deployment half and public rate limiting deferred) | Not started | P1–P4 | Not produced |
| P6.1–P6.5 | Research expansion (optional) | Not started | G5 | Not produced |

## Gate board

| Gate | Required evidence | State |
|---|---|---|
| G1 | 100-paper replay, identity checks, 20-paper parser audit, snapshot restore | Partly met — identity checks pass; parser audit passes (19/20 usable, ≥90% rule met, 2026-09-14); replay and snapshot restore outstanding |
| G2–G6 | See specification §11 | Not started |

## Decisions recorded during implementation

| Decision | Choice | Why | Where |
|---|---|---|---|
| Restart baseline | P1.1/P1.2 restored from `04f6e3e` and adapted, not rewritten | Reviewed code; only the 2026-09-09 design revision needed applying | commits `c9ec8f4`, `35c0da1` |
| Services | Docker Desktop 4.90 + WSL 2.7 installed 2026-09-09; both profiles bind-mount under `DATA_DIR` | Spec §2/§3 | `compose.yaml` |
| Qdrant on a Windows bind mount | Accepted for now; Qdrant warns "Unrecognized filesystem – cannot guarantee data safety" | Durability caveat for the core profile; decide before P2.2 builds the first serving index | `reports/m1-foundation.md` |
| First parser adapter | pypdf, not Docling | Docling needs PyTorch and model downloads, breaking offline deterministic CI on a zero-budget machine; swappable behind `PageAdapter` | `configs/parsing.yaml`, `reports/m1-parser-audit.md` |
| Proceedings adapter | Captured JSON listings, not per-venue HTML scraping | Scraping belongs to P6.1's verified venue adapters | `sources/proceedings.py` |
| OpenReview access | Login token from a free account; PDFs via `api2.openreview.net/attachment` | Guest API access returns `ChallengeRequiredError` (403) on notes and PDF routes | `sources/openreview.py`, `configs/corpus.yaml` |
| Version identity | `_upsert_version` treats (source, revision, version) as one observation | Lets a download replace the placeholder metadata checksum in place without duplicate rows | `corpus/dedupe.py` |
| Extracted-text repair (2026-09-14) | Repair NUL characters, UTF-16 surrogate halves, ligatures, line-break hyphenation and page furniture at the `PageAdapter` boundary; `PARSER_VERSION` raised to `pypdf-text-v2`, `chunker_version` untouched | PostgreSQL and UTF-8 reject NUL and unpaired surrogates outright (32% and 2% of sampled papers). The rest are quality defects the audit measured. Only the sections feeding the chunker changed, not the windowing algorithm, and nothing is indexed, so no re-parse is owed | `corpus/parse.py`, [parser evidence](../../reports/m1-parser-audit.md) |
| Source PDF retention (2026-09-14) | Corpus PDFs may be deleted once their chunks are stored, so bulk ingestion can stream venue by venue instead of holding every PDF at once | No planned task reads a corpus PDF after parsing: the page-image path in §9 and P4.3/P4.4 covers *private uploads* (`uploads/images.py`), not corpus papers. Re-parsing under a new `parser_version` would need them again, so settle the parser choice first | spec §9, P4.3/P4.4 |

## Session log

| Date | Tasks changed | Checks and results | Evidence paths / commit | Next action |
|---|---|---|---|---|
| 2026-09-09 | Repository reset; design docs revised for zero-budget local infrastructure; docs untracked | Markdown checks; no application tests | `a61a07c`, `9069dde`, `60953e6` | Implement P1.1 |
| 2026-09-09 | Docker Desktop + WSL installed; P1.1, P1.2 restored and adapted | 67 passed (22 offline, 45 integration), Ruff, mypy | `c9ec8f4`, `35c0da1`, `f652b9a` | P1.3 |
| 2026-09-10 | P1.3 implemented; P1.4 implemented; dotenv CORS fix | 119 passed, Ruff, mypy; live OpenReview listing returned 403 challenge | `9bbef4a`, `c02249e`, `d3fad84` | Add OpenReview credentials to `.env`, run and replay the 100-paper pilot, write `reports/m1-ingestion.md`, then P1.5 |
| 2026-09-14 | Environment rebuilt on WSL; extracted-text sanitization fixed; 2,260 ICLR 2024 PDFs collected; P1.3 audit measured | 120 passed, Ruff, mypy; 3,645 real chunks accepted by PostgreSQL; audit 19/20 usable text | `96130ef`, `4f8a3b7`; [parser evidence](../../reports/m1-parser-audit.md) | Fix the four structure defects (headings, caption-sections, header leakage, over-segmentation), then run venue ingestion and P1.5 |

## Definition of a completed task

All named acceptance cases have evidence; required tests actually ran; code was reviewed; the report contains actual versions and measurements; the task commit is recorded. Engineering completion and human-observation checkpoints are recorded separately. Do not publish synthetic or planned results as measured outcomes.
