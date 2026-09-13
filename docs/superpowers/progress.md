# ML Research Copilot — Progress Tracker

Last updated: 2026-09-12.

## Current state

- Implementation: **5 / 30 tasks complete** (P1.1–P1.5); M1 is code-complete.
- MVP implementation: **5 / 25 tasks complete**.
- Passing implementation gates: **0 / 6** (G1: replay done and clean; the 20-paper parser audit is the last item outstanding).
- Corpus ingested: **100 accepted ICLR 2024 main-conference papers**, 4,470 chunks, from 2,260 eligible notes (dev database, not indexed — Qdrant serving indexes arrive in P2.2). Benchmarks observed: none yet. User-study observations: none yet.
- Test evidence on this host (2026-09-12): full suite **144 passed** against `postgres:17.11-bookworm` and `qdrant/qdrant:v1.19.1` in Docker Desktop; Ruff and mypy strict clean.

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
| Export and releases (P1.5) | Parquet metadata/fulltext/edges shards with per-shard and manifest checksums; redistribution filter withholding rows without established rights; private-field and schema-version refusal; restore into an empty database; `GET /v1/corpus/coverage` with null denominators; staged releases that cannot activate until P2.2 index validation. |
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
| P1.3 | Parse documents into traceable sections and chunks | Done; agent audit ran on real PDFs and drove `pypdf-text-v3`. Human confirmation of the 20-paper bench still outstanding | P1.2 | `9bbef4a`, `38a32c4`; [parser audit](../../reports/m1-parser-audit.md) |
| P1.4 | Ingest an accepted-paper pilot through resumable jobs | Done — live 100-paper pilot and clean replay on 2026-09-12 | P1.2, P1.3 | `d3fad84`; [ingestion evidence](../../reports/m1-ingestion.md) |
| P1.5 | Export immutable corpus snapshots and publish coverage | Done — 3.0 MB snapshot exported, validated and restored into a clean database with identical IDs | P1.4 | [corpus evidence](../../reports/m1-corpus.md), [data card](../data-card.md) |
| P2.1–P2.5 | Retrieval and ranking | Not started | P1.5 | Not produced |
| P3.1–P3.5 | Recommendation workspace | Not started | P2.4 | Not produced |
| P4.1–P4.5 | Evidence assistant | Not started | P2.3, P3.1 | Not produced |
| P5.1–P5.5 | Portfolio hardening (P5.3 deployment half and public rate limiting deferred) | Not started | P1–P4 | Not produced |
| P6.1–P6.6 | Research expansion (optional), including P6.6 comparing knowledge/graph and agentic answering against the single-shot baseline | Not started | G5 | Not produced |

## Gate board

| Gate | Required evidence | State |
|---|---|---|
| G1 | 100-paper replay, identity checks, 20-paper parser audit, snapshot restore | In progress — replay clean (100 canonical IDs byte-identical, 0 duplicate jobs); every parse classified; snapshot restore verified into a clean database with identical IDs; **20-paper audit: agent pass done (0 invalid page spans of 4,385 chunks; running headers and precomposed ligatures eliminated); human confirmation of the review bench outstanding**. Note: zero duplicate strong IDs passes vacuously — the pilot has only `openreview` identifiers |
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
| Parser furniture removal | Detect running headers by repetition across pages, not by matching a venue string | The banner appeared 2,590 times in 98/100 papers and would be quoted back as evidence; repetition generalises to any venue | `corpus/parse.py` |
| Ligature expansion | Targeted replacement of U+FB00-FB04, not NFKC | NFKC also flattens superscripts, changing what a formula means | `corpus/parse.py` |
| Small-caps section names | Repair only lines with two or more runs; leave 598 mangled | A single run is indistinguishable from an appendix label, and joining `J THEORETICAL` yields `JTHEORETICAL`; a wrong join is worse than a mangled name | `corpus/parse.py`, `reports/m1-parser-audit.md` |
| Export eligibility spelling | Accept both `eligible` (spec §5 enum) and `allowed` (plan example) as publishable | The same decision under two names; rejecting one would silently withhold every row the pipeline writes | `corpus/export.py` |
| Release activation | Default validator refuses until P2.2 | An empty check returning ok would let a release serve with no vectors behind it, which readiness would report as healthy | `corpus/releases.py` |
| Version identity | `_upsert_version` treats (source, revision, version) as one observation | Lets a download replace the placeholder metadata checksum in place without duplicate rows | `corpus/dedupe.py` |
| Dotenv reaches `os.environ` | `cli.load_dotenv_files()` at the CLI entry point | `Settings` parses `.env` into its own object only, so `os.environ`-based lookups (OpenReview credentials, `default_staging_dir`) read as unset and the pilot silently fell back to guest access | `cli.py`, `test_config.py` |
| Control characters in extracted text | Strip C0 and DEL (keeping tab) at `sections_from_pages`; bump `pypdf-text-v1` → `v2` | pypdf maps ligature/math glyphs onto low codepoints; NUL makes PostgreSQL reject the write. Sanitizing at the Section boundary covers every `PageAdapter`. Lossy: 0.054% of characters across 69/100 documents | `corpus/parse.py`, `configs/parsing.yaml`, `reports/m1-ingestion.md` |

## Session log

| Date | Tasks changed | Checks and results | Evidence paths / commit | Next action |
|---|---|---|---|---|
| 2026-09-09 | Repository reset; design docs revised for zero-budget local infrastructure; docs untracked | Markdown checks; no application tests | `a61a07c`, `9069dde`, `60953e6` | Implement P1.1 |
| 2026-09-09 | Docker Desktop + WSL installed; P1.1, P1.2 restored and adapted | 67 passed (22 offline, 45 integration), Ruff, mypy | `c9ec8f4`, `35c0da1`, `f652b9a` | P1.3 |
| 2026-09-10 | P1.3 implemented; P1.4 implemented; dotenv CORS fix | 119 passed, Ruff, mypy; live OpenReview listing returned 403 challenge | `9bbef4a`, `c02249e`, `d3fad84` | Add OpenReview credentials to `.env`, run and replay the 100-paper pilot, write `reports/m1-ingestion.md`, then P1.5 |
| 2026-09-12 | P1.3 audited against real PDFs; parser revised to `pypdf-text-v3`; 20-paper review bench published | 144 passed, Ruff clean, strict mypy clean. Corpus re-driven: running headers 2,590 → 0, precomposed ligatures 1,217 → 0, invalid page spans 0 of 4,385 chunks. Section-name repair deliberately partial (598 remain) | [parser audit](../../reports/m1-parser-audit.md) | Human confirmation of the bench, then M2 (P2.1) |
| 2026-09-12 | P1.5 implemented: export, validate, restore, coverage endpoint, guarded releases | 140 passed, Ruff clean, strict mypy shows 4 pre-existing errors only. Pilot snapshot 3.0 MB (100 metadata + 4,470 fulltext rows); public export withholds all 4,570 rows; restore into a clean database reproduced 100 byte-identical paper IDs | [corpus evidence](../../reports/m1-corpus.md), [data card](../data-card.md) | Fix the mypy invocation, then the 20-paper parser audit to close G1 |
| 2026-09-12 | P1.4 live pilot and replay; dotenv loading fixed; parser sanitization added (`pypdf-text-v2`) | 121 passed, Ruff, mypy. Live: 2,260 eligible notes → 100 papers, 100 PDFs, 4,470 chunks; 41/100 parses failed on NUL bytes before the fix; replay processed 0 jobs and left canonical IDs byte-identical | [ingestion evidence](../../reports/m1-ingestion.md); working tree not yet committed | Run the 20-paper parser audit against the downloaded PDFs, then P1.5 |

## Definition of a completed task

All named acceptance cases have evidence; required tests actually ran; code was reviewed; the report contains actual versions and measurements; the task commit is recorded. Engineering completion and human-observation checkpoints are recorded separately. Do not publish synthetic or planned results as measured outcomes.
