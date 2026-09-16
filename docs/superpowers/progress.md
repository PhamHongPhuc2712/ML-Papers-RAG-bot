# ML Research Copilot — Progress Tracker

Last updated: 2026-09-16.

## Current state

- Implementation: **4 / 30 tasks complete** (P1.1–P1.4); P1.3's parser audit and P1.4's pilot and replay are both closed. Next task: P1.5.
- MVP implementation: **4 / 25 tasks complete**.
- Passing implementation gates: **0 / 6** (G1's parser audit passes at 95% usable text and the 100-paper replay passes with zero identity churn; only the snapshot export/restore from P1.5 remains).
- Corpus ingested: 100 ICLR 2024 papers, **7,500 chunks** in `copilot_pilot_v2` (2026-09-16 re-run on model-token windows, registry records and observed membership). The superseded 2026-09-15 pilot — 100 papers, 4,713 word-counted chunks — still sits in the `copilot` database and should be dropped before the multi-venue run. Corpus **indexed** (Qdrant): none yet — that is P2.2. Benchmarks observed: none yet. User-study observations: none yet.
- Test evidence on this host (2026-09-16, WSL/Linux, 12 cores / 23 GB RAM): full suite **161 passed**; Ruff clean; mypy clean **under the project's strict config**, which the documented command had never loaded (see the note below the session log). Corpus on disk: 2,260 ICLR 2024 PDFs (13.70 GB), the 89 MB `papers.parquet` registry and the derived venue-year index, under `${DATA_DIR}/sources/papercli/`.

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
| P1.4 | Ingest an accepted-paper pilot through resumable jobs | Done; 100-paper pilot and replay run 2026-09-15 from a local mirror, identity/version/chunk digests identical across runs, 0 retries | P1.2, P1.3 | `d3fad84`; [ingestion evidence](../../reports/m1-ingestion.md) |
| P1.5 | Export immutable corpus snapshots and publish coverage | Not started | P1.4 | Not produced |
| P2.1–P2.5 | Retrieval and ranking | Not started | P1.5 | Not produced |
| P3.1–P3.5 | Recommendation workspace | Not started | P2.4 | Not produced |
| P4.1–P4.5 | Evidence assistant | Not started | P2.3, P3.1 | Not produced |
| P5.1–P5.5 | Portfolio hardening (P5.3 deployment half and public rate limiting deferred) | Not started | P1–P4 | Not produced |
| P6.1–P6.5 | Research expansion (optional) | Not started | G5 | Not produced |

## Gate board

| Gate | Required evidence | State |
|---|---|---|
| G1 | 100-paper replay, identity checks, 20-paper parser audit, snapshot restore | Partly met — identity checks pass; parser audit passes (19/20 usable, 2026-09-14); 100-paper replay passes with zero duplicate strong IDs and no canonical ID change (2026-09-15); snapshot export/restore outstanding with P1.5 |
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
| Pilot source (2026-09-15) | Ingest the pilot from a local mirror of the papercli datasets (`sources/papercli.py`) rather than OpenReview; `adopt_pdf` registers an already-mirrored file after re-checksumming it | OpenReview guest access returns `ChallengeRequiredError` (403) on both routes. The mirror is captured once, so the run makes no network request, Hugging Face stays an offline artifact store, and `download.py`'s host/address/redirect restrictions are untouched. Provenance records `papercli` plus the dataset revision, not an OpenReview retrieval | `configs/corpus.yaml`, [ingestion evidence](../../reports/m1-ingestion.md) |
| Mirror membership (2026-09-15) | Acceptance asserted per manifest via `membership_is_acceptance`, fail-closed when absent | The index has no decision field, and a venue-year listing is not reliably the accepted set: ICLR 2024 holds exactly its 2,260 accepted papers, ICLR 2023 holds 3,792 against roughly 1,574. Without the flag every record is `unknown` and `is_eligible` rejects the listing | `sources/papercli.py` |
| Membership evidence (2026-09-16) | Acceptance and track are classified per record from the venue's own `track` label by `classify_track`; `membership_is_acceptance` removed. Main is matched **positively**, so an unrecognized label is `other` and excluded | The label is specific evidence: ICLR 2023's 3,792 rows resolve to 1,573 accepted plus 2,219 "Submitted to ICLR 2023", matching the published count. All 262 labels across 2023-2026 classify with none unmatched: 85,732 main/accepted, 7,499 Findings, 4,883 workshop, 2,219 submissions, 1,118 co-located conferences (WMT, IWSLT, ArabicNLP filed under ACL/EMNLP), 706 industry, 414 demo, 157 shared task, 61 tutorial. Precedence is load-bearing — 163 of 165 workshop labels begin "Proceedings of" | `sources/base.py`, `configs/corpus.yaml`, `data/fixtures/tracks/` |
| Mirror record source (2026-09-16) | Ingest from the registry `papers.parquet` (11 columns) via `corpus mirror-index`, not the five-column `browse/<venue>/<year>.parquet` view; `source_item_id` is the registry id and both the registry id and the venue's forum id are recorded as aliases | The browse view publishes no authors, no track and no PDF path, which is why the pilot recorded "no authors" and had to assert acceptance. The registry gives 100% author and abstract coverage on ICLR 2024, `track` carries the venue's own decision label, and `hf_pdf_path` removes the first-page text matching that mapped PDFs before. The rebuilt index agrees with that mapping on all 2,260 checksums and titles | `corpus/mirror.py`, `sources/papercli.py`, `configs/corpus.yaml` |
| Chunk window unit (2026-09-16) | Windows are cut on the pinned BGE-M3 tokenizer's spans, not whitespace words; `chunker_version` → `fixed-window-v2`, tokenizer pinned by revision `5617a9f` and sha256 under `${DATA_DIR}/models/tokenizers/`, with no fallback to word counting | `configs/parsing.yaml` documented the window in embedding-tokenizer tokens while the code counted words. Measured over the 100-paper pilot's 4,713 chunks: median 631 BGE-M3 tokens and **51.4% above the 600 hard cap**, prose median 656, 1.73 tokens per word. Re-chunked with the real tokenizer: **0% over the cap**, median 450. Chunk text is now a literal slice of the section taken on token character offsets, so nothing is detokenized or rewritten | `corpus/chunk.py`, `configs/parsing.yaml` |
| Source PDF retention (2026-09-14) | Corpus PDFs may be deleted once their chunks are stored, so bulk ingestion can stream venue by venue instead of holding every PDF at once | No planned task reads a corpus PDF after parsing: the page-image path in §9 and P4.3/P4.4 covers *private uploads* (`uploads/images.py`), not corpus papers. Re-parsing under a new `parser_version` would need them again, so settle the parser choice first | spec §9, P4.3/P4.4 |

## Session log

| Date | Tasks changed | Checks and results | Evidence paths / commit | Next action |
|---|---|---|---|---|
| 2026-09-09 | Repository reset; design docs revised for zero-budget local infrastructure; docs untracked | Markdown checks; no application tests | `a61a07c`, `9069dde`, `60953e6` | Implement P1.1 |
| 2026-09-09 | Docker Desktop + WSL installed; P1.1, P1.2 restored and adapted | 67 passed (22 offline, 45 integration), Ruff, mypy | `c9ec8f4`, `35c0da1`, `f652b9a` | P1.3 |
| 2026-09-10 | P1.3 implemented; P1.4 implemented; dotenv CORS fix | 119 passed, Ruff, mypy; live OpenReview listing returned 403 challenge | `9bbef4a`, `c02249e`, `d3fad84` | Add OpenReview credentials to `.env`, run and replay the 100-paper pilot, write `reports/m1-ingestion.md`, then P1.5 |
| 2026-09-14 | Environment rebuilt on WSL; extracted-text sanitization fixed; 2,260 ICLR 2024 PDFs collected; P1.3 audit measured | 120 passed, Ruff, mypy; 3,645 real chunks accepted by PostgreSQL; audit 19/20 usable text | `96130ef`, `4f8a3b7`; [parser evidence](../../reports/m1-parser-audit.md) | Fix the four structure defects (headings, caption-sections, header leakage, over-segmentation), then run venue ingestion and P1.5 |
| 2026-09-14 | Five parser/section defects repaired; `pypdf-text-v2` | 129 passed, Ruff, mypy; mangled headings 198→1, header leaks 531→0, prose in caption sections 28%→3%, prose chunk median 182→401 tokens | `c541429`, `8829d2f`; [parser evidence](../../reports/m1-parser-audit.md) | Close P1.4 with a real pilot and replay |
| 2026-09-15 | `sources/papercli.py` mirror adapter and `adopt_pdf` handler; 100-paper pilot and replay | 134 passed, Ruff, mypy; 300 jobs, 0 retries, 100 papers / 4,713 chunks; replay identical on identity, version and chunk digests with 0 new jobs | [ingestion evidence](../../reports/m1-ingestion.md) | P1.5: export snapshots, coverage endpoint and data card |

| 2026-09-16 | mypy was never running strict | 27 files clean under `--config-file backend/pyproject.toml`; 4 pre-existing errors fixed (untyped lifespan, unparameterized dict, Any return, redundant cast) | pending commit | A3 |
| 2026-09-16 | `corpus mirror` capture command (B1) | 172 passed, Ruff, mypy strict; live path verified against `papercli-papers-jmlr` (revision resolved in 1.0 s, two members fetched and checksummed); superseded v1 pilot truncated from the `copilot` database | pending commit | B2 driver, then the WACV 2023 smoke run |
| 2026-09-16 | Pilot re-run and replay on the new pipeline (A4) | 100 papers, 200 identifiers, 561 authors, **7,500 chunks**, 300 jobs, 0 retries, 69 s across 4 workers; 0 of 7,500 chunks over the 600-token cap (was 51.4%); replay enqueued 0 jobs with all three digests identical | [ingestion evidence](../../reports/m1-ingestion.md) | B1: `corpus mirror` download half, then the venue-by-venue run |
| 2026-09-16 | Per-record membership from the venue label (A3) | 161 passed, Ruff, mypy strict; all 262 registry labels classify with exact row totals asserted; real ICLR 2024 index yields 2,260 eligible, all main/accepted (1,807 poster, 367 spotlight, 86 oral) | pending commit | A4: re-run the pilot end to end on the new chunker, authors and observed decisions |
| 2026-09-16 | Registry-backed mirror index (A2) | 143 passed, Ruff clean; `corpus mirror-index --venue ICLR --year 2024` → 2,260 rows, 2,260 with verified local PDFs, 0 missing; 100% authors and abstracts | pending commit | A3: map `track` to a canonical track and an observed decision |
| 2026-09-16 | Chunk window measured in model tokens (A1) | 138 passed (78 offline), Ruff, mypy; 15 pilot papers re-chunked with the pinned tokenizer: 1,010 chunks, median 450 tokens, 0 over the 600 cap | pending commit | A2: read `papers.parquet` (authors, abstract, track, `hf_pdf_path`) instead of the 5-column browse shard |

> Evidence note: every "mypy strict clean" recorded before 2026-09-16 was produced by
> `mypy backend/src` run from the repository root, which finds no configuration file there
> and therefore ran with mypy's defaults, not `strict = true`. The command now passes
> `--config-file backend/pyproject.toml` in CI, CLAUDE.md and the backend README.

## Definition of a completed task

All named acceptance cases have evidence; required tests actually ran; code was reviewed; the report contains actual versions and measurements; the task commit is recorded. Engineering completion and human-observation checkpoints are recorded separately. Do not publish synthetic or planned results as measured outcomes.
