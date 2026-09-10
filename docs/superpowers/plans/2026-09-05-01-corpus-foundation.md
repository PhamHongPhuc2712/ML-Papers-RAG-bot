# Corpus Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a replayable, auditable accepted-paper corpus before retrieval tuning.

**Architecture:** Offline source adapters normalize works and versions in PostgreSQL. A durable worker parses PDFs; immutable Parquet exports and staged releases feed later indexing.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL, Qdrant, Next.js/TypeScript, pytest, Playwright; subsystem additions are specified below.

**Spec:** [Technical specification](../specs/2026-09-05-ml-research-copilot-design.md). Read the relevant sections and global contracts before implementing.

**Status:** In execution. P1.1–P1.4 implemented (see [progress.md](../progress.md)); the P1.4 live pilot is pending OpenReview credentials.

## Global Constraints

- Python 3.12; Node.js 22; TypeScript strict mode; UTF-8 files; UTC timestamps.
- Lock Python and JavaScript dependencies; pin container images and model revisions before benchmark or deployment runs.
- Hugging Face is an offline corpus/artifact destination and is never queried during user requests.
- Public corpus data and private user documents use separate Qdrant collections and storage namespaces.
- Derive user identity from a verified token; never trust a client-supplied user_id.
- All benchmark runs record corpus, split, model, configuration, code, hardware, and seed versions.
- No paid cloud resources, public publishing, or application implementation occur while preparing these documents.
- Postgres and Qdrant are self-hosted via Docker on the developer's own machine by default; no managed database, vector, or storage subscription is used unless a future, explicitly-recorded decision changes this.
- All persistent local data lives under one configurable root directory, default `C:\ml-copilot-data\` on the developer's machine, so the entire dataset can be deleted by removing that directory.
- The hosted LLM/multimodal generation API is the one paid resource in the project, funded personally by the developer; its cost tracking and daily spend cap from P5.2 remain required because real personal money is involved.

## Entry checkpoint and working conventions

Entry: repository is empty apart from these documents; the design is reviewed. Start with 100 accepted papers, not an assumed full-corpus download. Read spec §§1–6, 11–12.

Run commands from the repository root. Paths name future files. Each task contains a concrete regression example and critical implementation logic; finish the stated contracts and acceptance cases in the same task. These snippets are design artifacts, not tested application code. A task is typically several focused work sessions; each checkbox may be split into 2–5 minute edit/run actions while executing. No task is complete merely because its example test passes.

For each task: capture the expected behavioral failure, implement one behavior at a time, run the listed suite, inspect the diff, and record command/exit status, report path and commit in the progress tracker. Commit only that task's files after review. Keep external API tests opt-in and deterministic fixtures in normal CI. Missing dependencies are setup failures, not the intended red test.

## P1.1: Create a runnable API and isolated persistence test harness

**Depends on:** Design review

**Files:**

- Create `backend/pyproject.toml`, `backend/uv.lock`, `backend/src/copilot/{__init__,app,config,contracts}.py`.
- Create `backend/src/copilot/db/{models,session}.py`, `backend/migrations/env.py`, `backend/alembic.ini`.
- Create `compose.yaml`, `infra/Dockerfile.backend`, `.env.example`, `.gitignore`, `.github/workflows/checks.yml`.
- Create `backend/tests/conftest.py`, `backend/tests/integration/test_health.py`.

**Interfaces:** `create_app(settings: Settings, overrides: dict | None = None) -> FastAPI`; `Settings` validates environment; `GET /health/live` returns status=ok; `/health/ready` checks DB and, after an active release exists, vector availability. Define section 6 contract models now. The `api_client` pytest fixture yields a synchronous FastAPI TestClient against an isolated migrated PostgreSQL database and test Qdrant collections, with deterministic model overrides.

- [x] Write the regression test below in `backend/tests/integration/test_health.py` and add the additional acceptance cases listed for this task.

```python
def test_liveness_and_missing_release(api_client):
    assert api_client.get("/health/live").json()["status"] == "ok"
    response = api_client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "corpus_not_ready"
```

- [x] Run `uv run --project backend pytest backend/tests/integration/test_health.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [x] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
from fastapi import FastAPI

def create_app(settings, overrides=None):
    app = FastAPI(title="ML Research Copilot", version="0.1.0")
    app.state.settings = settings
    app.state.overrides = overrides or {}

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    return app
```

Add readiness using a bounded DB query and the active-release lookup; empty database is not ready. Configure `uv` package discovery for `src/copilot`, pytest unit/integration markers, Ruff and strict typing. Dependencies: FastAPI, uvicorn, Pydantic Settings, SQLAlchemy, psycopg, Alembic, qdrant-client, httpx; pytest/Ruff/mypy as development dependencies. Resolve compatible releases, commit the lockfile, and pin Postgres/Qdrant container versions. Postgres and Qdrant are self-hosted Docker services on the developer's machine; no managed database or vector endpoint is configured. `Settings` validates `DATA_DIR` — documented Windows default `C:\ml-copilot-data\` — and refuses to start when it is unset, so the path is configurable rather than hardcoded. `compose.yaml` bind-mounts the Postgres and Qdrant services to `${DATA_DIR}/postgres/` and `${DATA_DIR}/qdrant/` instead of named Docker volumes, keeping container state inside the one deletable data root, and `.env.example` documents the variable with that default. Compose test services use distinct database names, collection prefixes and a separate subdirectory under the same root; refuse destructive fixture cleanup without the `test_` prefix. `api_client` creates a fresh app for each test and rolls back/cleans only its namespace. Add `user_headers(user_id: str)` fixture in M3, not a production auth bypass. CI runs lint/unit tests immediately; expand its scope with each subsystem.

- [x] Verify the additional acceptance cases: process liveness does not imply readiness; missing DB produces typed 503 without credentials in the response; production rejects mock mode/default secrets; startup rejects an unset `DATA_DIR` in every environment; Postgres and Qdrant start and pass readiness on Windows bind mounts under `DATA_DIR`, with any ownership or filesystem-type warnings recorded in the report; migrations start cleanly; fixture cleanup cannot target a normal database or a data root outside `${DATA_DIR}/test/`.

- [x] Run `uv run --project backend pytest backend/tests/integration/test_health.py -q` again, then run `uv run --project backend ruff check backend` and `uv run --project backend mypy backend/src`. Expected: all listed cases pass, with no skipped required integration checks.

- [x] Save `reports/m1-foundation.md` with chosen dependency/image versions and the clean migration output; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `build: establish api and isolated persistence harness`.

## P1.2: Normalize publication identity and preserve version provenance

**Depends on:** P1.1

**Files:**

- Create `backend/src/copilot/corpus/{normalize,dedupe}.py`.
- Create `backend/migrations/versions/0001_corpus.py`; modify `backend/src/copilot/db/models.py`.
- Create `backend/tests/unit/test_identity.py`, `backend/tests/integration/test_deduplication.py`, `data/fixtures/metadata.jsonl`.

**Interfaces:** `normalize_doi(value: str) -> str`; `normalize_arxiv(value: str) -> tuple[str,str|None]`; `resolve_paper(record: dict, session) -> UUID`. Records include source, external IDs, title, authors, venue/year, source_revision, retrieved_at and acceptance decision. UUID is allocated once then resolved through paper_identifiers.

- [x] Write the regression test below in `backend/tests/unit/test_identity.py` and add the additional acceptance cases listed for this task.

```python
from copilot.corpus.normalize import normalize_doi, normalize_arxiv

def test_external_id_normalization():
    assert normalize_doi(" https://doi.org/10.1000/ABC ") == "10.1000/abc"
    assert normalize_arxiv("2401.01234v2") == ("2401.01234", "v2")
```

- [x] Run `uv run --project backend pytest backend/tests/unit/test_identity.py backend/tests/integration/test_deduplication.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [x] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
import re

def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value)
    if not re.fullmatch(r"10\.\d{4,9}/\S+", value):
        raise ValueError("invalid_doi")
    return value

def normalize_arxiv(value: str) -> tuple[str, str | None]:
    value = re.sub(r"^https?://arxiv.org/(abs|pdf)/", "", value.strip())
    value = value.removesuffix(".pdf")
    match = re.fullmatch(r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", value)
    if not match:
        raise ValueError("invalid_arxiv_id")
    return match.group(1), match.group(2)
```

Create the corpus tables from spec section 5 with uniqueness and FK constraints. Use transactional lookup/upsert of identifiers and explicit conflict records when IDs disagree. Add redirects when manually merging work identities, while preserving paper_versions. Raw source JSON is stored in a private/local staging artifact by checksum; canonical metadata includes per-field provenance. An exact title with incompatible authors never auto-merges. Missing DOI is valid; invalid supplied DOI is quarantined. Add idempotent seed fixture import to `copilot.cli fixtures load` for later integration tests.

- [x] Verify the additional acceptance cases: two providers for one DOI resolve to one work; concurrent duplicate upserts produce one ID; arXiv versions remain separate document versions; contradictory strong IDs create a review conflict; Unicode title variants don't lose original spelling; unrelated papers with identical titles remain separate.

- [x] Run `uv run --project backend pytest backend/tests/unit/test_identity.py backend/tests/integration/test_deduplication.py -q` again, then run the corpus migration up/down/up against the isolated test database and all identity tests. Expected: all listed cases pass, with no skipped required integration checks.

- [x] Save `reports/m1-identity.md` with merge/conflict fixtures and migration output; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add canonical paper identity and source provenance`.

## P1.3: Parse documents into traceable sections and chunks

**Depends on:** P1.2

**Files:**

- Create `backend/src/copilot/corpus/{parse,chunk}.py`, `configs/parsing.yaml`.
- Create `backend/tests/unit/test_chunks.py`, `backend/tests/integration/test_parser.py`.
- Create `data/fixtures/papers/fixture.pdf` and its redistributable source text/license note.

**Interfaces:** `Section` is a frozen dataclass with `name: str`, `text: str`, `page_start: int|None`, `page_end: int|None`, `ordinal: int`; `chunk_sections(sections: list[Section], tokenize, detokenize, target: int, overlap: int) -> list[dict]`; `parse_pdf(path: Path) -> list[Section]`. Chunk dicts expose section/text/token_count/page span/ordinal.

- [x] Write the regression test below in `backend/tests/unit/test_chunks.py` and add the additional acceptance cases listed for this task.

```python
from copilot.corpus.chunk import Section, chunk_sections

def test_overlap_never_crosses_sections():
    sections = [Section("Method", "a b c d e", 1, 1, 0),
                Section("Results", "f g h", 2, 2, 1)]
    chunks = chunk_sections(sections, str.split, " ".join, target=4, overlap=1)
    assert [c["text"] for c in chunks] == ["a b c d", "d e", "f g h"]
    assert chunks[-1]["page_start"] == 2
    assert all(c["token_count"] <= 4 for c in chunks)
```

- [x] Run `uv run --project backend pytest backend/tests/unit/test_chunks.py backend/tests/integration/test_parser.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [x] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Section:
    name: str
    text: str
    page_start: int | None
    page_end: int | None
    ordinal: int

def chunk_sections(sections, tokenize, detokenize, target, overlap):
    if not 0 <= overlap < target:
        raise ValueError("invalid_chunk_window")
    chunks = []
    for section in sections:
        tokens = tokenize(section.text)
        for start in range(0, len(tokens), target - overlap):
            part = tokens[start:start + target]
            chunks.append(dict(section=section.name, text=detokenize(part),
                               token_count=len(part), page_start=section.page_start,
                               page_end=section.page_end, ordinal=len(chunks)))
            if start + target >= len(tokens):
                break
    return chunks
```

Wrap pinned Docling output in the Section contract, retaining original source spans separately from embedding text. Use the BGE tokenizer in production; whitespace tokens above are a deterministic fixture only. Implement heading-aware splitting using target 450/hard cap 600/overlap 60 and version the policy; begin with the fixed-window function as baseline. Tables get coherent blocks and labeled fragments; references get separate records and are excluded from default evidence. Assign UUIDv5 chunk IDs as specified. Reject encrypted/corrupt/oversized input with typed parse status. Add parser version/checksum to persisted paper_versions. The offline parser integration suite runs against fixture PDF content that the repo owns, not copyrighted source downloads.

- [x] Verify the additional acceptance cases: headings/page spans survive; long sections respect cap; empty sections emit no chunks; unknown pages remain null; tables/references are classified; rerunning identical input keeps chunk IDs; changed parser/chunker revision changes chunk identity; multi-column reading order is checked in the manual audit.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_chunks.py backend/tests/integration/test_parser.py -q` again, then run parser/chunker unit tests and create the 20-paper audit with explicit source/version/page checks. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m1-parser-audit.md` listing 20 outcomes, failure categories, parser configuration and usable-text rate; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: parse section-aware chunks with source provenance`.

## P1.4: Ingest an accepted-paper pilot through resumable jobs

**Depends on:** P1.2, P1.3

**Files:**

- Create `backend/src/copilot/jobs/{queue,worker}.py`, `backend/src/copilot/corpus/sources/{base,openreview,proceedings,arxiv,semantic_scholar}.py`.
- Create `backend/src/copilot/corpus/download.py`, `backend/src/copilot/cli.py`, `configs/corpus.yaml`.
- Create `backend/tests/unit/test_membership.py`, `backend/tests/integration/test_ingestion.py` and captured synthetic provider-response fixtures.

**Interfaces:** `is_eligible(record: dict, manifest: dict) -> bool`; `ingest(manifest_path: Path, limit: int|None) -> str` returns run ID; `enqueue(kind: str, payload: dict, key: str) -> UUID`; `run_once(worker_id: str) -> bool`. Source adapters implement `fetch_page(cursor: str|None) -> tuple[list[dict],str|None]` and emit normalized source records for P1.2.

- [x] Write the regression test below in `backend/tests/unit/test_membership.py` and add the additional acceptance cases listed for this task.

```python
from copilot.corpus.sources.base import is_eligible

def test_under_review_is_not_accepted():
    manifest = {"venue": "ICLR", "years": [2024], "track": "main"}
    row = {"venue": "ICLR", "year": 2024, "track": "main",
           "decision": "under_review", "withdrawn": False}
    assert not is_eligible(row, manifest)
    assert is_eligible({**row, "decision": "accepted"}, manifest)
    assert not is_eligible({**row, "decision": "accepted", "withdrawn": True}, manifest)
```

- [x] Run `uv run --project backend pytest backend/tests/unit/test_membership.py backend/tests/integration/test_ingestion.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [x] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def is_eligible(record: dict, manifest: dict) -> bool:
    return (
        record["venue"] == manifest["venue"]
        and record["year"] in manifest["years"]
        and record["track"] == manifest["track"]
        and record["decision"] == "accepted"
        and not record.get("withdrawn", False)
    )
```

Store exact venue-year source configuration; map official decisions to the enum before eligibility checks. Main API ingest command: `uv run --project backend python -m copilot.cli corpus ingest --manifest configs/corpus.yaml --limit 100`. Default manifest selects accepted ICLR 2024 main-track items sorted by source ID before taking 100, recording that deterministic sample bias. Implement all source interfaces, with enrichment optional and independently retryable. Durable jobs lease using a short transaction with `FOR UPDATE SKIP LOCKED`, 60-second lease, heartbeat every 15 seconds, at most five attempts and exponential backoff capped at 60 seconds honoring longer Retry-After. Use server-side timestamps and lease tokens. Validate URL/IP/redirect restrictions in download.py; stream under size limits. Inject clock and provider transports in tests. Persist cursor only after normalized rows and emitted jobs are committed.

- [x] Verify the additional acceptance cases: kill/restart after metadata write and after parse write; repeated ingestion creates no duplicates; two workers cannot complete one lease twice; expired worker cannot overwrite new owner; 429 respects backoff; rejected/withdrawn records excluded; failed enrichment leaves base paper searchable; private-network redirect rejected; poisoned PDF produces classified failure.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_membership.py backend/tests/integration/test_ingestion.py -q` again, then run a real 100-paper pilot once under provider limits, replay it, and compare canonical IDs/counts/checkpoints. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m1-ingestion.md` with coverage, provider membership queries, retry counts and replay differences; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: ingest accepted papers with durable resumable jobs`.

## P1.5: Export immutable corpus snapshots and publish coverage

**Depends on:** P1.4

**Files:**

- Create `backend/src/copilot/corpus/{export,releases,api}.py`, modify `backend/src/copilot/cli.py` and `backend/src/copilot/app.py`.
- Create `backend/tests/unit/test_export_policy.py`, `backend/tests/integration/test_snapshot.py`, `configs/artifacts.yaml`.
- Create `docs/data-card.md`, `reports/m1-corpus.md`.

**Interfaces:** `public_export_rows(rows: list[dict]) -> list[dict]`; `export_snapshot(run_id: str, destination: Path) -> dict` returns a manifest; `validate_manifest(path: Path) -> dict`; `GET /v1/corpus/coverage` returns source counts, known denominators, nullable percentages and as-of timestamps. A staged corpus release cannot become active until P2.2 indexes validate.

- [ ] Write the regression test below in `backend/tests/unit/test_export_policy.py` and add the additional acceptance cases listed for this task.

```python
from copilot.corpus.export import public_export_rows

def test_private_or_unlicensed_text_is_not_exported():
    rows = [
        {"id": "a", "kind": "corpus", "redistribution": "allowed", "text": "public"},
        {"id": "b", "kind": "upload", "redistribution": "allowed", "text": "private"},
        {"id": "c", "kind": "corpus", "redistribution": "unknown", "text": "unknown"},
    ]
    assert [r["id"] for r in public_export_rows(rows)] == ["a"]
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_export_policy.py backend/tests/integration/test_snapshot.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def public_export_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows
            if row["kind"] == "corpus" and row["redistribution"] == "allowed"]
```

Produce separate metadata/fulltext/edges shards, per-record eligibility, manifest checksum, source revisions, parser/chunker versions, schema version and exact counts. The export filter applies to each publishable artifact, including metadata if rights are restricted. Configure public HF publication as an explicit CLI mode separate from local export; the plan-preparation task never calls it. Local export artifacts stage under `${DATA_DIR}/exports/`, outside the repository tree, so snapshots are covered by the delete-one-directory guarantee and no bulk data reaches version control; relative `--out` and `--manifest` paths resolve against that directory so the commands need no shell-specific variable syntax. `corpus export --run RUN_ID --out pilot` creates local artifacts; `corpus validate --manifest pilot/manifest.json` verifies them; `corpus publish --manifest ... --repo OWNER/NAME` is only for an authorized implementation-stage publication. Reject private fields and unknown schemas. Preserve prior snapshots. Coverage null denominator never becomes 100%. Restore a local snapshot into an empty staging DB to prove portability.

- [ ] Verify the additional acceptance cases: checksum tampering fails; schema mismatch fails before mutation; export excludes private/unknown-rights text; missing abstracts and missing PDFs get separate counts; restore reproduces counts/identities; snapshots are immutable; HF failure does not affect serving; no active release before a matching validated index pair exists.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_export_policy.py backend/tests/integration/test_snapshot.py -q` again, then run all corpus tests and validate/restore the pilot export in a clean staging namespace. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m1-corpus.md`, local snapshot manifest/checksum and the source/data card; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: export versioned corpus snapshots and coverage`.

## Exit checkpoint

G1 passes only with the pilot replay, conflict report, artifact restore and 20-paper parser audit. Record actual counts and failures. The active serving release remains unset until P2.2. Next: Plan 2.
