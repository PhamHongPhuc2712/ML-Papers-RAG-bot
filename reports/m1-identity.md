# P1.2 identity and provenance evidence

Date: 2026-09-06
Task: P1.2, normalize publication identity and preserve version provenance
Worktree: `codex/p1-identity`

## Runtime and services

The required portable runtime was used for every command:
`C:\Computing\GitHub\ML-Papers-RAG-bot\.worktrees\p1-foundation\.superpowers\runtime\uv-0.12.10\uv.exe`.
It reported uv 0.12.10 and Python 3.12.5. The locked environment reported
Alembic 1.16.5, FastAPI 0.116.1, httpx 0.28.1, psycopg 3.2.9,
qdrant-client 1.19.0, SQLAlchemy 2.0.43, pytest 8.4.1, Ruff 0.12.10 and
mypy 1.17.1. The disposable services were PostgreSQL 17.11 at
`127.0.0.1:55432/test_copilot` and Qdrant 1.19.1 at `127.0.0.1:56333`.

## TDD red evidence

The required focused command was run before the P1.2 modules existed:

```text
uv.exe run --project backend pytest backend/tests/unit/test_identity.py backend/tests/integration/test_deduplication.py -q
2 errors during collection
ModuleNotFoundError: No module named 'copilot.corpus'
ModuleNotFoundError: No module named 'copilot.cli'
```

After adding the same-source concurrent replay case, the targeted test failed
with PostgreSQL unique violation `uq_source_records_revision_checksum`.
After adding the versionless arXiv assertion, the targeted test failed with
`AssertionError: assert 'v1' is None`. The preprint/proceedings year case first
returned two UUIDs, and the conflict provenance assertion first observed zero
rows for the conflicted source record. Each failure was corrected by the
task-scoped implementation and then re-run green.

## Green and verification evidence

The focused identity suite ran against the configured PostgreSQL service with
no skipped integration checks:

```text
uv.exe run --project backend pytest backend/tests/unit/test_identity.py backend/tests/integration/test_deduplication.py -q
26 passed in 7.05s
```

The full backend suite and offline CI subset passed:

```text
uv.exe run --project backend pytest backend/tests -q
38 passed in 10.31s

uv.exe run --project backend pytest backend/tests -m "not integration" -q
13 passed, 25 deselected in 0.09s
```

Static checks and whitespace validation passed:

```text
uv.exe run --project backend ruff check backend
All checks passed!

uv.exe run --project backend mypy backend/src
Success: no issues found in 11 source files

git diff --check
no output; exit code 0
```

Migration replay against the isolated PostgreSQL database passed in order:

```text
uv.exe run --project backend alembic -c backend/alembic.ini upgrade head
exit code 0; database already at 0001_corpus

uv.exe run --project backend alembic -c backend/alembic.ini downgrade 0000_foundation
Running downgrade 0001_corpus -> 0000_foundation

uv.exe run --project backend alembic -c backend/alembic.ini upgrade head
Running upgrade 0000_foundation -> 0001_corpus
```

The resulting database tables included `venues`, `papers`,
`paper_identifiers`, `paper_versions`, `authors`, `paper_authors`,
`source_records`, `field_provenance`, `identity_conflicts`,
`quarantine_records`, `paper_redirects`, and the unchanged P1.1 release
tables.

The fixture command was exercised directly twice:

```text
uv.exe run --project backend python -m copilot.cli fixtures load data/fixtures/metadata.jsonl --database-url <test-url> --staging-dir <temp-dir>
{"conflicts": 0, "loaded": 3, "quarantined": 0}

uv.exe run --project backend python -m copilot.cli fixtures load data/fixtures/metadata.jsonl --database-url <test-url> --staging-dir <temp-dir>
{"conflicts": 0, "loaded": 3, "quarantined": 0}
```

Read-only count inspection after replay reported `papers=2`,
`source_records=3`, and `paper_versions=3`. The acceptance fixtures covered
normalization, invalid-ID quarantine, provider and replay idempotency, arXiv
version splitting and versionless aliases, cross-source DOI/arXiv identity,
contradictory and incompatible metadata conflicts, Unicode title preservation,
conservative title/author/year matching, concurrent PostgreSQL upserts,
manual redirects with version/source/authorship preservation, checksummed raw
artifacts, per-field provenance, malformed records, and fixture replay.

## Self-review

The resolver owns no commit; successful records and review outcomes are
flushed for the caller to commit. Strong aliases use PostgreSQL uniqueness and
conflict-safe inserts, title-only equality requires compatible authors and
years, and arXiv versions stay in `paper_versions` while the alias remains
versionless. Manual merges retain both paper UUID history and redirect rows.
Only identity/provenance tables were added; chunks, citations, jobs, user
objects, retrieval, and later release schemas remain deferred.

## Limitations

The source artifact checksum is a canonical metadata-record checksum when a
provider does not supply a document checksum; PDF validation and parsing are
owned by later tasks. Review conflicts are durable but have no review UI yet.
The command evidence uses disposable loopback services and synthetic fixture
metadata; no live provider corpus was ingested.

Previous-round commit: `226c971` (`feat: add canonical paper identity and source provenance`).

## Round 1 review-fix evidence

The amended focused identity suite passed after the review fixes:

```text
uv.exe run --project backend pytest backend/tests/unit/test_identity.py backend/tests/integration/test_deduplication.py -q
34 passed in 11.41s

uv.exe run --project backend pytest backend/tests -q
46 passed in 14.47s

uv.exe run --project backend pytest backend/tests -m "not integration" -q
13 passed, 33 deselected in 0.18s

uv.exe run --project backend ruff check backend
All checks passed!

uv.exe run --project backend mypy backend/src
Success: no issues found in 11 source files
```

Migration replay passed through `0002_identity_hardening -> 0001_corpus ->
0000_foundation -> 0001_corpus -> 0002_identity_hardening`. Direct fixture
replay twice returned `{"conflicts": 0, "loaded": 3, "quarantined": 0}` and
the resulting counts were `(2, 3, 3)` for papers, source records, and paper
versions. Review fixes cover concurrent winner adoption, idempotent conflict
and quarantine persistence, exact DOI/arXiv year compatibility, arXiv version
mismatch rejection, trusted atomic artifacts, duplicate-version merge
rejection, merge provenance/authorship, and explicit test-service variables.

Current review-fix commit hash is recorded in the ignored task report at
`.superpowers/sdd/2026-09-05-01-corpus-foundation/task-2-report.md`.
