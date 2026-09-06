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

## Round 2 review-fix evidence

The second review pass added PostgreSQL regressions before implementation. The
initial collection run failed because `MergeValidationError` was not yet
defined; the first implementation run then produced the intended three red
deduplication cases (incompatible-race loser binding, an unsupported query
assertion, and a post-rollback duplicate-version assertion).

The corrected regressions passed:

```text
uv.exe run --project backend pytest backend/tests/integration/test_deduplication.py -k "concurrent_incompatible_doi_resolution_creates_one_conflict or manual_merge_preserves_versions_and_records_redirect or manual_merge_rejects_duplicate_versions_within_losing_paper" -q
3 passed, 25 deselected in 1.87s
```

The compatible-race provenance assertion was rerun with the amendments:

```text
uv.exe run --project backend pytest backend/tests/integration/test_deduplication.py -k "concurrent_doi_resolution_converges_without_orphan_papers or concurrent_incompatible_doi_resolution_creates_one_conflict or manual_merge_preserves_versions_and_records_redirect or manual_merge_rejects_duplicate_versions_within_losing_paper" -q
4 passed, 24 deselected in 2.49s
```

```text
uv.exe run --project backend pytest backend/tests/integration/test_migrations.py -q
1 passed in 1.03s
```

Full validation passed against explicit PostgreSQL/Qdrant test services:

```text
uv.exe run --project backend pytest backend/tests/unit/test_identity.py backend/tests/integration/test_deduplication.py backend/tests/integration/test_migrations.py -q
37 passed in 11.74s

uv.exe run --project backend pytest backend/tests -q
49 passed in 15.08s

uv.exe run --project backend pytest backend/tests -m "not integration" -q
13 passed, 36 deselected in 0.18s

uv.exe run --project backend ruff check backend
All checks passed!

uv.exe run --project backend mypy backend/src
Success: no issues found in 11 source files
```

Migration down/up/no-op replay passed, the legacy duplicate-conflict
migration test retained the earliest deterministic row and unrelated conflict,
and direct fixture replay twice returned `{"conflicts": 0, "loaded": 3,
"quarantined": 0}` with final counts `(2, 3, 3)`. The fixes check metadata
before raced candidate deletion, reject duplicate version keys within either
merge input, preserve source/provenance/authorship during manual merges, and
reconcile legacy conflict duplicates before adding the database key.

The complete command/output record and final hash are in the ignored task
report at `.superpowers/sdd/2026-09-05-01-corpus-foundation/task-2-report.md`.

## Final review-fix evidence

The final review wave added PostgreSQL regressions for concurrent shared venue
and author upserts, NULL and non-NULL document-version observations, legacy
duplicate-version migration reconciliation, and replay into a new trusted
staging directory after a missing or tampered old artifact. The committed-tree
regression selector passed:

```text
uv.exe run --project backend pytest backend/tests/integration/test_deduplication.py backend/tests/integration/test_migrations.py -k "concurrent_shared_venue_upsert_is_singleton or concurrent_shared_author_upsert_is_singleton or concurrent_version_observations_are_idempotent or replay_repoints_to_verified_artifact_in_new_staging_directory or version_constraint_migration_reconciles_legacy_null_duplicates" -q
7 passed, 29 deselected in 3.66s
```

Final validation passed with explicit PostgreSQL/Qdrant test services:

```text
uv.exe run --project backend pytest backend/tests/unit/test_identity.py backend/tests/integration/test_deduplication.py backend/tests/integration/test_migrations.py -q
44 passed in 15.59s

uv.exe run --project backend pytest backend/tests -q
56 passed in 17.89s

uv.exe run --project backend pytest backend/tests -m "not integration" -q
13 passed, 43 deselected in 0.19s

uv.exe run --project backend ruff check backend
All checks passed!

uv.exe run --project backend mypy backend/src
Success: no issues found in 11 source files
```

Alembic completed `upgrade head`, `downgrade 0000_foundation`, `upgrade
head`, and a second no-op `upgrade head`, including migration
`0002_identity_hardening -> 0003_version_hardening`. Fixture replay twice
returned `{"conflicts": 0, "loaded": 3, "quarantined": 0}` and final counts
were `(2, 3, 3, 0)` for papers, source records, paper versions, and conflicts.
The first full-suite attempt encountered stale disposable health collections;
after deleting those exact collections, the unchanged rerun passed all 56
tests.

The final code-fix commit is `0c01520606bea02b3b4706d9515be0ef45efa4a8`
(`fix: harden corpus shared upserts and version replay`). It contains the
conflict-safe shared upserts, PostgreSQL `NULLS NOT DISTINCT` version
constraint and reconciliation migration, verified artifact repointing, and
their integration regressions.
