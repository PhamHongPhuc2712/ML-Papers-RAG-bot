# M1 identity evidence

Date: 2026-09-10
Task: P1.2, canonical paper identity and source provenance
Commits: `35c0da1` (task), `f652b9a` (integration evidence)

Adapted from `04f6e3e`, which had passed three review rounds before the
reset. Changes for the revised design: the two later hardening migrations
were folded into `0001_corpus` (document versions unique with `NULLS NOT
DISTINCT`; one review conflict per source record and reason), and the fixture
loader's raw-artifact directory defaults to `<DATA_DIR>/sources`. Services
and versions as in `m1-foundation.md`. Local-only report.

## RED evidence

```text
backend/.venv/Scripts/python -m pytest backend/tests/unit/test_identity.py backend/tests/integration/test_deduplication.py -q
ImportError while loading conftest '...backend\tests\conftest.py'.
E   ModuleNotFoundError: No module named 'copilot'
```

## GREEN evidence

Part of the 67-test full run (`67 passed in 56.12s`) against
`postgres:17.11-bookworm` and `qdrant/qdrant:v1.19.1` on the
`C:\ml-copilot-data\test` bind mounts; Ruff and mypy strict clean. Unit
identity tests are in the 22-test offline set.

## Acceptance cases named by the plan

| Case | Test |
|---|---|
| Two providers for one DOI resolve to one work | `test_two_providers_sharing_doi_resolve_to_one_work` |
| Concurrent duplicate upserts produce one ID | `test_concurrent_doi_resolution_converges_without_orphan_papers`, plus concurrent venue/author/version/conflict/quarantine cases |
| arXiv versions remain separate document versions | `test_arxiv_versions_share_work_but_keep_distinct_documents` |
| Contradictory strong IDs create a review conflict | `test_contradictory_strong_aliases_create_review_conflict`, `test_strong_id_collision_with_incompatible_metadata_creates_conflict` |
| Unicode title variants keep original spelling | `test_unicode_title_variants_preserve_each_original_spelling` |
| Identical titles with disjoint authors stay separate | `test_identical_titles_with_disjoint_authors_remain_separate` |
| Missing DOI valid; invalid DOI quarantined | `test_missing_doi_is_valid`, `test_invalid_doi_is_quarantined_when_caller_commits_review` |
| Manual merge preserves versions and records a redirect | `test_manual_merge_preserves_versions_and_records_redirect` and two rejection cases |
| Raw artifact checksum and field provenance stable | `test_raw_artifact_checksum_and_field_provenance_are_stable` |
| Fixture loader replay idempotent | `test_fixture_loader_replay_is_idempotent` |

## Migration replay

`test_migrations_replay_up_down_up` downgraded to `base` and back to `head`
on the isolated database and confirmed all eleven corpus tables plus the
foundation tables. `test_corpus_constraints_are_created_hardened` read
`pg_indexes` / `pg_constraint` and confirmed
`uq_paper_versions_source_revision_content_version` carries `NULLS NOT
DISTINCT` and `uq_identity_conflicts_source_reason` exists — i.e. the folded
constraints are created by `0001_corpus` on a fresh schema, so the legacy
reconciliation migrations were not carried over.

## Limits

- The concurrency tests exercise thread-level races against one PostgreSQL; multi-process behaviour is untested.
- The fixture file `data/fixtures/metadata.jsonl` is a three-record seed, not a corpus sample.
