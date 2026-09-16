# M1 corpus evidence — export, coverage and restore

Date: 2026-09-16
Task: P1.5, export immutable corpus snapshots and publish coverage
Host: WSL2 Linux, 12 cores / 23 GB RAM, `postgres:17.11-bookworm`
Snapshot: `m1-20260916T093334Z` under `${DATA_DIR}/exports/`

Taken while the venue run was still in progress, so it is a point-in-time
snapshot of 8,499 papers across 12 venue-years, not the finished corpus.

## Commands and outcomes

```text
corpus export --run m1-20260916T093334Z --out m1-20260916T093334Z
  -> papers 8,499 | chunks_total 513,335 | chunks_exported 0 | withheld 513,335
corpus validate --manifest m1-20260916T093334Z/manifest.json
  -> valid: true, shards papers.parquet + fulltext.parquet
corpus restore --manifest m1-20260916T093334Z/manifest.json --database-url .../copilot_restore_check
  -> papers 8,499 | identifiers 8,499
corpus coverage
  -> 12 venue-years, 8,499 papers, 8,463 abstracts, 8,494 full text, 4 failed
```

Artifacts: `papers.parquet` 7.96 MB, `fulltext.parquet` 329 bytes (empty by
policy), `manifest.json` 814 bytes.

## Restore into a clean database

A fresh `copilot_restore_check` database was created empty and restored from the
validated manifest. Identity is preserved, which is what makes this a restore
rather than a re-ingest:

| Digest | Source | Restored | |
|---|---|---|---|
| paper IDs | `8c22d782f3c92b7a` | `8c22d782f3c92b7a` | **match** |
| identifiers | `a785bd9f8355378b` | `a785bd9f8355378b` | **match** |

## Rights outcome — the export is deliberately empty of text

Every record carries `redistribution: unknown`, so **0 of 513,335 chunks were
exported** and the manifest records all of them as withheld. That is the
intended behaviour, not a bug: the dataset card declares CC-BY-4.0 for the
collection but per-paper rights are unverified, and public availability is not a
redistribution right. Bibliographic metadata is exported as factual reference
data. Establishing per-paper rights is what would unlock the full-text shard.

`corpus publish` is deliberately **not implemented**. With every record's rights
unverified there is nothing this corpus may legitimately publish, and an upload
path that could ship unlicensed text is not worth having before that changes.

## Acceptance cases

| Case | Evidence |
|---|---|
| Private or unlicensed text is not exported | `test_private_or_unlicensed_text_is_not_exported`, `test_upload_rows_never_leave_even_when_marked_allowed`, and the live export's 0 exported chunks |
| Checksum tampering fails | `test_checksum_tampering_fails`, `test_tampered_snapshot_is_refused_before_it_touches_the_database` |
| Schema mismatch fails before mutation | `test_schema_mismatch_fails_before_anything_is_read` — the version is checked before any shard is opened |
| Missing abstracts and missing full text counted separately | `test_coverage_counts_missing_abstracts_and_missing_fulltext_separately`; live totals 8,463 abstracts vs 8,494 full text |
| Unknown denominator never becomes 100% | `test_coverage_percentage_needs_a_known_denominator`; every live venue-year reports a null percentage |
| Restore reproduces counts and identities | `test_restore_reproduces_counts_and_identities` plus the digest match above |
| Snapshots are immutable | `test_snapshots_are_immutable` — a second export into the same directory is refused |
| Restore refuses a populated database | `test_restore_refuses_a_populated_database` |
| No active release before a validated index pair | `test_activation_is_refused_when_the_index_pair_does_not_validate`, `test_staging_does_not_change_what_requests_read` |
| Coverage endpoint is public and leaks nothing | `test_coverage_is_public_and_leaks_no_configuration` |
| HF failure does not affect serving | Nothing in the serving path contacts Hugging Face; capture is offline and `corpus publish` is unimplemented |

## Suites

```text
pytest backend/tests -q                 200 passed
ruff check backend                      All checks passed
mypy --config-file backend/pyproject.toml backend/src   Success: 31 source files
```

## Limits

- The snapshot was taken mid-run, so it describes 12 of 42 venue-years. A final
  snapshot is owed when the run finishes.
- Restore covers what the snapshot carries — metadata and identifiers. Chunks
  are not restorable while rights keep them out of the export, so a full
  rebuild still means re-parsing from the mirror.
- Only ICLR 2024 has a verified published total, and the run has not reached it
  yet, so every coverage percentage currently reads null.
