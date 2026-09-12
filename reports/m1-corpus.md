# M1 corpus snapshot evidence

Date: 2026-09-12
Task: P1.5, export immutable corpus snapshots and publish coverage
Host: Windows 11 Home 10.0.26200, Docker Desktop 4.90.0, `postgres:17.11-bookworm`

All figures measured on this host against the P1.4 pilot corpus.

## Dependency added

`pyarrow==21.0.0` for Parquet, pinned and locked. First dependency addition
since the foundation.

## Snapshot produced

`corpus export --run ICLR-2024-20260912T134001Z-25d98688 --out pilot`

| Artifact | Rows | Bytes |
|---|---|---|
| `metadata-0000.parquet` | 100 | 83,108 |
| `fulltext-0000.parquet` | 4,470 | 3,043,838 |
| `edges-0000.parquet` | 0 | 369 |
| `manifest.json` | — | 1,672 |

Total 3.0 MB at `C:\ml-copilot-data\exports\pilot`, zstd-compressed, from
7,646,599 characters of extracted text. Manifest checksum
`e7fc563bccae38b0571d4ef9651152ba10dd838dc1d4a48381aa59a8dbe3a882`.

One shard per kind: the 128 MiB shard target does not engage at pilot scale.
The `edges` shard is present and empty rather than absent, so the artifact set
has a stable shape before P6.3 ingests citations.

## Public export withholds everything, correctly

`corpus export --public-only` on the same corpus:

| Mode | metadata | fulltext | withheld |
|---|---|---|---|
| Local snapshot | 100 | 4,470 | — |
| Public export | **0** | **0** | 100 metadata, 4,470 fulltext |

Every document version carries `redistribution = unknown` because the
OpenReview adapter does not read per-paper licence metadata. The policy
withholds by default, so the entire pilot is unpublishable until licences are
established. This is the designed behaviour under spec §4 ("public availability
of a PDF does not by itself establish redistribution rights"), not a defect, and
it is why `corpus publish` stays disabled in `configs/artifacts.yaml`.

## Validation and restore

```text
corpus validate --manifest pilot
{"artifacts": 3, "counts": {"edges": 0, "fulltext": 4470, "metadata": 100},
 "run_id": "ICLR-2024-...", "schema_version": 1, "valid": true}

corpus restore --manifest pilot --database-url .../staging_copilot
{"authors": 517, "chunks": 4470, "identifiers": 100, "manifest_chunks": 4470,
 "manifest_papers": 100, "papers": 100, "versions": 100}
```

Restored into a freshly created, empty `staging_copilot` database. Comparing
canonical paper IDs between the source database and the restored one:

```text
diff ids_copilot.txt ids_staging_copilot.txt
IDENTICAL (100 papers)
```

Identities survive the round trip, not merely the counts. Row counts in the
restored database: 100 papers, 100 versions, 100 identifiers, 4,470 chunks, 517
authors — all equal to the source.

## Acceptance cases

| Case | Test |
|---|---|
| Plan regression: private or unlicensed text is not exported | `test_private_or_unlicensed_text_is_not_exported` |
| Checksum tampering fails | `test_checksum_tampering_is_detected`, `test_manifest_tampering_is_detected` |
| Schema mismatch fails before mutation | `test_schema_mismatch_fails_before_any_mutation` |
| Export excludes private/unknown-rights text | `test_public_export_withholds_text_without_established_rights`, `test_export_refuses_rows_carrying_private_fields` |
| Missing abstracts and missing PDFs get separate counts | `test_snapshot_exports_validates_and_counts_missing_fields` |
| Restore reproduces counts and identities | `test_restore_into_an_empty_database_reproduces_counts_and_identities` |
| Snapshots are immutable; prior ones preserved | `test_snapshots_are_immutable_and_prior_ones_survive` |
| No active release before a validated index pair | `test_no_release_becomes_active_before_its_indexes_validate` |
| Coverage null denominator never becomes 100% | `test_unknown_denominator_yields_null_coverage_never_one_hundred`, `test_coverage_endpoint_reports_null_percentages_without_a_denominator` |
| HF publication failure does not affect serving | `corpus publish` is a separate CLI mode, disabled in config, touching no serving path; no test exercises a live publication |

## Gates

```text
pytest backend/tests -q                     140 passed in 72.55s
ruff check backend                          All checks passed!
mypy src (from backend/, strict)            4 pre-existing errors, none in P1.5 files
```

The four strict-mode errors are in `contracts.py`, `jobs/queue.py`,
`corpus/dedupe.py` and `app.py` and predate this task. They surfaced because
strict mode had never actually been applied — see the note below.

## Defect found: mypy strict was never running

The documented command `uv run --project backend mypy backend/src` runs from the
repository root, which contains no mypy configuration. `mypy -v` reports
`Config File: Default`, so `backend/pyproject.toml`'s `strict = true` was never
loaded. Every previous report's claim of "mypy strict clean" describes a
default-mode run.

Running mypy with its real configuration surfaces 4 errors, all pre-existing.
Fixed in a separate commit rather than folded into P1.5, since it concerns the
CI workflow, `CLAUDE.md` and `backend/README.md` — files this task does not own.

## Limits

- The `edges` artifact is structurally present but empty; nothing here evidences
  citation-graph export.
- Shard splitting at the 128 MiB target is implemented but unexercised: the
  pilot produces one shard per kind, so the multi-shard path has no live
  evidence.
- `corpus publish` has no implementation behind its authorization check. It
  refuses, by design, and no publication has been attempted.
- Release activation is proven only against an injected validator. The real
  index validation arrives with P2.2, and the default validator refuses until
  then.
- `staging_copilot` was left in place in the dev PostgreSQL instance as restore
  evidence; it is disposable.
