# M1 ingestion evidence — 100-paper pilot and replay

Date: 2026-09-15
Task: P1.4, ingest an accepted-paper pilot through resumable jobs
Host: WSL2 Linux (kernel 6.18.33.2), Docker 29.4.3, `postgres:17.11-bookworm`
Run IDs: `ICLR-2024-20260915T032448Z-630a28af` (pilot),
`ICLR-2024-20260915T032829Z-3189815e` (replay)

## Source change, and why

The pilot was specified against OpenReview. Guest access to both the notes and
attachment routes returns `ChallengeRequiredError` (HTTP 403), re-confirmed on
2026-09-14 and unaffected by the User-Agent, so anonymous ingestion is not
possible and the adapter's login path needs an account that does not exist yet.

The pilot therefore runs from a **local mirror** of the papercli datasets, added
as `sources/papercli.py`:

| Artifact | Dataset | Revision |
|---|---|---|
| Metadata index | `GenAI4ELab/papercli-papers`, `browse/iclr/2024.parquet` | `90a1fbd3c355717092966debf5f7f69bdc6a1cf6` |
| PDFs | `GenAI4ELab/papercli-papers-iclr`, `pdfs/iclr/2024/**` | `050f8a4983fb3ff656e9cbaa32b8aeb3f561d2ef` |

Both were captured once under `${DATA_DIR}/sources/papercli/` (2,260 PDFs,
13.70 GB, every file's sha256 verified against its LFS pointer). Ingestion then
reads local files and **makes no network request at all**, which keeps Hugging
Face an offline artifact store rather than a request-time dependency and makes
the run replayable. `adopt_pdf` registers a mirrored file after re-checksumming
it, so `download.py`'s host, address and redirect restrictions stay untouched
for records that genuinely need fetching.

Provenance consequence: these PDFs are a third-party mirror, not OpenReview
originals. `paper_versions.source` records `papercli` and the dataset revision
rather than claiming an OpenReview retrieval.

## Membership

The index carries **no decision, track or acceptance field**. A venue-year
listing is not reliably the accepted set: ICLR 2024 holds exactly its 2,260
accepted papers, but the ICLR 2023 shard holds 3,792 rows against roughly 1,574
accepted. Membership is therefore asserted per manifest with
`membership_is_acceptance`, checked against the venue's published total. Without
that flag every record is emitted as `unknown` and `is_eligible` rejects the
whole listing, so the failure mode is an empty ingest rather than silently
admitting rejected submissions.

This is weaker than the OpenReview path, which maps each note's own venue label
onto the decision enum. It is recorded as a manifest-level assertion, not a
per-record observation.

## Pilot run

```text
corpus ingest --manifest configs/corpus.yaml --limit 100      2.0 s
worker run --worker-id pilot-1                                3 m 02 s, processed 300
```

Membership listing: 5 pages over the mirrored index, 2,260 eligible records,
100 selected. Selection is deterministic — eligible records sorted by
`source_item_id`, first 100 taken — which is a recorded sampling bias, not a
random sample.

| Measure | Value |
|---|---|
| Jobs | 300 succeeded (100 `resolve_record`, 100 `adopt_pdf`, 100 `parse_pdf`) |
| Retries | **0** — maximum attempt count across all jobs is 1 |
| Papers / identifiers / versions | 100 / 100 / 100 |
| Source records / field provenance rows | 100 / 700 |
| Parse outcomes | 100 `parsed`, no typed failures |
| Chunks | 4,713, all carrying a page span, all `pypdf-text-v2/fixed-window-v1` |
| Identity conflicts / quarantined | 0 / 0 |
| Checkpoint | `papercli` / `ICLR:2024:main` / `complete:pages=5:eligible=2260:selected=100` |

Chunk composition:

| Kind | Chunks | In default evidence |
|---|---:|---:|
| body | 2,117 | 2,117 |
| references | 1,352 | 0 |
| table | 367 | 367 |
| figure | 360 | 360 |
| appendix | 274 | 274 |
| abstract | 102 | 102 |
| front matter | 100 | 100 |
| acknowledgments | 41 | 41 |

Prose chunks (body, appendix, abstract) mean 322 tokens, median 393 against a
450 target — consistent with the post-fix parser audit. References are kept but
excluded from default evidence, as the chunking policy requires.

## Replay

The replay re-ran `corpus ingest` and `worker run` against the same mirror
revision. Because idempotency keys derive from source, item ID and source
revision, **no new job was enqueued and the worker processed 0**.

| Field | Pilot | Replay | Same |
|---|---|---|---|
| papers / identifiers / versions | 100 / 100 / 100 | 100 / 100 / 100 | yes |
| chunks | 4,713 | 4,713 | yes |
| canonical identity digest | `33f50719aca5bc65` | `33f50719aca5bc65` | **yes** |
| version digest | `4946e0d461a76539` | `4946e0d461a76539` | **yes** |
| chunk identity digest | `9a3adcb92433a845` | `9a3adcb92433a845` | **yes** |
| jobs by status | 300 succeeded | 300 succeeded | yes |
| conflicts / quarantined | 0 / 0 | 0 / 0 | yes |
| checkpoint cursor | `complete:pages=5:eligible=2260:selected=100` | identical | yes |
| checkpoint `run_id` | `…630a28af` | `…3189815e` | no — by design |

The only difference across the two runs is the checkpoint's `run_id`, which
advances so the latest run that confirmed the partition is recorded. Zero
duplicate papers, zero canonical ID churn, zero chunk identity churn.

## Coverage gaps

- **No authors.** The mirror index publishes none, so `authors` and
  `paper_authors` are empty for all 100 papers. Author coverage needs
  enrichment, and P1.2's title matching loses its author-compatibility signal
  for these records — two papers sharing a title would rely on year and venue
  alone to stay separate.
- **Redistribution unknown** for every record. The dataset card declares
  CC-BY-4.0, but per-paper rights are unverified, so P1.5's export filter will
  emit nothing from this corpus until rights are established per paper.
- **Acceptance asserted, not observed** — see Membership above.
- Enrichment was not exercised: `semantic_scholar` stays disabled, so the
  optional-enrichment path is covered only by the synthetic fixture suite.

## Limits

- One venue-year, 100 of 2,260 papers, on one machine with one worker process.
  Multi-worker lease contention is covered by the fixture suite, not by this run.
- Kill/restart resumption was not exercised against the live pilot; it is
  covered by `test_kill_after_metadata_write_is_resumable` and
  `test_kill_after_parse_write_is_resumable`.
- Timing is for a warm local mirror. It excludes the 17.5 minutes spent
  collecting the 13.70 GB of PDFs, which is a one-off per venue-year.
