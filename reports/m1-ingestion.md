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

---

# Re-run 2026-09-16 — model-token chunking, registry records, observed membership

Task: A4, validating the three pipeline changes committed as `c82b2c2`
(model-token windows), `359dca7` (registry-backed records) and `f917341`
(membership from the venue label) before the multi-venue run.
Host: WSL2 Linux, 12 cores / 23 GB RAM, `postgres:17.11-bookworm`.
Run IDs: `ICLR-2024-20260916T062711Z-07408b04` (pilot),
`ICLR-2024-20260916T062855Z-82cc3fa8` (replay).
Database: `copilot_pilot_v2`, created empty for this run so the superseded
2026-09-15 pilot in `copilot` stays intact for comparison.

## What changed since the 2026-09-15 run

| | 2026-09-15 | 2026-09-16 |
|---|---|---|
| Chunk window | 450 **whitespace words** | 450 **BGE-M3 tokens** (`5617a9f`, pinned by sha256) |
| `chunker_version` | `fixed-window-v1` | `fixed-window-v2` |
| Records from | `browse/iclr/2024.parquet` (5 columns) | `papers.parquet` (11 columns) via `corpus mirror-index` |
| Authors | none — `paper_authors` empty | **561 distinct, 568 authorships, 0 papers without** |
| Acceptance | asserted per manifest | **observed** per record from the venue's label |
| Aliases per paper | `openreview` | `papercli` + `openreview` |

The deterministic sample is ordered by `source_item_id`, which is now the
registry id rather than the OpenReview forum id, so this is **a different 100
papers**: only 5 titles overlap with the 2026-09-15 sample. Counts below are
therefore not a like-for-like comparison with that report. Ordering by an
opaque registry id is closer to a random deterministic sample than ordering by
forum id, which correlates with submission time.

## Pilot run

```text
corpus ingest --manifest configs/corpus.yaml --limit 100     2 s
worker run  x4 (a4-1 … a4-4)                                69 s wall clock, 300 jobs
```

| Measure | Value |
|---|---|
| Jobs | 300 succeeded (100 `resolve_record`, 100 `adopt_pdf`, 100 `parse_pdf`) |
| Retries | **0** — maximum attempt across all jobs is 1 |
| Papers / identifiers / versions | 100 / **200** / 100 |
| Authors / authorships | 561 / 568 (max 34 on one paper) |
| Parse outcomes | 100 `parsed`, no typed failures |
| Chunks | **7,500** (75.0 per paper) |
| Acceptance recorded on source records | 100 `accepted`, observed from the label |
| Identity conflicts / quarantined | 0 / 0 |
| Papers without authors / without abstract | **0 / 0** |
| Checkpoint | `papercli` / `ICLR:2024:main` / `complete:pages=5:eligible=2260:selected=100` |

Chunk composition: body 3,216 · references 2,789 · table 473 · appendix 412 ·
figure 351 · abstract 105 · front matter 100 · acknowledgments 54.

## Chunk sizing — the defect this run exists to check

Every stored chunk was re-tokenized with the configured BGE-M3 tokenizer:

| | Value |
|---|---|
| Stored `token_count` | median **450**, mean 344, max 450 |
| Re-measured from stored text | median **450**, mean 345, max **467** |
| **Over the 600-token hard cap** | **0 of 7,500 (0.00%)** — was 51.4% on 2026-09-15 |

Re-measuring a stored chunk can exceed its recorded `token_count` by a few
tokens (max 467 against a 450 window) because a boundary piece re-encodes
differently in isolation. It stays well inside the cap and affects nothing
downstream.

Chunks per paper rose 47 → 75 as a direct consequence of counting real tokens.
Extrapolated to the 85,732 eligible main-conference papers that is roughly
**6.4 M chunks**, of which about 2.4 M would be reference lists — already
excluded from default evidence, and worth excluding from the first index too.

## Replay

Re-running `corpus ingest` and `worker run` against the same mirror revision
enqueued **no new job** and the worker processed **0**.

| Field | Pilot | Replay | Same |
|---|---|---|---|
| papers / identifiers / versions | 100 / 200 / 100 | identical | yes |
| authors / authorships | 561 / 568 | identical | yes |
| chunks | 7,500 | 7,500 | yes |
| canonical identity digest | `e7dfd1d07ebab0b0` | `e7dfd1d07ebab0b0` | **yes** |
| version digest | `3903a97d8d44db92` | `3903a97d8d44db92` | **yes** |
| chunk identity digest | `408534b00f8f6021` | `408534b00f8f6021` | **yes** |
| jobs by status | 300 succeeded | 300 succeeded | yes |
| checkpoint cursor | `complete:pages=5:…` | identical | yes |

## Coverage gaps

Closed since 2026-09-15: **authors** (0 papers without) and **acceptance**,
which is now read from each record's own label rather than asserted per
manifest.

Still open, all recorded rather than worked around:

- **Redistribution unknown** for every record. The dataset card declares
  CC-BY-4.0 but per-paper rights are unverified, so P1.5's export filter will
  emit no full text from this corpus until rights are established per paper.
- **No publication date**, only a year, so P3.3's freshness component is
  year-granular.
- **No citation counts**, so §8's popularity component has no signal. In-corpus
  citation edges are derivable from the stored reference chunks without any
  further download.
- Enrichment not exercised: `semantic_scholar` stays disabled.

## Limits

One venue-year, 100 of 2,260 papers, one machine. Four worker processes did
contend for leases here, which the 2026-09-15 single-worker run did not
exercise; kill/restart resumption is still covered by the fixture suite rather
than by this run.

---

# Smoke run 2026-09-16 — JMLR 2025, the full venue-year loop

Task: B3, proving `corpus run`'s mirror → ingest → work → verify → sweep loop on
the smallest venue-year before committing to all 42.
Campaign `corpus-20260916T065119Z-4e5836a9`, run `JMLR-2025-20260916T065314Z-ffa7921a`,
state under `${DATA_DIR}/runs/corpus-20260916T065119Z-4e5836a9/`.

```text
corpus run --only JMLR:2025
```

| Stage | Result |
|---|---|
| Mirror | 308 downloaded, 0 already present, **0 failed**, 0 missing; 709 MB in ~115 s (~6.2 MB/s) |
| Shard revision | `GenAI4ELab/papercli-papers-jmlr` at `a5b19b5df86f3f2f884f3f1b3a313304a6f113a5`, recorded on every index record |
| Ingest + parse | 924 jobs, **0 failed, 0 unfinished**, 170 s across 9 workers (91–115 jobs each) |
| Stored | 308 papers, 1,159 authorships, **37,433 chunks**, 308/308 `parsed`, 0 papers without authors |
| Chunk sizing | median **450** tokens, max 450 — the cap holds on a second venue |
| **Sweep** | **308 deleted, 0 kept, 0 missing, 709 MB freed**; no PDF remains under `pdfs/jmlr/` |

Nothing was kept because every paper parsed. The keep-on-failure path is covered
by the unit suite rather than by this run, which had no failures to exercise it.

Chunk composition: body 16,161 · appendix 14,705 · references 2,978 · figure
1,744 · table 722 · abstract 534 · front matter 312 · acknowledgments 277.

## Two measurements that change the plan's arithmetic

**Chunks per paper vary by venue far more than expected.** JMLR gives **121.5**
against ICLR 2024's 75.0 — journal articles carry appendices nearly as large as
their bodies (14,705 appendix chunks against 16,161 body). A single
chunks-per-paper figure cannot size the corpus; the earlier 6.4 M projection was
built from the ICLR rate alone and should be treated as a floor until more
venues are measured.

**Download throughput is lower on small files.** 6.2 MB/s here against the
13 MB/s measured on the ICLR shard, because JMLR's files average 2.3 MB and
per-file overhead dominates with 8 concurrent downloads. At 6.2 MB/s the ~450 GB
corpus would take about 20 hours of downloading rather than 10, which would make
the whole run roughly 20–22 hours rather than 14–16. Raising
`defaults.download_workers` is the obvious lever and is worth measuring on the
next venue-year before the long run starts.

## What the smoke proves

The loop is resumable and self-describing: the campaign state file records the
completed venue-year with its mirror counts, run ID, worker outcome and sweep
result, and a re-run skips it. Peak disk for this venue-year was 709 MB, and it
returned to zero afterwards.
