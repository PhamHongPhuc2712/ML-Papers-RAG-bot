# ML Research Copilot corpus — data card

Status: **complete.** The rebuild under `paragraph-pack-v1` finished 2026-09-22; the
figures below are the finished corpus, read from `copilot_v2` on 2026-09-23. Verification
evidence is in `reports/m1-rebuild.md`. Live figures come from `GET /v1/corpus/coverage`
or `python -m copilot.cli corpus coverage`.

## What this corpus is

Accepted **main-conference** papers from AI venues published 2023–2026, stored
as canonical metadata plus section-aware full-text chunks in PostgreSQL. It
exists to serve retrieval, recommendation and evidence-grounded answering; it is
not a redistribution of the papers themselves.

## Where it comes from

| Artifact | Repository | Pinned revision |
|---|---|---|
| Metadata registry | `GenAI4ELab/papercli-papers`, `papers.parquet` | `90a1fbd3c355717092966debf5f7f69bdc6a1cf6` |
| PDFs, per venue | `GenAI4ELab/papercli-papers-<venue>` | resolved per venue at capture, recorded on every index record |

These are a third-party mirror, not the publishers' originals. Provenance
records `papercli` and the dataset revision rather than claiming retrieval from
OpenReview, the ACL Anthology or a proceedings site. The mirror is captured
once, offline; nothing in the serving path contacts Hugging Face.

## What counts as a member

Membership is read from each record's own venue label by `classify_track`, never
asserted for a listing. Across the 262 distinct labels the registry uses for
2023–2026, all 102,789 rows classify, and **85,732 are accepted main-conference
papers**. Excluded, with the count each accounts for:

| Excluded | Rows | Why |
|---|---:|---|
| Findings (ACL/EMNLP/NAACL) | 7,499 | A real venue, but not the main conference |
| Workshops | 4,883 | Spec §4 excludes workshops |
| Unaccepted submissions (`Submitted to ICLR 2023`) | 2,219 | Not accepted work |
| Co-located conferences (WMT, IWSLT, ArabicNLP, NLP4DH) | 1,118 | Separate conferences the registry files under ACL/EMNLP |
| Industry / demo / shared task / tutorial tracks | 1,338 | Not the main research track |

Main is matched positively, so a label this project has not seen classifies as
`other` and is excluded rather than silently admitted.

## Coverage as of 2026-09-23

**85,729 papers across all 42 venue-years, 3,426,221 chunks.** 85,694 carry an abstract
and 81,929 have full text. The `No text` column counts papers with no chunks.

| Venue | Year | Papers | Abstract | Full text | No text |
|---|---:|---:|---:|---:|---:|
| AAAI | 2023 | 2,023 | 2,023 | 2,021 | 2 |
| AAAI | 2024 | 2,864 | 2,864 | 2,864 | — |
| AAAI | 2025 | 3,485 | 3,485 | 3,485 | — |
| AAAI | 2026 | 4,920 | 4,920 | 4,920 | — |
| ACL | 2023 | 1,077 | 1,075 | 1,077 | — |
| ACL | 2024 | 942 | 939 | 942 | — |
| ACL | 2025 | 1,701 | 1,699 | 1,701 | — |
| CVPR | 2023 | 2,353 | 2,353 | 2,350 | 3 |
| CVPR | 2024 | 2,716 | 2,713 | 2,712 | 4 |
| CVPR | 2025 | 2,871 | 2,870 | 2,870 | 1 |
| CVPR | 2026 | 4,068 | 4,067 | 4,067 | 1 |
| ECCV | 2024 | 2,387 | 2,387 | 2,377 | 10 |
| EMNLP | 2023 | 1,047 | 1,046 | 1,047 | — |
| EMNLP | 2024 | 1,268 | 1,236 | 1,268 | — |
| EMNLP | 2025 | 1,809 | 1,808 | 1,809 | — |
| ICCV | 2023 | 2,156 | 2,156 | 2,155 | 1 |
| ICCV | 2025 | 2,701 | 2,701 | 2,699 | 2 |
| ICLR | 2023 | 1,573 | 1,573 | 1,571 | 2 |
| ICLR | 2024 | 2,260 | 2,260 | 2,259 | 1 |
| ICLR | 2025 | 3,702 | 3,702 | 3,702 | — |
| ICLR | 2026 | 5,351 | 5,351 | 3,615 | **incomplete** |
| ICML | 2023 | 1,828 | 1,828 | 1,828 | — |
| ICML | 2024 | 2,610 | 2,610 | 2,610 | — |
| ICML | 2025 | 3,257 | 3,257 | 3,257 | — |
| IJCAI | 2023 | 851 | 851 | 850 | 1 |
| IJCAI | 2024 | 1,048 | 1,048 | 1,048 | — |
| IJCAI | 2025 | 1,279 | 1,279 | 1,279 | — |
| Interspeech | 2023 | 1,141 | 1,141 | 1,141 | — |
| Interspeech | 2024 | 1,065 | 1,065 | 1,065 | — |
| Interspeech | 2025 | 1,179 | 1,179 | 1,175 | 4 |
| JMLR | 2023 | 401 | 401 | 400 | 1 |
| JMLR | 2024 | 421 | 421 | 421 | — |
| JMLR | 2025 | 308 | 308 | 308 | — |
| NAACL | 2024 | 564 | 562 | 564 | — |
| NAACL | 2025 | 720 | 695 | 720 | — |
| NeurIPS | 2023 | 3,218 | 3,218 | 3,216 | 2 |
| NeurIPS | 2024 | 4,034 | 4,034 | 4,034 | — |
| NeurIPS | 2025 | 5,286 | 5,286 | 3,259 | **incomplete** |
| WACV | 2023 | 639 | 637 | 639 | — |
| WACV | 2024 | 846 | 845 | 845 | 1 |
| WACV | 2025 | 929 | 927 | 929 | — |
| WACV | 2026 | 831 | 831 | 830 | 1 |

Two venue-years are marked **incomplete**: NeurIPS 2025 is missing 2,027 of 5,286 (38.3%)
and ICLR 2026 is missing 1,736 of 5,351 (32.4%), because the mirror published zero-byte
PDFs for those papers. This is a source gap, not a parse failure — see **Known gaps**.
Across the other 40 venue-years, 39 papers of 75,092 lack text (0.05%).

Published totals are recorded only where checked against the venue — so far ICLR 2024
(2,260), which matches. Every other denominator is unknown, and **an unknown denominator
reports a null coverage percentage** rather than an implied 100%.

## Rights, and what that means for exports

Every record currently carries `redistribution: unknown`. The papercli dataset
card declares CC-BY-4.0 for the collection, but per-paper rights are not
verified, and a PDF being publicly readable is not a right to republish it. So:

- **No full text is exported.** The 2026-09-16 snapshot carries 8,499 metadata
  rows and **0 of 513,335 chunks**; the manifest records them as withheld.
- Bibliographic metadata — titles, authors, venue, year, identifiers, checksums
  — is exported as factual reference data.
- Establishing per-paper rights would make full-text export possible; until
  then the export filter emits nothing, which is the intended failure mode.

## How papers are processed

pypdf text extraction (`pypdf-text-v2`) behind a swappable adapter, repaired for
NUL bytes, unpaired surrogates, ligatures, line-break hyphenation and repeated
page furniture. Sections are heading-aware with page spans; captions are bounded
blocks rather than section boundaries. Chunks are `paragraph-pack-v1`: whole
paragraphs packed to 800 tokens, closed when the next would pass 900, a paragraph
over 1,200 split on sentence boundaries, and the previous chunk's last two
sentences repeated as overlap — all measured in the pinned **BAAI/bge-m3**
tokenizer (`5617a9f61b028005a4858fdac845db406aefb181`) and never crossing a
section. The earlier build used `fixed-window-v2` (450 target, 600 cap, 60
overlap); it ended 47.2% of prose chunks mid-sentence against the current
policy's 14.4%. References are chunked but excluded from default evidence. Chunk IDs
are UUIDv5 over work ID, document checksum, parser and chunker revisions and the
section/chunk ordinals.

Measured quality: a 20-paper audit scored 95% usable text against abstracts as
ground truth. Chunks per paper vary widely by venue — 75 for ICLR, 121 for JMLR,
whose appendices are nearly as large as their bodies.

## Known gaps

- **No publication dates**, only years, so recency can only be ranked at
  year granularity.
- **No citation counts or reference edges** from the source. In-corpus citation
  edges are derivable from the stored reference chunks without further
  downloading; global counts would need an enrichment pass.
- **No DOI or arXiv identifiers.** Papers are keyed by the registry id plus, for
  OpenReview-hosted venues, the forum id.
- **3,802 papers have no full text (4.4%).** 3,759 of them are zero-byte PDFs *at the
  source*: `get_hf_file_metadata` reports `size: 0` and the index records the
  empty-string SHA-256. They are concentrated in NeurIPS 2025 and ICLR 2026, the two
  newest venue-years, where the mirror appears to have published placeholders before the
  proceedings were final. The recorded OpenReview URLs return 403, so these are not
  recoverable from here; they need a repopulated upstream shard or another source. The
  remaining 43 are ordinary parse failures — oversized, corrupt or not-a-PDF — each
  keeping its PDF for retry; see `reports/m1-parse-failures.md` and
  `reports/m1-rebuild.md`.
- Full-text quality is inferred from abstract fidelity; body text has not been
  measured against ground truth, and page-span correctness has not been
  inspected by hand.

## Privacy

This corpus contains published papers only. Private uploads and user history
never enter it: the export filter drops any row whose kind is `upload`
regardless of its rights, and that rule is covered by a test.

## Reproducing it

```powershell
uv run --project backend python -m copilot.cli corpus mirror --venue ICLR --year 2024
uv run --project backend python -m copilot.cli corpus run --only ICLR:2024
uv run --project backend python -m copilot.cli corpus export --run SNAPSHOT --out SNAPSHOT
uv run --project backend python -m copilot.cli corpus validate --manifest SNAPSHOT/manifest.json
```

Snapshots are immutable and stage under `${DATA_DIR}/exports/`. Validation
recomputes every shard checksum before a restore is allowed to touch a database.
