# ML Research Copilot corpus — data card

Status: **being rebuilt.** The coverage figures below are a point-in-time reading taken
at 2026-09-16T09:35Z, when 12 of 42 venue-years had landed and chunking used the
`fixed-window-v2` policy. Since then the full plan completed at 85,729 papers, and on
2026-09-18 a rebuild began under the `paragraph-pack-v1` policy, re-parsing every paper
from its PDF into a fresh database. Treat the venue-year table as historical; the
authoritative live figures come from `GET /v1/corpus/coverage` or
`python -m copilot.cli corpus coverage`.

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

## Coverage as of 2026-09-16T09:35Z

8,499 papers across 12 venue-years; 8,463 carry an abstract and 8,494 have full
text. Four papers failed to parse.

| Venue | Year | Papers | Abstract | Full text | Failed | Published total | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACL | 2024 | 942 | 939 | 942 | 0 | unknown | — |
| EMNLP | 2023 | 1,047 | 1,046 | 1,046 | 0 | unknown | — |
| IJCAI | 2023 | 851 | 851 | 850 | 1 | unknown | — |
| JMLR | 2023 | 401 | 401 | 400 | 1 | unknown | — |
| JMLR | 2024 | 421 | 421 | 421 | 0 | unknown | — |
| JMLR | 2025 | 308 | 308 | 308 | 0 | unknown | — |
| NAACL | 2024 | 564 | 562 | 564 | 0 | unknown | — |
| NAACL | 2025 | 720 | 695 | 720 | 0 | unknown | — |
| WACV | 2023 | 639 | 637 | 639 | 0 | unknown | — |
| WACV | 2024 | 846 | 845 | 845 | 1 | unknown | — |
| WACV | 2025 | 929 | 927 | 929 | 0 | unknown | — |
| WACV | 2026 | 831 | 831 | 830 | 1 | unknown | — |

A published total is recorded only where it has been checked against the venue
— so far only ICLR 2024 (2,260), which this run has not yet reached. Every other
denominator is unknown, and **an unknown denominator reports a null coverage
percentage** rather than an implied 100%.

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
- **Four parse failures** so far — two oversized, one corrupt, one 0-byte file
  in the shard. Each keeps its PDF for retry; see `reports/m1-parse-failures.md`.
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
