# Data card — ML Research Copilot corpus

Version: snapshot schema 1. Last updated: 2026-09-12.

This card describes what the corpus actually contains today, on one developer
machine. It is a 100-paper pilot, not a research dataset release, and nothing
here has been published anywhere.

## What this is

A replayable snapshot of accepted machine-learning conference papers: canonical
metadata, document versions, and section-aware full-text chunks, exported as
Parquet shards with a checksummed manifest.

| Field | Value |
|---|---|
| Papers | 100 |
| Venue-years | ICLR 2024, main conference |
| Chunks | 4,470 (3,603 in default evidence; references excluded) |
| Authors | 517 |
| Document versions | 100 |
| Snapshot size | 3.0 MB (zstd Parquet) |
| Schema version | 1 |
| Parser revision | `pypdf-text-v2` |
| Chunker revision | `fixed-window-v1` (450 target / 600 cap / 60 overlap) |

## Provenance

Membership comes from OpenReview API v2,
`content.venueid=ICLR.cc/2024/Conference`, read on 2026-09-12 with an
authenticated free account. 2,260 accepted notes were found across 3 listing
pages. Every note's venue label is mapped onto a decision enum and checked
explicitly; the query is not treated as proof of acceptance.

PDFs come from `https://api2.openreview.net/attachment?name=pdf&id=<note-id>`.

## Sampling — read this before using the data

The 100 papers are **not a random sample**. Eligible records are sorted by
OpenReview note ID and the first 100 taken, so a replay selects the same
papers. Note IDs are not random with respect to submission time or content, so
this sample supports reproducibility and **no distributional claim whatsoever**
about ICLR 2024. Present in the sample: 4 oral, 9 spotlight, 87 poster.

The 2,260 figure is what this provider returned on this date. It has not been
reconciled against the official ICLR proceedings, so it is not an authoritative
denominator, and every coverage percentage in the API and the manifest is
reported as null rather than computed against it.

## Coverage

| Metric | Count | Coverage |
|---|---|---|
| Abstract present | 100 / 100 | null — no authoritative denominator |
| PDF downloaded | 100 / 100 | null |
| Full text parsed and chunked | 100 / 100 | null |
| Identity conflicts | 0 | — |
| Quarantined records | 0 | — |

Counts are exact. Percentages are deliberately null: a count says nothing about
what fraction of the venue-year it represents until the denominator is
established.

## Redistribution and licensing

**No part of this corpus is currently publishable.** All 100 document versions
carry `redistribution = unknown` with no license label, because the OpenReview
adapter does not yet read per-paper licence metadata. Under the export policy
that means:

| Export mode | Metadata rows | Full-text rows |
|---|---|---|
| Local snapshot (restore, portability) | 100 | 4,470 |
| Public export (`--public-only`) | 0 | 0 |

This is the policy working, not a failure. Public availability of a PDF does
not establish redistribution rights (spec §4), so rows default to withheld.
Determining actual licences — ICLR papers are commonly CC BY 4.0, but this has
not been verified per paper — is the prerequisite for any publication, and
`corpus publish` is disabled in `configs/artifacts.yaml` until then.

Private user content never enters any export. The export refuses outright if a
row carries a private field.

## Known quality limitations

- **Ligature and math glyph loss.** The pypdf text adapter maps some glyphs onto
  C0 control codepoints, which cannot be stored. They are stripped, so
  `identification` can appear as `identication`. Measured: 69 of 100 documents
  affected, 4,098 of 7,646,599 characters removed (0.054%), concentrated inside
  affected words. Exact-match lexical search on those specific terms will miss.
- **Parse quality is unaudited.** The 20-paper manual audit of usable text,
  multi-column reading order and page-span correctness has not been performed.
  No claim is made about usable-text rate.
- **Text quality is labelled low.** The adapter extracts plain text per page
  with no layout model.
- **One namespace only.** Every identifier is `openreview`; there are no DOI or
  arXiv aliases, so cross-provider identity resolution is untested on real data.
- **No citation edges.** The `edges` shard exists and is empty until P6.3.
- **One venue-year.** Nothing here supports a claim about ICML, NeurIPS, other
  years, or machine learning publishing generally.

## Reproducing it

```powershell
uv run --project backend python -m copilot.cli corpus ingest --manifest configs/corpus.yaml --limit 100
uv run --project backend python -m copilot.cli worker run --worker-id pilot-1
uv run --project backend python -m copilot.cli corpus export --run <RUN_ID> --out pilot
uv run --project backend python -m copilot.cli corpus validate --manifest pilot
```

Requires `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` in `.env`; guest API
access is refused with a browser challenge. Snapshots stage under
`${DATA_DIR}/exports/`, outside the repository.
