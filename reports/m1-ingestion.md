# M1 ingestion evidence — accepted-paper pilot

Date: 2026-09-12
Task: P1.4, ingest an accepted-paper pilot through resumable jobs
Commit: `d3fad84` (task), plus the parser fix recorded below
Host: Windows 11 Home 10.0.26200, Docker Desktop 4.90.0, `postgres:17.11-bookworm`

This is the first run of the pipeline against real papers. Everything below is
measured on this host; nothing is projected.

## Provider membership query

OpenReview API v2, `GET https://api2.openreview.net/notes` with
`content.venueid=ICLR.cc/2024/Conference`, `limit=1000`, offset paging.
Authenticated with a free account's bearer token from `POST /login`; guest
access returns HTTP 403 `ChallengeRequiredError` on both the notes and
attachment routes. PDFs from
`https://api2.openreview.net/attachment?name=pdf&id=<note-id>`.

Eligibility is checked explicitly per record rather than assumed from the
query: every note's venue label is mapped onto the decision enum, and
`is_eligible` requires venue, year, track, `decision == accepted` and
`withdrawn == false`.

| Quantity | Value |
|---|---|
| Listing pages fetched | 3 |
| Eligible accepted notes found | 2,260 |
| Selected for the pilot | 100 |
| Selection rule | sort by `source_item_id`, take first 100 |

The sample is deterministic and **biased on purpose**: sorting by OpenReview
note ID and truncating is not a random sample of ICLR 2024, and no claim about
the venue's distribution should be drawn from it. `configs/corpus.yaml` records
the rule. The authoritative ICLR 2024 accepted count was not independently
confirmed against the official proceedings, so 2,260 is this provider's answer
on this date, not a verified denominator.

## Coverage of the 100 selected papers

| Metric | Count |
|---|---|
| Papers resolved to canonical IDs | 100 |
| Abstract present | 100 |
| PDF URL present | 100 |
| PDF downloaded | 100 |
| Full text parsed and chunked | 100 |
| Chunks persisted | 4,470 |
| Chunks in default evidence (references excluded) | 3,603 |
| Distinct authors | 517 |
| Identity conflicts | 0 |
| Quarantined records | 0 |

Acceptance presentation types: 4 oral, 9 spotlight, 87 poster — all three
eligible types present, as spec §4 requires.

Chunk shape under `fixed-window-v1` (450 target / 600 cap / 60 overlap,
whitespace tokenizer): mean 302 tokens, maximum 598, so the hard cap held.
Section kinds classified: body 2,087, references 867, figure 535, table 526,
appendix 212, abstract 102, front matter 100, acknowledgments 41.

## Retry counts

| Job kind | Jobs | Total attempts | Retries |
|---|---|---|---|
| `resolve_record` | 100 | 100 | 0 |
| `download_pdf` | 100 | 100 | 0 |
| `parse_pdf` (after fix) | 100 | 100 | 0 |

No provider throttling was observed: zero 429 or 503 responses across roughly
104 outbound requests (3 listing pages, 1 login, 100 attachment downloads). The
backoff and `Retry-After` paths therefore remain covered only by the synthetic
fixture suite, not by live evidence.

## Defect found by the live run

41 of 100 parses initially failed with `handler_exception` (39 terminally, 2 in
`retry_wait`); 59 succeeded:

```text
sqlalchemy.exc.DataError: (psycopg.DataError)
PostgreSQL text fields cannot contain NUL (0x00) bytes
```

All 100 PDFs **parsed** successfully; the failure was at persistence. pypdf maps
LaTeX ligature and math glyphs onto C0 control codepoints when a font carries a
custom encoding. Observed across the corpus: `0x00`–`0x08`, `0x10`–`0x15`,
`0x18`, `0x1a`, `0x1b`, `0x7f`. Exactly 41 of 100 documents contained NUL, which
PostgreSQL `text` rejects outright — the same 41 that failed, so the defect is
fully accounted for. The other control codes stored without error but are junk
in evidence text, so the fix removes them too.

The repository's fixture PDF is generated with clean fonts, so the 119-test
suite could not have caught this. It required real papers.

**Fix:** `sanitize_text` at the `sections_from_pages` boundary — the point every
`PageAdapter` output flows through, so a replacement adapter inherits it. Tab is
preserved; `str.splitlines` already consumes vertical tab and form feed.
`PARSER_VERSION` bumped `pypdf-text-v1` → `pypdf-text-v2` in both `parse.py` and
`configs/parsing.yaml`, because changing text extraction is a parser change and
chunk identity must follow it. All 100 documents were re-driven at v2 so the
corpus is a single revision rather than a 59/41 mix. Regression test:
`test_control_characters_from_font_encodings_are_stripped`.

**Cost of the fix, stated plainly:** stripping keeps a word whole instead of
splitting it, but it does not recover the glyph — `identi<0x01>cation` becomes
`identication`, losing the "fi". Measured across the 100 papers:

| Quantity | Value |
|---|---|
| Documents with at least one stripped character | 69 / 100 |
| Characters stripped | 4,098 |
| Characters retained | 7,646,599 |
| Proportion lost | 0.054% |

The aggregate proportion is small, but the loss is concentrated inside words
containing ligatures, so exact-match lexical search on those specific terms will
miss. This is parser-quality evidence for the P1.3 audit and a concrete argument
for evaluating a font-aware parser; it is not something the pypdf text adapter
can recover.

39 `parse_pdf` jobs remain in the `failed` state. They are the superseded v1
keys and are retained deliberately as the durable record of this defect; their
work was redone under the v2 keys.

## Replay

Second run of `corpus ingest --manifest configs/corpus.yaml --limit 100`
(run `ICLR-2024-20260912T141459Z-631a7548`) after the first
(`ICLR-2024-20260912T134001Z-25d98688`).

| Check | Result |
|---|---|
| Jobs processed by the replay worker | 0 |
| Canonical paper IDs | byte-identical, 100/100 (`diff` empty) |
| Row counts across 8 tables | identical |
| Checkpoint cursor | identical: `complete:pages=3:eligible=2260:selected=100` |
| `run_id` on the checkpoint | advanced, as designed |

Zero jobs processed means every idempotency key matched, so the replay created
no duplicate work. This satisfies G1's "repeat run changes no canonical IDs".

## Limits — what this run does not establish

- **No strong IDs in the corpus.** All 100 identifiers are in the `openreview`
  namespace; the arXiv and Semantic Scholar sources are disabled in the
  manifest. G1's "zero duplicate strong IDs" therefore passes vacuously — there
  are no DOI or arXiv aliases to collide. The cross-provider identity logic in
  `dedupe.py` remains covered only by fixtures, and enabling a second source is
  the real test of it.
- **The 20-paper manual audit is still outstanding.** Usable-text rate,
  multi-column reading order and page-span correctness have not been checked by
  a human. Until they are, no claim is made about pypdf's usable-text rate on
  real ICLR papers and G1's "≥90% audited usable text" is unmet. The PDFs are
  now on disk under `${DATA_DIR}/sources/pdfs/`, so the audit is unblocked.
- **Throttling behaviour is untested live** (see retry counts above).
- One venue-year only; no claim about ICML, NeurIPS or other years.
