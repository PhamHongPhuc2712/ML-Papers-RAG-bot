# Bibliography chunks searched as body text (2026-10-20)

## Results in plain words

- **69,019 chunks** in the live corpus were reference lists that the parser had labelled
  as body text. Search and the evidence step both read them. Almost all were in **ECCV 2024**
  (59,448 chunks; 2,279 papers) and **Interspeech 2023–2025** (9,491 chunks; 3,314 papers).
- So a query that named a cited paper ranked the paper citing it. "Find papers on contrastive
  learning of sentence embeddings" read three ECCV 3D-generation papers (ranks 3–5) through
  bibliography lines that cite WhitenedCSE. This happened in U4's walk and again in the redesign's walk.
- **Fixed**: the parser no longer makes these mistakes, and the stored chunks were relabelled in
  place (`references`, not evidence) in PostgreSQL and in the active release's chunk collection.
  No re-parse and no re-embedding: the text and vectors were right, only the label was wrong.
- On eight hand-written requests, the three 3D papers and four other papers found through
  bibliographies are gone. On the six paper-finding requests, my own reading of titles and passages
  (not a judged measure) counts 3 of 60 papers off topic, down from 7 of 60. The
  remaining 3 come from the reranker reading one passage per paper, U4's open follow-up.

## The two parser faults (`corpus/parse.py`, `_classify_heading`)

| Venue | Line in the PDF | Was | Now |
|---|---|---|---|
| Interspeech | `6. References` | a numbered heading named "References" with kind `body` | kind `references` (a numbered title is looked up among the named headings first) |
| ECCV (LNCS) | `2. Barron, J.T., Mildenhall, B., …: Mip-nerf` after "References" | a new `body` section per entry | not a heading; the entry stays in the references section (`_CITATION`: a surname, a comma, initials) |

Regression tests, which failed before the fix: `test_numbered_references_heading_is_a_references_section`,
`test_numbered_citations_do_not_open_sections` (`backend/tests/integration/test_parser.py`, offline).

`PARSER_VERSION` stays `pypdf-text-v2`. The live corpus and `configs/parsing.yaml` are v2, and
the version is part of every chunk id, so bumping it would change every id on any re-parse.
A re-parse under the fixed parser would also split ECCV's reference lists into fewer, larger
chunks than the stored ones; labels match, boundaries do not.

## The repair (`corpus relabel-references`)

`corpus/relabel.py` applies the same rule to stored section names: a references heading, or
a citation line taken for a heading. "Appendix C, I." is excluded, since the parser classifies
it as an appendix before the numbered rule. Unit tests are in `backend/tests/unit/test_relabel.py`.
Without `--apply` it only counts.

```bash
uv run --env-file .env --project backend python -m copilot.cli corpus relabel-references          # counts
uv run --env-file .env --project backend python -m copilot.cli corpus relabel-references --apply  # relabels
```

Order: the undo log, then Qdrant `set_payload` (`kind: references`, `evidence_default: false`,
batches of 1,000), then PostgreSQL in one transaction. An interrupted run selects the same chunks again.

Run 2026-10-20T03:31Z on `copilot_v2`, release `m2-20260924T095724Z`, collection
`dev_paper_chunks_m2-20260924T095724Z`:

| Venue-year | Chunks |
|---|---|
| ECCV 2024 | 59,448 |
| Interspeech 2023 / 2024 / 2025 | 3,117 / 3,002 / 3,372 |
| NeurIPS 2024 / 2025 | 39 / 6 |
| ICLR 2023 / 2025 | 23 / 11 |
| EMNLP 2025 | 1 |
| **Total** | **69,019** |

Checks afterwards:
- A second dry run finds 0 chunks.
- Of the 69,019 points, 0 still carry `evidence_default: true` (exact Qdrant count).
- The collection still holds 3,426,221 points.
- The undo log, with each chunk's previous `kind` and `evidence_default`, is
  `${DATA_DIR}/runs/relabel-references-20261020T033138Z.jsonl`.

## What this changes, and what it does not

- **The release is changed in place.** This was the developer's choice (2026-10-20) over building a new release,
  which would re-embed 3.4 M chunks. The index digest covers (chunk id, text sha256) only, so the
  recorded build digest still describes the collection. The relabel was not followed by `search validate-index`.
- **Earlier measurements on our corpus used the old labels**: S1 / S3 runs, U4, the walks.
  They are not re-run here. Open RAG Bench is unaffected: separate database and release.
- A new corpus export would differ from the snapshot the release was built from, in these
  chunks' `kind` and `evidence_default`.
- Not fixed: about 5,600 NeurIPS papers keep their reproducibility checklist ("Question: Do the
  main claims … Answer: [Yes]") as body text. It is boilerplate, not a bibliography, and was
  left alone.

## How the before-and-after lists were made

- The search was built with the API's own `default_search_service` (chunk candidates,
  title + chunk reranking), warmed until three searches in a row ran undegraded. No query
  degraded and no LLM was called.
- The GPU was 6–48% busy from other work, so rankings are recorded here but timings are not.
- `search pilot` was **not** used: it builds paper-collection branches, not the served chunk-candidate
  search, so its lists are not what the chat reads.
- The per-query lists are under the session's scratch directory, not in the repository.

## Second check: twenty new requests (2026-10-20, after the repair)

Seventeen paper-finding requests weighted toward vision and speech, where ECCV and Interspeech
papers come up, and three questions. Same served search, warmed first; no query degraded, GPU idle (0%).

- **No bibliography passage among the 200 read.** An automatic check (three or more years plus
  two or more citation marks in the passage) flagged 13. All 13 are introduction or related-work
  prose with inline citations, not reference lists.
- **Paper-finding, by my reading of titles and passages (not a judged measure):** 141 of 170 on topic,
  24 partly, 5 off topic. ECCV and Interspeech papers appear on their merits: speaker
  verification, speech emotion recognition and low-resource ASR lists are almost all Interspeech
  and on topic.
- **A separate corpus fault, fixed on 2026-10-21 (below):** 13 records are whole proceedings volumes, titled
  "Proceedings of the … Association for Computational Linguistics", whose text is the table of contents.
  A table of contents names many papers, so it matches many queries. One reached rank 2 for
  "knowledge distillation for large language models".
- Front matter is still the passage read for 13 of the 200: the earlier citation observation, unchanged.

## Proceedings volumes taken out of search (2026-10-21)

The registry admits 13 ACL, EMNLP and NAACL records (2023–2025) that are whole proceedings volumes,
not papers: titled "Proceedings of the …", no abstract, their text the volume's front matter and
table of contents. `corpus exclude-volumes` (`corpus/relabel.py` `proceedings_volume`: such a title
and no abstract; unit tests in `backend/tests/unit/test_relabel.py`) sets `evidence_default: false` on
their chunks, kind unchanged, in the same order as the relabel (undo log, Qdrant, PostgreSQL). The
paper records stay; only search and the evidence step stop reading them.

```bash
uv run --env-file .env --project backend python -m copilot.cli corpus exclude-volumes          # counts
uv run --env-file .env --project backend python -m copilot.cli corpus exclude-volumes --apply
```

Run 2026-10-21T06:20Z on the same release and collection: **1,788 chunks** (ACL 2023 / 2024 / 2025:
211 / 215 / 367; EMNLP 2023 / 2024 / 2025: 180 / 233 / 324; NAACL 2024 / 2025: 146 / 112). Undo log
`${DATA_DIR}/runs/exclude-volumes-20261021T062007Z.jsonl`. Checks afterwards:
- A second dry run finds 0; the relabel's dry run still finds 0.
- Of the 1,788 points, 0 still carry `evidence_default: true`; the collection holds 3,426,221 points.
- `search validate-index --release m2-20260924T095724Z`: valid, no problems (the check the
  2026-10-20 relabel had not run).
- The twenty requests above, run again through the served search (now reading title + abstract +
  passage, `reports/rerank-title-abstract-chunk.md`), none degraded: no volume in any top 10, and the
  knowledge-distillation request's ten papers are all on knowledge distillation for language models.

The proceedings volumes still appear in the paper (abstract) collection, which the chat's first stage
does not search; a future ingest should leave them out at the registry step.
