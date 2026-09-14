# M1 parser evidence and audit status

Date: 2026-09-10
Task: P1.3, parse documents into traceable sections and chunks
Commit: `9bbef4a`
Local-only report (`reports/` is gitignored by the owner's choice).

## Parser configuration

`configs/parsing.yaml` schema 1. Adapter `pypdf-text` (pypdf 6.10.0 plain
text per page), `parser_version=pypdf-text-v1`, quality label `low`,
`max_pdf_bytes=52428800`, encrypted input rejected. Chunker
`fixed-window-v1`: target 450, hard cap 600, overlap 60 (whitespace tokens in
CI; the pinned BGE-M3 tokenizer is injected once P2.2 pins the model
revision), references excluded from default evidence, UUIDv5 namespace
`9d2f5a6c-7b3e-4c1a-8e5d-2f6b9c1d3e47`.

Decision recorded: the plan names Docling as the first adapter. Docling pulls
in PyTorch and downloads models, which conflicts with the deterministic
offline CI rule and is a multi-gigabyte install on a zero-budget machine, so
pypdf is the first adapter behind the `PageAdapter` boundary. Docling is the
planned second adapter if the 20-paper audit below shows pypdf's text is not
usable; swapping it bumps `parser_version`, which intentionally changes every
chunk identifier.

## RED evidence

```text
pytest backend/tests/unit/test_chunks.py backend/tests/integration/test_parser.py -q
ERROR backend\tests\unit\test_chunks.py
ERROR backend\tests\integration\test_parser.py
Interrupted: 2 errors during collection  (copilot.corpus.chunk / parse missing)
```

## GREEN evidence

Part of the 119-test full run against `postgres:17.11-bookworm` in Docker;
Ruff and mypy strict clean.

| Acceptance case | Test |
|---|---|
| Plan regression: overlap never crosses sections | `test_overlap_never_crosses_sections` |
| Headings and page spans survive; kinds classified | `test_fixture_headings_pages_and_kinds_survive` |
| Long sections respect the window and cover every token | `test_long_sections_respect_target_and_cover_every_token` |
| Empty sections emit no chunks; ordinals contiguous | `test_empty_sections_emit_no_chunks_and_ordinals_are_contiguous` |
| Unknown pages remain null | `test_unknown_page_positions_remain_null` |
| Tables coherent up to the cap, else labeled fragments | `test_table_sections_stay_coherent_or_become_labeled_fragments` |
| References kept but excluded from default evidence | `test_references_are_kept_but_excluded_from_default_evidence`, fixture chunk test |
| Rerunning identical input keeps chunk IDs; revision change alters them | `test_chunk_ids_are_stable_and_depend_on_processing_revisions`, `test_parse_results_persist_with_stable_chunk_identity` |
| Encrypted, corrupt, non-PDF, missing, oversized, text-less inputs typed | `test_encrypted_pdf_is_typed`, `test_corrupt_non_pdf_and_missing_inputs_are_typed`, `test_oversized_pdf_is_rejected_before_parsing`, `test_pdf_without_extractable_text_is_a_typed_failure` |
| Parser version and checksum persisted on paper_versions | `test_parse_results_persist_with_stable_chunk_identity` |

## Fixture outcome

`data/fixtures/papers/fixture.pdf` (CC0, generated from `fixture.txt` by
`build_fixture.py`, 3 pages, 2,415 bytes): parsed → 7 sections (front
matter, Abstract, Introduction, Method, Table 1, Results, References) with
correct page spans; deterministic across reruns; checksum stable. Usable
text rate on the fixture: 1/1 — not evidence about real papers.

## Failure categories implemented

`missing`, `not_pdf`, `oversized`, `encrypted`, `corrupt`, `empty_text`.
Each is stored as the version's `parse_status`; `parsed` marks success. A
`corrupt` outcome is reproduced end to end by the P1.4 poisoned-PDF case.

## 20-paper audit — measured 2026-09-14

Performed against real ICLR 2024 PDFs collected from the HuggingFace mirror
`GenAI4ELab/papercli-papers-iclr` (2,260 files, 13.70 GB, sha256 verified
against the LFS pointers). Sample: 20 papers drawn with `random.seed(2024)`
from `iclr-2024-corpus.jsonl`, so the audit replays exactly.

Text fidelity is scored objectively rather than by eye. The corpus metadata
ships each paper's full abstract, so the extracted `abstract` section can be
compared against ground truth: broken reading order cannot reconstruct a real
abstract in sequence. Two measures are reported — token **recall** (is the
content present) and character **order** (is it in the right sequence,
`difflib` with `autojunk=False`; the default heuristic treats frequent
characters as junk and understates long abstracts badly).

### Text extraction — passes the G1 criterion

| Measure | As shipped | With line-break hyphenation rejoined |
|---|---|---|
| Abstract token recall, mean | 0.962 | **0.978** |
| Abstract token recall, min | 0.857 | 0.878 |
| Abstract order, mean | 0.976 | 0.973 |
| Abstract order, min | 0.753 | 0.666 |
| Papers with recall ≥ 0.95 | 15/20 | **17/20** |
| Papers with order ≥ 0.90 | 19/20 | 19/20 |

Rejoining hyphenation improves *content* recovery and leaves sequence alone:
recall rises and two more papers clear 0.95, while the order measure moves
within noise. It is worth doing for token quality, not for reading order.

Parse outcomes: **20/20 parsed**, no typed failures in the sample. Abstract
section detected 20/20; references section detected 20/20. Page counts 15–72,
sizes 0.8–31 MB.

**Usable-text rate: 19/20 = 95%**, against the G1 rule of ≥90%. "Usable" is
defined here as abstract recall ≥ 0.90 *and* abstract order ≥ 0.90, both
measured against ground truth. The single paper below the bar is
*Source-Free and Image-Only Unsupervised Domain Adaptation…* (recall 0.857,
order 0.753), which also carries the sample's heaviest hyphenation (171 broken
words) and its worst over-segmentation (175 sections). It is degraded, not
unreadable. The criterion is met and the parser is not revised. Docling remains unnecessary: the text
pypdf produces is faithful and correctly ordered, including on two-column
front matter. Reading order was specifically checked after an early metric
artifact suggested otherwise; with `autojunk=False` the apparent disorder
disappeared.

### Text-level defects (parser-owned, cheap)

| Defect | Incidence | Effect |
|---|---|---|
| Line-break hyphenation never rejoined (`estima-\ntion`) | **1,195 broken words across 20 papers**, mean 60/paper, max 171 | Produces junk tokens; degrades BM25 and embeddings directly |
| Ligatures left as single code points (ﬁ, ﬂ) | 88 across 20 papers | Splits words under alphanumeric tokenization |
| NUL bytes and unpaired UTF-16 surrogates | 32% / 2% of papers | Fixed 2026-09-14 in `_storable_text`; PostgreSQL rejected both |

### Structure defects (section logic, not pypdf)

These come from `sections_from_pages` and `_classify_heading`, not from text
extraction, and they are more damaging to retrieval than anything above.

| Defect | Incidence |
|---|---|
| Small-caps headings mangled (`I NTRODUCTION`, `M- PATTERN`) | 19/20 papers, **198 headings** |
| Figure/table captions promoted to sections that then absorb body prose | 19/20 papers, **181 sections holding 28% of all body+caption characters** |
| Running header leaking inline (`Published as a conference paper at ICLR …`) | 18/20 papers, **531 occurrences** |
| Over-segmentation | mean chunk **287 tokens against a 450 target**; **329 of 1,194 chunks under 50 tokens** |

Worst case in the sample: a 36-page paper split into 175 sections and 190
chunks averaging 72 tokens.

### Consequences

Body prose labelled `kind="figure"` is still returned as default evidence, so
a cited answer can report its source section as "Figure 5" when the text is
Methods prose. Undersized chunks weaken both the embedding and the
cross-encoder rerank that M2 depends on, and they inflate chunk counts and
index size. Fixing the four structure defects is P1.3-owned work that needs no
re-download; it changes chunk text and should bump `chunker_version`.

### Scope and limits

- One venue-year (ICLR 2024). Layout conventions differ at CVF and ACL venues; re-audit when P6.1 adds them.
- Text fidelity is measured against abstracts only, because that is the only ground truth the corpus carries. Body-text fidelity is inferred, not measured.
- Per-paper human inspection of page-span correctness and figure/table placement has **not** been performed; the measured evidence above is what supports the G1 decision.
- The 20 papers are listed with per-paper metrics in the audit output; the script and its seed reproduce the table exactly.
