# M1 parser evidence and audit

Date: 2026-09-10, audited 2026-09-12
Task: P1.3, parse documents into traceable sections and chunks
Commits: `9bbef4a` (task), `38a32c4` (control characters), plus the audit revision below

## Parser configuration

`configs/parsing.yaml` schema 1. Adapter `pypdf-text` (pypdf 6.10.0 plain text per
page), `parser_version=pypdf-text-v3`, quality label `low`,
`max_pdf_bytes=52428800`, encrypted input rejected. Chunker `fixed-window-v1`:
target 450, hard cap 600, overlap 60, references excluded from default evidence,
UUIDv5 namespace `9d2f5a6c-7b3e-4c1a-8e5d-2f6b9c1d3e47`.

Decision recorded: the plan names Docling as the first adapter. Docling pulls in
PyTorch and downloads models, which conflicts with deterministic offline CI on a
zero-budget machine, so pypdf is first behind the `PageAdapter` boundary.

## Scope of this audit — read before trusting the numbers

This is an **agent audit**, not the human audit the plan specifies. It combines
mechanical checks over all 100 pilot papers with visual comparison of rendered
PDF pages against extracted text for a sample. A human review bench covering the
20 sampled papers is published for the owner to confirm or overturn; until those
verdicts exist, no human-verified usable-text rate is claimed and **G1's "≥90%
audited usable text" remains formally unmet.**

Sample: every fifth paper by canonical ID over the 100-paper pilot (systematic,
reproducible, unbiased). Pages rendered with `pdftoppm` at 105 dpi.

## Mechanical checks across all 100 papers

| Check | Result |
|---|---|
| Documents parsed | 100 / 100 |
| Abstract section detected | 99 / 100 |
| References section detected | 100 / 100 |
| Invalid page spans (start > end, or null) | **0 of 4,385 chunks** |
| Alphabetic-character ratio | median 0.89, min 0.79 |
| Chunks over the 600-token hard cap | 0 |

Zero invalid page spans is the strongest single result here: evidence citations
depend on page provenance, and it holds across the whole corpus.

## Visual verification

Rendered body pages were compared against the text the parser attributed to them.
**ICLR 2024 is a single-column format**, so the multi-column reading-order failure
the plan anticipated largely does not arise for this venue; it returns as a risk
for venues that use two columns (CVPR, ACL), and must be re-checked in P6.1
rather than assumed settled.

On the pages inspected, reading order, page attribution and section attribution
were correct. Inline mathematics is flattened to plain text, which is expected of
a text adapter and does not corrupt the surrounding prose.

## Defects found and fixed (`v2` → `v3`)

| Defect | Before | After |
|---|---|---|
| Running header `"Published as a conference paper at ICLR 2024"` inside chunk text | 2,590 occurrences in 98/100 docs | **0** |
| Precomposed ligatures (`ﬁnite` as U+FB01) unmatched by any lexical query | 1,217 chars in 15/100 docs | **0** |
| Section names mangled by small caps (`B ACKGROUND`) | 696 / 1,696 names | 598 / 1,676 (partial, by design) |

**Running furniture** is now detected by repetition across pages — a line on at
least half the pages of a document of four or more pages — rather than by matching
a venue string, so the rule holds for any source. Bare page-number lines go the
same way. A line appearing on only a couple of pages is kept: dropping it would
silently delete real text.

**Ligatures** are expanded by targeted replacement rather than NFKC, which would
also flatten superscripts and change what a formula means.

**Section names are only partially repaired, deliberately.** A small-capitals
heading (`B ACKGROUND`) and an appendix label (`J THEORETICAL CONNECTION`) are
spelled identically, and joining the latter produces `JTHEORETICAL`. A first
attempt did exactly that, along with `FBEHAVIOR`, `GFULL` and `AWHIRLWIND`; it
also mis-classified maths fragments such as `R 0` as headings, inflating the
corpus from 4,470 to 5,327 chunks. The rule was narrowed to lines carrying two or
more small-capital runs, which an appendix label never has. 598 names remain
mangled. Telling the remaining cases apart needs a dictionary, and a wrong join is
worse than an unrepaired name.

## Known defects NOT fixed

- **Footnotes are interleaved into body prose.** Observed in RA-DIT (paper 3):
  a body sentence is interrupted mid-clause by footnote text, then resumes. The
  text is all present and attributed to the right page, but a chunk can contain a
  sentence spliced by an unrelated footnote. Affects quote fidelity for evidence.
  Not quantified across the corpus and not addressed in `v3`.
- **Dropped ligatures from control-codepoint glyphs.** Where pypdf emits a
  control character instead of a ligature, the glyph is gone before this parser
  sees it: `identification` arrives as `identication`. 48 unambiguous damaged
  tokens across 10 of 100 documents. Irreducible with a plain-text adapter.
- **Mathematics is flattened.** Expected; a structural parser would be needed.

A correction to an earlier draft: a first ligature-damage estimate reported 94 of
100 documents affected. That figure came from a regex matching ordinary English
("reference", "refer", "prefer") and was wrong. Measured against an explicit list
of non-words, it is 10 of 100.

## Corpus after `v3`

4,385 chunks over 100 documents, 1,676 distinct section names, mean 301 tokens,
maximum 598. All 100 versions at `parse_status=parsed`,
`parser_version=pypdf-text-v3`.

## Fixture outcome

`data/fixtures/papers/fixture.pdf` (CC0, 3 pages, 2,415 bytes): 7 sections with
correct page spans, deterministic across reruns. Usable-text rate on the fixture
is 1/1 — not evidence about real papers.

## Failure categories implemented

`missing`, `not_pdf`, `oversized`, `encrypted`, `corrupt`, `empty_text`, each
stored as the version's `parse_status`. Zero occurred across the 100 pilot PDFs.

## Verdict on Docling

Not yet warranted for ICLR. The failures that remain — interleaved footnotes and
dropped ligature glyphs — are ones a layout-aware parser would plausibly fix, but
they affect a small fraction of text and the single-column format removes the
strongest argument. Revisit when P6.1 adds a two-column venue, where reading order
becomes a real risk rather than a hypothetical one.
