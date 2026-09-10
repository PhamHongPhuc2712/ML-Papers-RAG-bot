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

## 20-paper manual audit — PENDING

Not performed. It requires real PDFs from the P1.4 pilot (blocked on
OpenReview credentials at the time of writing) and human review of
source/version/page correctness and multi-column reading order. Until it is
done, no claim is made about pypdf's usable-text rate on ICLR 2024 papers,
and the G1 rule "≥90% audited usable text or parser revised" is unmet.
