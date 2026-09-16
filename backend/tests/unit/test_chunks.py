"""Pure chunking tests over token spans, whitespace and model alike."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from copilot.corpus.chunk import (
    Section,
    TokenizerSpec,
    TokenizerUnavailableError,
    chunk_id,
    chunk_sections,
    load_parsing_config,
    load_token_spans,
    token_spans_for,
    tokenizer_cache_path,
    whitespace_spans,
)

PAPER = UUID("11111111-1111-4111-8111-111111111111")
SHA = "ab" * 32


def test_overlap_never_crosses_sections():
    sections = [
        Section("Method", "a b c d e", 1, 1, 0),
        Section("Results", "f g h", 2, 2, 1),
    ]
    chunks = chunk_sections(sections, whitespace_spans, target=4, overlap=1)
    assert [c["text"] for c in chunks] == ["a b c d", "d e", "f g h"]
    assert chunks[-1]["page_start"] == 2
    assert all(c["token_count"] <= 4 for c in chunks)


def test_long_sections_respect_target_and_cover_every_token():
    text = " ".join(f"w{i}" for i in range(2000))
    chunks = chunk_sections(
        [Section("Body", text, 1, 4, 0)], whitespace_spans, target=450, overlap=60
    )
    assert len(chunks) == 5
    assert all(c["token_count"] <= 450 for c in chunks)
    assert chunks[0]["text"].startswith("w0 ")
    assert chunks[-1]["text"].endswith(" w1999")
    assert [c["ordinal"] for c in chunks] == [0, 1, 2, 3, 4]
    assert all(c["section_ordinal"] == 0 for c in chunks)


def test_empty_sections_emit_no_chunks_and_ordinals_are_contiguous():
    sections = [
        Section("Empty", "   ", 1, 1, 0),
        Section("Body", "a b", 1, 1, 1),
        Section("Blank", "", 2, 2, 2),
        Section("More", "c", 2, 2, 3),
    ]
    chunks = chunk_sections(sections, whitespace_spans, target=4, overlap=1)
    assert [c["section"] for c in chunks] == ["Body", "More"]
    assert [c["ordinal"] for c in chunks] == [0, 1]
    assert [c["section_ordinal"] for c in chunks] == [1, 3]


@pytest.mark.parametrize(("target", "overlap"), [(0, 0), (4, 4), (4, 5), (4, -1)])
def test_chunk_window_must_be_positive_and_non_overlapping(target, overlap):
    with pytest.raises(ValueError, match="invalid_chunk_window"):
        chunk_sections([Section("S", "a b", 1, 1, 0)], whitespace_spans, target, overlap)


def test_hard_cap_bounds_the_target_window():
    with pytest.raises(ValueError, match="invalid_chunk_window"):
        chunk_sections(
            [Section("S", "a b", 1, 1, 0)], whitespace_spans, target=8, overlap=1, hard_cap=6
        )


def test_chunk_ids_are_stable_and_depend_on_processing_revisions():
    first = chunk_id(PAPER, SHA, "pypdf-text-v1", "fixed-window-v1", 0, 0)
    same = chunk_id(PAPER, SHA, "pypdf-text-v1", "fixed-window-v1", 0, 0)
    other_chunker = chunk_id(PAPER, SHA, "pypdf-text-v1", "fixed-window-v2", 0, 0)
    other_parser = chunk_id(PAPER, SHA, "pypdf-text-v2", "fixed-window-v1", 0, 0)
    other_document = chunk_id(PAPER, "cd" * 32, "pypdf-text-v1", "fixed-window-v1", 0, 0)
    other_position = chunk_id(PAPER, SHA, "pypdf-text-v1", "fixed-window-v1", 0, 1)
    assert first == same
    assert first.version == 5
    assert len({first, other_chunker, other_parser, other_document, other_position}) == 5


def test_table_sections_stay_coherent_or_become_labeled_fragments():
    small = Section("Table 1: Results", "r1 r2 r3", 2, 2, 0, kind="table")
    (single,) = chunk_sections([small], whitespace_spans, target=4, overlap=1, hard_cap=6)
    assert single["text"] == "r1 r2 r3"
    assert single["kind"] == "table"
    assert single["fragment"] is None

    big = Section("Table 2: Ablation", " ".join(f"c{i}" for i in range(10)), 3, 3, 1, kind="table")
    fragments = chunk_sections([big], whitespace_spans, target=4, overlap=1, hard_cap=6)
    assert [f["fragment"] for f in fragments] == ["1/3", "2/3", "3/3"]
    assert all(f["text"].startswith("Table 2: Ablation (part ") for f in fragments)
    assert all(f["token_count"] <= 4 for f in fragments)
    # Fragments repeat the label rather than overlapping rows.
    assert fragments[1]["text"].endswith("c4 c5 c6 c7")


def test_references_are_kept_but_excluded_from_default_evidence():
    sections = [
        Section("Introduction", "a b", 1, 1, 0),
        Section("References", "[1] x y", 5, 5, 1, kind="references"),
    ]
    chunks = chunk_sections(sections, whitespace_spans, target=4, overlap=1)
    assert [c["evidence_default"] for c in chunks] == [True, False]
    assert chunks[1]["kind"] == "references"


def test_whitespace_spans_point_at_the_original_characters():
    text = "a  b\nc\t d"
    spans = whitespace_spans(text)
    assert [text[start:end] for start, end in spans] == ["a", "b", "c", "d"]
    assert spans[0] == (0, 1)


FIXTURE_TOKENIZER = Path("data/fixtures/tokenizer/tokenizer.json")


def test_windows_are_measured_in_model_tokens():
    """The window is a token budget, not a word budget (audit 2026-09-16).

    Counting whitespace words let real chunks reach ~780 BGE-M3 tokens against
    a documented 600 cap, because scientific English runs ~1.7 tokens a word.
    """

    spans = load_token_spans(FIXTURE_TOKENIZER)
    text = " ".join(
        "Research assistants must cite evidence that exists in the retrieved corpus."
        for _ in range(40)
    )
    chunks = chunk_sections(
        [Section("Method", text, 1, 2, 0)], spans, target=64, overlap=8, hard_cap=80
    )
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        measured = len(spans(chunk["text"]))
        assert measured <= 80, f"{measured} tokens exceeds the hard cap"
        assert chunk["token_count"] == measured
    assert max(chunk["token_count"] for chunk in chunks) > 32, "windows should fill"


def test_chunk_text_is_sliced_from_the_section_without_rewriting_it():
    spans = load_token_spans(FIXTURE_TOKENIZER)
    text = "Reliable retrieval needs  irregular   spacing, punctuation; and (parentheses)."
    (chunk,) = chunk_sections([Section("S", text, 1, 1, 0)], spans, target=400, overlap=10)
    assert chunk["text"] == text


def test_model_tokenizer_is_required_rather_than_silently_replaced(tmp_path):
    spec = TokenizerSpec(
        kind="model", repo="BAAI/bge-m3", revision="deadbeef", file="tokenizer.json", sha256=""
    )
    with pytest.raises(TokenizerUnavailableError, match="tokenizer_missing"):
        token_spans_for(spec, tmp_path)


def test_pinned_tokenizer_checksum_is_verified(tmp_path):
    spec = TokenizerSpec(
        kind="model", repo="BAAI/bge-m3", revision="abc123", file="tokenizer.json", sha256="00" * 32
    )
    target = tokenizer_cache_path(spec, tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(FIXTURE_TOKENIZER.read_bytes())
    with pytest.raises(TokenizerUnavailableError, match="tokenizer_checksum_mismatch"):
        token_spans_for(spec, tmp_path)


def test_parsing_config_requires_a_tokenizer(tmp_path):
    config = tmp_path / "parsing.yaml"
    config.write_text(
        "schema_version: 1\n"
        "parser: {adapter: a, parser_version: v, max_pdf_bytes: 1, reject_encrypted: true}\n"
        "chunker: {policy: fixed-window, chunker_version: v, target_tokens: 10,\n"
        "  hard_cap_tokens: 20, overlap_tokens: 2}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parsing_config_invalid:tokenizer"):
        load_parsing_config(config)
