"""Pure chunking tests using deterministic whitespace tokens."""

from __future__ import annotations

from uuid import UUID

import pytest

from copilot.corpus.chunk import (
    Section,
    chunk_id,
    chunk_sections,
    whitespace_detokenize,
    whitespace_tokenize,
)

PAPER = UUID("11111111-1111-4111-8111-111111111111")
SHA = "ab" * 32


def test_overlap_never_crosses_sections():
    sections = [
        Section("Method", "a b c d e", 1, 1, 0),
        Section("Results", "f g h", 2, 2, 1),
    ]
    chunks = chunk_sections(sections, str.split, " ".join, target=4, overlap=1)
    assert [c["text"] for c in chunks] == ["a b c d", "d e", "f g h"]
    assert chunks[-1]["page_start"] == 2
    assert all(c["token_count"] <= 4 for c in chunks)


def test_long_sections_respect_target_and_cover_every_token():
    text = " ".join(f"w{i}" for i in range(2000))
    chunks = chunk_sections(
        [Section("Body", text, 1, 4, 0)], str.split, " ".join, target=450, overlap=60
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
    chunks = chunk_sections(sections, str.split, " ".join, target=4, overlap=1)
    assert [c["section"] for c in chunks] == ["Body", "More"]
    assert [c["ordinal"] for c in chunks] == [0, 1]
    assert [c["section_ordinal"] for c in chunks] == [1, 3]


@pytest.mark.parametrize(("target", "overlap"), [(0, 0), (4, 4), (4, 5), (4, -1)])
def test_chunk_window_must_be_positive_and_non_overlapping(target, overlap):
    with pytest.raises(ValueError, match="invalid_chunk_window"):
        chunk_sections([Section("S", "a b", 1, 1, 0)], str.split, " ".join, target, overlap)


def test_hard_cap_bounds_the_target_window():
    with pytest.raises(ValueError, match="invalid_chunk_window"):
        chunk_sections(
            [Section("S", "a b", 1, 1, 0)], str.split, " ".join, target=8, overlap=1, hard_cap=6
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
    (single,) = chunk_sections([small], str.split, " ".join, target=4, overlap=1, hard_cap=6)
    assert single["text"] == "r1 r2 r3"
    assert single["kind"] == "table"
    assert single["fragment"] is None

    big = Section("Table 2: Ablation", " ".join(f"c{i}" for i in range(10)), 3, 3, 1, kind="table")
    fragments = chunk_sections([big], str.split, " ".join, target=4, overlap=1, hard_cap=6)
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
    chunks = chunk_sections(sections, str.split, " ".join, target=4, overlap=1)
    assert [c["evidence_default"] for c in chunks] == [True, False]
    assert chunks[1]["kind"] == "references"


def test_whitespace_fixture_tokenizer_round_trips():
    tokens = whitespace_tokenize("a  b\nc\t d")
    assert tokens == ["a", "b", "c", "d"]
    assert whitespace_detokenize(tokens) == "a b c d"
