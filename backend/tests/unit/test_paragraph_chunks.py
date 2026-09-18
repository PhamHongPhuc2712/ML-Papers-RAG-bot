"""Paragraph-aware packing: an alternative chunking policy to the fixed window."""

from __future__ import annotations

import pytest

from copilot.corpus.chunk import Section, whitespace_spans
from copilot.corpus.chunk_paragraph import (
    pack_paragraphs,
    split_paragraphs,
    split_sentences,
)


def _para(word: str, count: int) -> str:
    """A paragraph of `count` whitespace tokens ending in a full stop."""

    return " ".join([word] * (count - 1)) + " end."


# --- paragraph detection --------------------------------------------------


def test_blank_lines_separate_paragraphs_when_present():
    text = "First one.\nStill first.\n\nSecond starts here.\n\n\nThird."
    assert split_paragraphs(text) == ["First one. Still first.", "Second starts here.", "Third."]


def test_paragraphs_are_inferred_from_short_sentence_final_lines():
    """PDF extraction drops blank lines, so a short closing line is the signal."""

    text = "\n".join(
        [
            "This line runs the full width of the column and keeps going onward",
            "and it continues here at full width without stopping at all yet",
            "but this one ends early.",
            "A new paragraph now begins and again runs the full column width here",
            "and finishes short.",
        ]
    )
    paragraphs = split_paragraphs(text)
    assert len(paragraphs) == 2
    assert paragraphs[0].endswith("but this one ends early.")
    assert paragraphs[1].startswith("A new paragraph")


def test_a_full_width_line_ending_in_a_full_stop_does_not_split():
    """Mid-paragraph sentences end at the margin; only short lines end paragraphs."""

    text = "\n".join(
        [
            "The method converges quickly on every dataset we evaluated here.",
            "The second sentence continues in the very same paragraph as before",
            "and it stops short here.",
        ]
    )
    assert len(split_paragraphs(text)) == 1


# --- sentence splitting ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "We follow Smith et al. 2023 in this setup.",
        "Results improve by 0.5 points on average.",
        "See Fig. 3 for the ablation study.",
        "We use a small model, i.e. the base variant.",
    ],
)
def test_abbreviations_and_decimals_do_not_end_a_sentence(text):
    assert split_sentences(text) == [text]


def test_sentences_split_on_terminal_punctuation():
    text = "The first claim holds. The second does not! Does the third? Yes."
    assert split_sentences(text) == [
        "The first claim holds.",
        "The second does not!",
        "Does the third?",
        "Yes.",
    ]


# --- packing --------------------------------------------------------------


def _pack(section: Section, **kwargs):
    defaults = dict(
        target_tokens=800,
        max_tokens=900,
        paragraph_max_tokens=1200,
        overlap_sentences=0,
    )
    defaults.update(kwargs)
    return pack_paragraphs([section], whitespace_spans, **defaults)


def test_paragraphs_accumulate_until_the_maximum_then_start_a_new_chunk():
    text = "\n\n".join([_para("alpha", 400), _para("beta", 400), _para("gamma", 400)])
    chunks = _pack(Section("Method", text, 1, 2, 0))
    assert len(chunks) == 2
    # 400 + 400 fits under 900; the third would reach 1200, so it opens a chunk.
    assert [c["token_count"] for c in chunks] == [800, 400]


def test_a_paragraph_is_never_split_across_chunks_while_it_fits():
    text = "\n\n".join([_para("alpha", 500), _para("beta", 500)])
    chunks = _pack(Section("Method", text, 1, 2, 0))
    assert [c["token_count"] for c in chunks] == [500, 500]
    assert chunks[0]["text"].startswith("alpha")
    assert chunks[1]["text"].startswith("beta")


def test_an_oversized_paragraph_is_split_on_sentence_boundaries():
    sentence = " ".join(["word"] * 99) + " stop."
    text = " ".join([sentence] * 15)  # 1,500 tokens in one paragraph
    chunks = _pack(Section("Method", text, 1, 2, 0))
    assert len(chunks) > 1
    assert all(c["token_count"] <= 900 for c in chunks)
    # Every chunk ends where a sentence ends, never mid-sentence.
    assert all(c["text"].rstrip().endswith("stop.") for c in chunks)


def test_overlap_repeats_whole_sentences_from_the_previous_chunk():
    # Ordinary prose sentences, well inside the carry budget.
    first = " ".join(_para("alpha", 30) for _ in range(20))
    second = " ".join(_para("beta", 30) for _ in range(20))
    chunks = _pack(Section("Method", f"{first}\n\n{second}", 1, 2, 0), overlap_sentences=1)
    assert len(chunks) >= 2
    tail = split_sentences(chunks[0]["text"])[-1]
    assert chunks[1]["text"].startswith(tail)


def test_a_sentence_larger_than_the_carry_budget_is_not_repeated():
    """Carrying it would duplicate the chunk rather than overlap it."""

    first = _para("alpha", 600)
    second = _para("beta", 600)
    chunks = _pack(Section("Method", f"{first}\n\n{second}", 1, 2, 0), overlap_sentences=1)
    assert [c["token_count"] for c in chunks] == [600, 600]


def test_packing_never_crosses_a_section_boundary():
    sections = [
        Section("Method", _para("alpha", 100), 1, 1, 0),
        Section("Results", _para("beta", 100), 2, 2, 1),
    ]
    chunks = pack_paragraphs(
        sections,
        whitespace_spans,
        target_tokens=800,
        max_tokens=900,
        paragraph_max_tokens=1200,
        overlap_sentences=2,
    )
    assert len(chunks) == 2
    assert chunks[0]["section"] == "Method" and "beta" not in chunks[0]["text"]
    assert chunks[1]["section"] == "Results" and "alpha" not in chunks[1]["text"]


def test_chunks_keep_the_shape_the_persistence_layer_expects():
    (chunk,) = _pack(Section("Table 1: Results", _para("alpha", 50), 3, 3, 4, kind="table"))
    assert set(chunk) == {
        "section",
        "text",
        "token_count",
        "page_start",
        "page_end",
        "ordinal",
        "kind",
        "section_ordinal",
        "evidence_default",
        "fragment",
    }
    assert chunk["kind"] == "table"
    assert chunk["section_ordinal"] == 4
    assert chunk["page_start"] == 3


def test_references_stay_out_of_default_evidence():
    (chunk,) = _pack(Section("References", _para("cite", 40), 9, 9, 7, kind="references"))
    assert chunk["evidence_default"] is False


def test_no_chunk_exceeds_the_ceiling_even_without_sentence_boundaries():
    """Reference lists and table dumps can run for thousands of tokens unbroken."""

    text = " ".join(f"[{i}] Author Name Title Venue Year" for i in range(400))
    assert len(split_sentences(text)) == 1, "no sentence boundary to cut on"
    chunks = _pack(Section("References", text, 9, 9, 3, kind="references"))
    assert len(chunks) > 1
    assert all(c["token_count"] <= 1200 for c in chunks)


def test_a_paragraph_between_the_budget_and_the_ceiling_stays_whole():
    """Keeping the argument intact is the point; a rounder number is not."""

    (chunk,) = _pack(Section("Method", _para("alpha", 1000), 1, 1, 0))
    assert chunk["token_count"] == 1000


def test_overlap_from_huge_sentences_cannot_blow_the_budget():
    """A reference list's "sentences" run to hundreds of tokens each."""

    blob = " ".join(["ref"] * 700) + " year."
    text = "\n\n".join([blob, blob, blob])
    chunks = _pack(Section("References", text, 9, 9, 2, kind="references"), overlap_sentences=2)
    assert all(c["token_count"] <= 1200 for c in chunks), [c["token_count"] for c in chunks]


def test_packing_cannot_emit_more_text_than_it_was_given():
    """Regression: a chunk with no sentence boundary used to carry itself forward.

    The overlap exempted its first sentence from the budget, so a boundary-free
    chunk was repeated into the next one and the output grew without bound — one
    real document reconstructed to 254 KB and produced 7,975 chunks of 1,200
    tokens, far more text than it contained.
    """

    blob = " ".join(["ref"] * 3000)  # no sentence boundary anywhere
    section = Section("References", blob, 9, 9, 0, kind="references")
    chunks = _pack(section, overlap_sentences=2)
    produced = sum(c["token_count"] for c in chunks)
    given = len(whitespace_spans(blob))
    # Overlap may repeat a little; it must not multiply the document.
    assert produced <= given * 1.5, f"{produced} tokens produced from {given}"
    assert len(chunks) < 10
