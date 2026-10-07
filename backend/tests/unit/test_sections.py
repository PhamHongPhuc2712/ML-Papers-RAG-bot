"""Finding a gold section inside our chunks (ORB plan, O3)."""

from __future__ import annotations

import json

from copilot.evaluation.sections import (
    ChunkText,
    locate_section,
    normalized_sentences,
    required_run,
    score_run,
    section_hit,
)

SECTION = """#### 3 Method

We train the model on molecular graphs. Each node carries an atom type.
Messages pass along bonds for four rounds. The readout sums node states.
Finally we fine-tune on the target task.
"""
SENTENCES = [
    "We train the model on molecular graphs.",
    "Each node carries an atom type.",
    "Messages pass along bonds for four rounds.",
    "The readout sums node states.",
    "Finally we fine-tune on the target task.",
]


def test_sentences_are_normalized_and_headings_and_blobs_dropped():
    out = normalized_sentences(SECTION + "\nAAAA" * 200 + "\n#### Next\nOk.")
    assert out[0] == "we train the model on molecular graphs"
    assert len(out) == 5, "the heading, the blob and the two-token line are not sentences"


def test_the_required_run_is_three_or_forty_percent_whichever_is_fewer():
    assert required_run(0) == 0
    assert required_run(1) == 1
    assert required_run(2) == 1
    assert required_run(5) == 2
    assert required_run(10) == 3
    assert required_run(100) == 3


def test_an_exact_section_is_found_in_its_chunk():
    chunks = [ChunkText("c1", "Intro text."), ChunkText("c2", " ".join(SENTENCES))]
    assert locate_section(SECTION, chunks) == {"c2"}


def test_a_section_split_across_two_chunks_maps_to_both():
    chunks = [
        ChunkText("c1", "3 Method " + " ".join(SENTENCES[:2])),
        ChunkText("c2", " ".join(SENTENCES[2:]) + " 4 Results"),
    ]
    assert locate_section(SECTION, chunks) == {"c1", "c2"}


def test_a_single_shared_sentence_is_not_enough_for_a_long_section():
    long_section = "\n".join(
        f"Sentence number {i} says something distinct here." for i in range(10)
    )
    chunks = [
        ChunkText("c1", "Sentence number 4 says something distinct here. Unrelated prose follows.")
    ]
    assert locate_section(long_section, chunks) is None
    # Three consecutive ones are.
    chunks = [
        ChunkText(
            "c1", " ".join(f"Sentence number {i} says something distinct here." for i in (3, 4, 5))
        )
    ]
    assert locate_section(long_section, chunks) == {"c1"}


def test_an_absent_section_returns_none_not_an_empty_set():
    assert locate_section(SECTION, [ChunkText("c1", "Nothing in common.")]) is None
    assert locate_section("", [ChunkText("c1", "x")]) is None


def test_section_hit_is_unknown_without_a_mapping():
    assert section_hit(["c1"], None) is None
    assert section_hit(["c1"], {"c2"}) is False
    assert section_hit(["c1", "c2"], {"c2"}) is True


def test_a_run_is_scored_with_coverage_and_the_two_rates():
    rows = [
        {
            "variant": "v",
            "query_id": "q1",
            "query_set": "a",
            "ranked": ["p1", "p2"],
            "best_chunks": json.dumps({"lexical": "c1"}),
        },
        {
            "variant": "v",
            "query_id": "q2",
            "query_set": "a",
            "ranked": ["p2"],
            "best_chunks": json.dumps({"dense": "c9"}),
        },
        {"variant": "v", "query_id": "q3", "query_set": "b", "ranked": ["p3"], "best_chunks": "{}"},
        {"variant": "v", "query_id": "q4", "query_set": "b", "ranked": [], "best_chunks": "{}"},
    ]
    golds = {"q1": {"c1"}, "q2": {"c2"}, "q3": None, "q4": {"c4"}}
    gold_paper = {"q1": "p1", "q2": "p2", "q3": "p3", "q4": "p4"}
    scores = score_run(rows, golds, gold_paper)
    cell = scores["v"]["all"]
    assert (cell["queries"], cell["mapped"]) == (4, 3)
    assert cell["paper_at_1"] == 0.75
    # q1 hit, q2 top paper right but chunk wrong, q4 no result; q3 unmapped and excluded.
    assert cell["section_hit_at_1"] == 1 / 3
    assert cell["section_hit_given_paper_at_1"] == 0.5
    assert scores["v"]["query_set:b"]["section_hit_at_1"] == 0.0
