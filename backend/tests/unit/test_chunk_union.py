"""Paper candidates through chunks (P2.6 step 4): the filter and the fused branch."""

from __future__ import annotations

import pytest
from qdrant_client import models

from copilot.contracts import PaperFilters
from copilot.search.chunks import EVIDENCE_FIELD, UnionRetriever, evidence_filter
from copilot.search.fusion import rrf


def _keys(conditions) -> list[str]:
    return [condition.key for condition in conditions or []]


def test_the_chunk_filter_keeps_the_shared_filter_and_adds_the_evidence_condition():
    plain = evidence_filter(None)
    assert _keys(plain.must) == [EVIDENCE_FIELD]
    assert _keys(plain.must_not) == ["deleted"]
    narrowed = evidence_filter(PaperFilters(year_from=2024, venues=["ICLR"]))
    assert _keys(narrowed.must) == ["year", "venue", EVIDENCE_FIELD]
    evidence = narrowed.must[-1]
    assert isinstance(evidence, models.FieldCondition)
    assert evidence.match == models.MatchValue(value=True)


class Fixed:
    def __init__(self, hits) -> None:
        self.hits = hits
        self.calls: list[tuple] = []

    def search(self, query, filters, limit, release_id):
        self.calls.append((query, filters, limit, release_id))
        return list(self.hits)


def test_a_union_branch_fuses_its_parts_by_rrf_and_asks_each_for_the_whole_limit():
    papers = Fixed([("a", 9.0), ("b", 8.0), ("c", 7.0)])
    chunks = Fixed([("c", 0.9), ("d", 0.8), ("a", 0.7)])
    union = UnionRetriever((papers, chunks), k=60)
    hits = union.search("q", None, 3, "release")
    expected = rrf([["a", "b", "c"], ["c", "d", "a"]], 60)[:3]
    assert hits == expected
    assert [paper_id for paper_id, _ in hits] == ["a", "c", "b"]
    assert papers.calls == chunks.calls == [("q", None, 3, "release")]


def test_a_union_needs_at_least_one_part():
    with pytest.raises(ValueError, match="union_requires_retrievers"):
        UnionRetriever(())
