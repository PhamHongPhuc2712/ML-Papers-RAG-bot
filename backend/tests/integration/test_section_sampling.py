"""Sampling is one passage per paper, body text only, deterministic."""

from __future__ import annotations

import pytest

from copilot.evaluation.section_queries import sample_chunks

pytestmark = pytest.mark.integration


def test_one_body_passage_per_paper_and_the_same_twice(search_corpus):
    first = sample_chunks(search_corpus.engine, per_venue_year=10, seed=42, min_tokens=1)
    assert first == sample_chunks(search_corpus.engine, per_venue_year=10, seed=42, min_tokens=1)
    assert len({sample.paper_id for sample in first}) == len(first) == len(search_corpus.ids)
    capped = sample_chunks(search_corpus.engine, per_venue_year=1, seed=42, min_tokens=1)
    assert len(capped) == len({(s.venue, s.year) for s in first})
