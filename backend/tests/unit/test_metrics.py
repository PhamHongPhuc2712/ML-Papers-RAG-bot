"""Rank metrics, and the judged coverage that has to travel with them."""

from __future__ import annotations

import pytest

from copilot.evaluation.metrics import (
    MetricError,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_ndcg_obeys_grades_and_deduplicates():
    grades = {"a": 3, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b"], grades, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], grades, 2) < 1.0
    assert ndcg_at_k(["a", "a", "b"], grades, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {}, 10) == 0.0


def test_ndcg_on_binary_grades_matches_hand_calculation():
    """LitSearch labels one gold paper, so every gain is 0 or 1."""

    grades = {"gold": 1}
    assert ndcg_at_k(["gold", "x"], grades, 10) == pytest.approx(1.0)
    # DCG = 1/log2(3) = 0.63093, IDCG = 1/log2(2) = 1.0
    assert ndcg_at_k(["x", "gold"], grades, 10) == pytest.approx(0.63093, abs=1e-5)
    assert ndcg_at_k(["x", "y"], grades, 10) == 0.0


def test_recall_and_mrr_read_the_cutoff():
    relevant = {"a", "b"}
    assert recall_at_k(["a", "z", "b"], relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(["a", "z", "b"], relevant, 3) == pytest.approx(1.0)
    assert mrr_at_k(["z", "a"], relevant, 10) == pytest.approx(0.5)
    assert mrr_at_k(["z", "y"], relevant, 10) == 0.0


def test_recall_counts_a_repeated_hit_once():
    assert recall_at_k(["a", "a"], {"a", "b"}, 10) == pytest.approx(0.5)


def test_empty_relevant_set_is_refused_not_scored():
    """Recall against nothing is 0/0. Returning 0.0 would read as a failure."""

    with pytest.raises(MetricError, match="no_relevant"):
        recall_at_k(["a"], set(), 10)


def test_k_must_be_positive():
    for call in (
        lambda: ndcg_at_k(["a"], {"a": 1}, 0),
        lambda: recall_at_k(["a"], {"a"}, -1),
        lambda: mrr_at_k(["a"], {"a"}, 0),
    ):
        with pytest.raises(MetricError, match="invalid_k"):
            call()


def test_negative_grades_are_rejected():
    with pytest.raises(MetricError, match="invalid_grade"):
        ndcg_at_k(["a"], {"a": -1}, 10)


def test_judged_coverage_is_reported_not_assumed():
    """An unlabelled result is unjudged, never silently irrelevant (spec §11)."""

    result = evaluate(ranked=["gold", "unknown"], grades={"gold": 1}, k=10)
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.judged_coverage == pytest.approx(0.5)
    assert result.judged == 1
    assert result.returned == 2


def test_judged_coverage_ignores_items_past_the_cutoff():
    result = evaluate(ranked=["gold", "a", "b"], grades={"gold": 1}, k=1)
    assert result.judged_coverage == pytest.approx(1.0)


def test_coverage_of_an_empty_ranking_is_unknown_not_perfect():
    result = evaluate(ranked=[], grades={"gold": 1}, k=10)
    assert result.judged_coverage is None
    assert result.recall_at_k == 0.0
