"""P2.6's offline gap analysis: read recorded runs, never the test split, never query text."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from copilot.evaluation.datasets import Dataset, QrelRecord, QueryRecord, SplitRecord, write_dataset
from copilot.evaluation.gaps import (
    GapError,
    gold_rank_buckets,
    load_outcomes,
    misses,
    render_gaps,
    stage_losses,
)
from copilot.evaluation.retrieval import dataset_digest

# (query id, query set, specificity, gold ids)
QUERIES = (
    ("q1", "set_a", 1, ("a",)),
    ("q2", "set_b", 0, ("b",)),
    ("q3", "set_c", 1, ("c", "d")),
    ("q4", "set_d", 0, ("e",)),
)
BOOTSTRAP = {"resamples": 200, "seed": 42, "confidence": 0.95}


def _ranking(gold_at: dict[str, int], length: int = 50) -> list[str]:
    """A ranking of fillers with each gold placed at its one-based rank."""

    ranked = [f"x{index}" for index in range(length)]
    for gold, rank in gold_at.items():
        ranked[rank - 1] = gold
    return ranked


def _row(variant, query_id, ranked, candidate_recall, failure=None, specificity=None):
    query = next(q for q in QUERIES if q[0] == query_id)
    return {
        "variant": variant,
        "query_id": query_id,
        "family_id": query_id,
        "query_set": query[1],
        "specificity": specificity,
        "ranked": ranked,
        "relevant": sorted(query[3]),
        "candidate_recall": candidate_recall,
        "rerank_pool_recall": None,
        "seconds": 0.1,
        "stages": "{}",
        "warnings": [],
        "failure": failure,
    }


ROWS = (
    # q1: gold at rank 1. q2: gold at rank 30, inside the pool.
    # q3: one gold at rank 3, the other in the pool but past 50.
    # q4: gold not in the pool at all.
    _row("hybrid_rerank", "q1", _ranking({"a": 1}), 1.0),
    _row("hybrid_rerank", "q2", _ranking({"b": 30}), 1.0),
    _row("hybrid_rerank", "q3", _ranking({"c": 3}), 1.0),
    _row("hybrid_rerank", "q4", _ranking({}), 0.0),
    # A failed query: nothing ranked, still counted.
    _row("bm25", "q1", [], 0.0, failure="candidates_unavailable"),
    _row("bm25", "q2", _ranking({"b": 4}), 1.0),
    _row("bm25", "q3", _ranking({"c": 1, "d": 2}), 1.0),
    _row("bm25", "q4", _ranking({"e": 12}), 1.0),
)


def _dataset(directory: Path) -> None:
    queries = tuple(
        QueryRecord(
            query_id=qid,
            query="",
            query_set=qset,
            workflow="known_item",
            specificity=spec,
            quality=2,
            in_domain=False,
            source="fixture",
            author="fixture",
        )
        for qid, qset, spec, _ in QUERIES
    )
    qrels = tuple(
        QrelRecord(query_id=qid, corpusid=gold, grade=1, paper_id=gold)
        for qid, _, _, golds in QUERIES
        for gold in golds
    )
    splits = tuple(
        SplitRecord(query_id=qid, family_id=qid, split="validation") for qid, *_ in QUERIES
    )
    write_dataset(Dataset(queries, qrels, splits), directory, include_query_text=False)


def _run(tmp_path: Path, rows=ROWS) -> Path:
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    run = tmp_path / "run"
    split = run / "validation"
    split.mkdir(parents=True)
    manifest = {
        "experiment": {"name": "fixture"},
        "corpus": {"release_id": "fixture-release", "papers": "database"},
        "dataset": {
            "path": str(dataset),
            "digest": dataset_digest(dataset),
            "split": "validation",
            "slice": "all",
            "labels": "paper_id",
        },
        "search": {"values": {"candidates_per_branch": 100, "rerank_depth": 50}},
        "bootstrap": BOOTSTRAP,
    }
    (split / "manifest.json").write_text(json.dumps(manifest))
    pq.write_table(pa.Table.from_pylist(list(rows)), split / "per_query.parquet")
    return run


def test_the_test_split_is_never_read(tmp_path):
    run = _run(tmp_path)
    with pytest.raises(GapError, match="held_out_split"):
        load_outcomes(run, "test")


def test_specificity_comes_from_the_frozen_dataset_when_the_run_lacks_it(tmp_path):
    _, outcomes = load_outcomes(_run(tmp_path), "validation")
    by_query = {outcome.query_id: outcome.specificity for outcome in outcomes}
    assert by_query == {"q1": 1, "q2": 0, "q3": 1, "q4": 0}


def test_a_dataset_changed_since_the_run_is_refused(tmp_path):
    run = _run(tmp_path)
    labels = tmp_path / "dataset" / "qrels.jsonl"
    labels.write_text(labels.read_text() + labels.read_text().splitlines()[0] + "\n")
    with pytest.raises(GapError, match="dataset_changed"):
        load_outcomes(run, "validation")


def test_a_failed_query_scores_zero_and_stays_in_the_denominator(tmp_path):
    _, outcomes = load_outcomes(_run(tmp_path), "validation")
    losses = stage_losses(outcomes, depth=50)["bm25"]
    # q1 failed: its gold is neither returned nor pooled, and it still counts.
    assert losses["all"]["queries"] == 4
    assert losses["query_set"]["set_a"]["not_in_pool"] == pytest.approx(1.0)
    ranks = gold_rank_buckets(outcomes)["bm25"]["all"]
    assert ranks["absent"] == 1


def test_every_gold_is_returned_cut_or_never_found(tmp_path):
    _, outcomes = load_outcomes(_run(tmp_path), "validation")
    losses = stage_losses(outcomes, depth=50)["hybrid_rerank"]
    whole = losses["all"]
    assert whole["queries"] == 4
    # q1 1, q2 1, q3 1/2, q4 0 within 50; q3's second gold is in the pool beyond 50.
    assert whole["within_depth"] == pytest.approx(2.5 / 4)
    assert whole["beyond_depth"] == pytest.approx(0.5 / 4)
    assert whole["not_in_pool"] == pytest.approx(1 / 4)
    for cell in [whole, *losses["query_set"].values(), *losses["specificity"].values()]:
        total = cell["within_depth"] + cell["beyond_depth"] + cell["not_in_pool"]
        assert total == pytest.approx(1.0)
    # A shallower cut moves q2's gold (rank 30) from returned to beyond.
    assert stage_losses(outcomes, depth=20)["hybrid_rerank"]["all"]["within_depth"] == (
        pytest.approx(1.5 / 4)
    )


def test_gold_ranks_count_each_gold_once(tmp_path):
    rows = [*ROWS[:2], _row("hybrid_rerank", "q3", ["c", "c", "x", "d"], 1.0), *ROWS[3:]]
    _, outcomes = load_outcomes(_run(tmp_path, rows), "validation")
    ranks = gold_rank_buckets(outcomes)["hybrid_rerank"]
    # The repeated "c" ranks once, so "d" is third, not fourth.
    assert ranks["all"] == {"1": 2, "2-5": 1, "6-10": 0, "11-20": 0, "21-50": 1, "absent": 1}
    assert sum(ranks["specificity"]["0"].values()) == 2


def test_misses_name_query_ids_by_slice_and_never_text(tmp_path):
    _, outcomes = load_outcomes(_run(tmp_path), "validation")
    found = misses(outcomes, "hybrid_rerank", depth=50)
    assert found["set_d / 0"]["not_in_pool"] == ["q4"]
    assert found["set_c / 1"]["beyond_depth"] == ["q3"]
    assert found["set_a / 1"] == {"queries": 1, "not_in_pool": [], "beyond_depth": []}


def test_the_report_names_the_losses_and_query_ids_only(tmp_path):
    report = render_gaps(_run(tmp_path), focus="hybrid_rerank")
    text = report.read_text()
    assert "Where the gold paper is lost" in text and "q4" in text
    data = json.loads((report.parent / "gaps.json").read_text())
    assert set(data["splits"]["validation"]) == {"queries", "stage_losses", "gold_ranks", "misses"}
