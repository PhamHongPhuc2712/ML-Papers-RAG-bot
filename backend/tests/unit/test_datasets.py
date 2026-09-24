"""Frozen evaluation splits, and the validation that keeps them honest."""

from __future__ import annotations

import json

import pytest

from copilot.evaluation.datasets import (
    DatasetError,
    QrelRecord,
    QueryRecord,
    assign_splits,
    build_dataset,
    hydrate,
    read_dataset,
    validate_dataset,
    workflow_for,
    write_dataset,
)
from copilot.evaluation.litsearch import LitSearchQuery


def _queries(n: int, query_set: str = "manual_acl") -> list[LitSearchQuery]:
    # Query ids carry the set so two strata never collide in the same fixture.
    return [
        LitSearchQuery(f"{query_set}-{i:04d}", f"q{i}", query_set, 1, 2, (f"c{i}",))
        for i in range(n)
    ]


def test_workflow_comes_from_how_the_query_was_built():
    """Author-written queries describe a known paper; inline ones describe a need."""

    assert workflow_for("manual_acl") == "known_item"
    assert workflow_for("manual_iclr") == "known_item"
    assert workflow_for("inline_acl") == "related_work"
    assert workflow_for("inline_nonacl") == "related_work"
    with pytest.raises(DatasetError, match="unknown_query_set"):
        workflow_for("something_new")


def test_splits_are_deterministic():
    queries = _queries(100)
    assert assign_splits(queries, seed=42) == assign_splits(queries, seed=42)


def test_splits_do_not_depend_on_input_order():
    queries = _queries(100)
    assert assign_splits(queries, seed=42) == assign_splits(list(reversed(queries)), seed=42)


def test_a_different_seed_gives_a_different_split():
    queries = _queries(100)
    assert assign_splits(queries, seed=42) != assign_splits(queries, seed=7)


def test_splits_are_stratified_by_query_set():
    queries = _queries(60, "manual_acl") + _queries(40, "inline_acl")
    splits = assign_splits(queries, seed=42)
    by_set: dict[str, dict[str, int]] = {}
    for query in queries:
        by_set.setdefault(query.query_set, {}).setdefault(splits[query.query_id], 0)
        by_set[query.query_set][splits[query.query_id]] += 1
    # 60/20/20 within each stratum, not merely overall.
    assert by_set["manual_acl"] == {"development": 36, "validation": 12, "test": 12}
    assert by_set["inline_acl"] == {"development": 24, "validation": 8, "test": 8}


def test_every_query_lands_in_exactly_one_split():
    queries = _queries(97)
    splits = assign_splits(queries, seed=42)
    assert set(splits) == {q.query_id for q in queries}
    assert set(splits.values()) <= {"development", "validation", "test"}


def test_in_domain_needs_every_gold_present():
    """A partly-present gold set understates recall, so it is not in-domain."""

    queries = [
        LitSearchQuery("q-all", "?", "manual_acl", 1, 2, ("c1", "c2")),
        LitSearchQuery("q-some", "?", "manual_acl", 1, 2, ("c1", "c9")),
        LitSearchQuery("q-none", "?", "manual_acl", 1, 2, ("c9",)),
    ]
    dataset = build_dataset(queries, matched={"c1": "paper-1", "c2": "paper-2"}, seed=42)
    flags = {q.query_id: q.in_domain for q in dataset.queries}
    assert flags == {"q-all": True, "q-some": False, "q-none": False}


def test_qrels_carry_the_corpus_paper_only_where_it_matched():
    queries = [LitSearchQuery("q1", "?", "manual_acl", 1, 2, ("c1", "c9"))]
    dataset = build_dataset(queries, matched={"c1": "paper-1"}, seed=42)
    qrels = {q.corpusid: q for q in dataset.qrels}
    assert qrels["c1"].paper_id == "paper-1"
    assert qrels["c9"].paper_id is None
    assert all(q.grade == 1 for q in dataset.qrels)


def test_round_trip_through_jsonl(tmp_path):
    dataset = build_dataset(_queries(20), matched={}, seed=42)
    write_dataset(dataset, tmp_path)
    reloaded = read_dataset(tmp_path)
    assert reloaded.queries == dataset.queries
    assert reloaded.qrels == dataset.qrels
    assert reloaded.splits == dataset.splits


def test_validation_rejects_a_family_spanning_two_splits(tmp_path):
    """A paraphrase in development and its twin in test leaks the answer."""

    dataset = build_dataset(_queries(20), matched={}, seed=42)
    write_dataset(dataset, tmp_path)
    records = [json.loads(line) for line in (tmp_path / "splits.jsonl").read_text().splitlines()]
    development = next(r for r in records if r["split"] == "development")
    test = next(r for r in records if r["split"] == "test")
    development["family_id"] = test["family_id"] = "shared-family"
    (tmp_path / "splits.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    )
    with pytest.raises(DatasetError, match="family_spans_splits"):
        validate_dataset(tmp_path)


def test_validation_rejects_a_qrel_for_an_unknown_query(tmp_path):
    dataset = build_dataset(_queries(5), matched={}, seed=42)
    write_dataset(dataset, tmp_path)
    with (tmp_path / "qrels.jsonl").open("a") as handle:
        handle.write('{"query_id": "ghost", "corpusid": "c1", "grade": 1, "paper_id": null}\n')
    with pytest.raises(DatasetError, match="unknown_query"):
        validate_dataset(tmp_path)


def test_validation_rejects_a_query_with_no_qrel(tmp_path):
    dataset = build_dataset(_queries(5), matched={}, seed=42)
    write_dataset(dataset, tmp_path)
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text("\n".join(qrels.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(DatasetError, match="query_without_qrel"):
        validate_dataset(tmp_path)


def test_validation_rejects_an_unsplit_query(tmp_path):
    dataset = build_dataset(_queries(5), matched={}, seed=42)
    write_dataset(dataset, tmp_path)
    splits = tmp_path / "splits.jsonl"
    splits.write_text("\n".join(splits.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(DatasetError, match="query_without_split"):
        validate_dataset(tmp_path)


def test_a_clean_dataset_validates_and_reports_its_shape(tmp_path):
    dataset = build_dataset(_queries(30), matched={f"c{i}": f"p{i}" for i in range(10)}, seed=42)
    write_dataset(dataset, tmp_path)
    summary = validate_dataset(tmp_path)
    assert summary["queries"] == 30
    assert summary["in_domain"] == 10
    assert sum(summary["splits"].values()) == 30


def test_records_carry_their_provenance():
    dataset = build_dataset(_queries(3), matched={}, seed=42)
    record = dataset.queries[0]
    assert isinstance(record, QueryRecord)
    assert record.source.startswith("litsearch@")
    assert record.author == "Ajith et al. 2024 (LitSearch)"
    assert isinstance(dataset.qrels[0], QrelRecord)


def test_the_repository_copy_carries_no_query_text(tmp_path):
    """LitSearch declares no license, so its text stays out of git."""

    dataset = build_dataset(_queries(5), matched={}, seed=42)
    write_dataset(dataset, tmp_path, include_query_text=False)
    raw = (tmp_path / "queries.jsonl").read_text()
    assert '"query"' not in raw
    assert '"query_id"' in raw and '"split' not in raw
    reloaded = read_dataset(tmp_path)
    assert all(record.query == "" for record in reloaded.queries)
    # The split assignment still survives the round trip — that is the point.
    assert reloaded.splits == dataset.splits


def test_a_text_free_dataset_still_validates(tmp_path):
    dataset = build_dataset(_queries(10), matched={}, seed=42)
    write_dataset(dataset, tmp_path, include_query_text=False)
    assert validate_dataset(tmp_path)["queries"] == 10


def test_hydrate_puts_the_text_back(tmp_path):
    dataset = build_dataset(_queries(3), matched={}, seed=42)
    write_dataset(dataset, tmp_path, include_query_text=False)
    reloaded = read_dataset(tmp_path)
    texts = {record.query_id: f"text for {record.query_id}" for record in reloaded.queries}
    filled = hydrate(reloaded, texts)
    assert filled.queries[0].query == f"text for {filled.queries[0].query_id}"


def test_hydrate_refuses_to_leave_a_query_empty(tmp_path):
    dataset = build_dataset(_queries(3), matched={}, seed=42)
    write_dataset(dataset, tmp_path, include_query_text=False)
    reloaded = read_dataset(tmp_path)
    with pytest.raises(DatasetError, match="missing_query_text"):
        hydrate(reloaded, {})
