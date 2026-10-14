"""Frozen datasets: the round trip, the validation rules, and text kept out of git."""

from __future__ import annotations

import json

import pytest

from copilot.evaluation.datasets import (
    Dataset,
    DatasetError,
    QrelRecord,
    QueryRecord,
    SplitRecord,
    hydrate,
    read_dataset,
    validate_dataset,
    write_dataset,
)

SPLIT_CYCLE = ("development", "validation", "test")


def _dataset(n: int, *, in_domain: int = 0) -> Dataset:
    """``n`` queries, one gold each, splits in turn; the first ``in_domain`` are matched."""

    ids = [f"q{i:04d}" for i in range(n)]
    return Dataset(
        queries=tuple(
            QueryRecord(
                query_id=qid,
                query=f"query {i}",
                query_set="synthetic",
                workflow="known_item",
                specificity=1,
                quality=2,
                in_domain=i < in_domain,
                source="test",
                author="test",
            )
            for i, qid in enumerate(ids)
        ),
        qrels=tuple(
            QrelRecord(
                query_id=qid, corpusid=f"c{i}", grade=1, paper_id=f"p{i}" if i < in_domain else None
            )
            for i, qid in enumerate(ids)
        ),
        splits=tuple(
            SplitRecord(query_id=qid, family_id=qid, split=SPLIT_CYCLE[i % 3])
            for i, qid in enumerate(ids)
        ),
    )


def test_round_trip_through_jsonl(tmp_path):
    dataset = _dataset(20)
    write_dataset(dataset, tmp_path)
    reloaded = read_dataset(tmp_path)
    assert reloaded.queries == dataset.queries
    assert reloaded.qrels == dataset.qrels
    assert reloaded.splits == dataset.splits


def test_validation_rejects_a_family_spanning_two_splits(tmp_path):
    """A paraphrase in development and its twin in test leaks the answer."""

    dataset = _dataset(20)
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
    dataset = _dataset(5)
    write_dataset(dataset, tmp_path)
    with (tmp_path / "qrels.jsonl").open("a") as handle:
        handle.write('{"query_id": "ghost", "corpusid": "c1", "grade": 1, "paper_id": null}\n')
    with pytest.raises(DatasetError, match="unknown_query"):
        validate_dataset(tmp_path)


def test_validation_rejects_a_query_with_no_qrel(tmp_path):
    dataset = _dataset(5)
    write_dataset(dataset, tmp_path)
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text("\n".join(qrels.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(DatasetError, match="query_without_qrel"):
        validate_dataset(tmp_path)


def test_validation_rejects_an_unsplit_query(tmp_path):
    dataset = _dataset(5)
    write_dataset(dataset, tmp_path)
    splits = tmp_path / "splits.jsonl"
    splits.write_text("\n".join(splits.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(DatasetError, match="query_without_split"):
        validate_dataset(tmp_path)


def test_a_clean_dataset_validates_and_reports_its_shape(tmp_path):
    dataset = _dataset(30, in_domain=10)
    write_dataset(dataset, tmp_path)
    summary = validate_dataset(tmp_path)
    assert summary["queries"] == 30
    assert summary["in_domain"] == 10
    assert sum(summary["splits"].values()) == 30


def test_the_repository_copy_carries_no_query_text(tmp_path):
    """Benchmark text without a redistribution right stays out of git."""

    dataset = _dataset(5)
    write_dataset(dataset, tmp_path, include_query_text=False)
    raw = (tmp_path / "queries.jsonl").read_text()
    assert '"query"' not in raw
    assert '"query_id"' in raw and '"split' not in raw
    reloaded = read_dataset(tmp_path)
    assert all(record.query == "" for record in reloaded.queries)
    # The split assignment still survives the round trip — that is the point.
    assert reloaded.splits == dataset.splits


def test_a_text_free_dataset_still_validates(tmp_path):
    dataset = _dataset(10)
    write_dataset(dataset, tmp_path, include_query_text=False)
    assert validate_dataset(tmp_path)["queries"] == 10


def test_hydrate_puts_the_text_back(tmp_path):
    dataset = _dataset(3)
    write_dataset(dataset, tmp_path, include_query_text=False)
    reloaded = read_dataset(tmp_path)
    texts = {record.query_id: f"text for {record.query_id}" for record in reloaded.queries}
    filled = hydrate(reloaded, texts)
    assert filled.queries[0].query == f"text for {filled.queries[0].query_id}"


def test_hydrate_refuses_to_leave_a_query_empty(tmp_path):
    dataset = _dataset(3)
    write_dataset(dataset, tmp_path, include_query_text=False)
    reloaded = read_dataset(tmp_path)
    with pytest.raises(DatasetError, match="missing_query_text"):
        hydrate(reloaded, {})
