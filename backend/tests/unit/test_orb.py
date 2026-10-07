"""Pinning Open RAG Bench and freezing its text-slice sample (ORB plan, O1)."""

from __future__ import annotations

import json

import pytest

from copilot.evaluation import orb
from copilot.evaluation.datasets import read_dataset, validate_dataset
from copilot.evaluation.orb import (
    ORB_REVISION,
    OrbPaths,
    OrbQuery,
    UnpinnedRevisionError,
    load_orb,
    read_qa_assignment,
    sample_qa,
    text_slice,
    write_orb_dataset,
)


def _queries():
    out = []
    for doc in range(20):
        for i in range(6):
            kind = "extractive" if i % 2 else "abstractive"
            source = "text" if i < 5 else "text-image"
            out.append(OrbQuery(f"q{doc}-{i}", "t", kind, source, f"24{doc:02d}.00001v1", 0, "a"))
    return out


def test_text_slice_keeps_only_text_sourced_queries():
    assert {q.source for q in text_slice(_queries())} == {"text"}


def test_sampling_is_stratified_capped_and_order_independent():
    queries = text_slice(_queries())
    first = sample_qa(queries, seed=42, frozen=20, development=10, per_document=2)
    second = sample_qa(list(reversed(queries)), seed=42, frozen=20, development=10, per_document=2)
    assert first == second
    frozen = [q for q in queries if first[q.query_id] == "frozen"]
    dev = [q for q in queries if first[q.query_id] == "development"]
    assert len(frozen) == 20 and len(dev) == 10
    assert {q.query_id for q in frozen}.isdisjoint({q.query_id for q in dev})
    kinds = [q.type for q in frozen]
    assert abs(kinds.count("extractive") - kinds.count("abstractive")) <= 1
    per_doc: dict[str, int] = {}
    for q in frozen + dev:
        per_doc[q.doc_id] = per_doc.get(q.doc_id, 0) + 1
    assert max(per_doc.values()) <= 2
    rest = [q for q in queries if first[q.query_id] == "retrieval_only"]
    assert len(rest) == len(queries) - 30
    # Another seed is another sample.
    assert sample_qa(queries, seed=7, frozen=20, development=10, per_document=2) != first


def test_the_repository_copy_carries_no_text(tmp_path):
    queries = text_slice(_queries())
    write_orb_dataset(queries, sample_qa(queries), tmp_path, include_text=False)
    body = (tmp_path / "queries.jsonl").read_text()
    assert '"query"' not in body and '"answer"' not in body
    assert not (tmp_path / "answers.jsonl").exists()
    # Labels and the split assignment are there, in the harness's shape.
    shape = validate_dataset(tmp_path)
    assert shape["queries"] == 100 and shape["splits"] == {"retrieval": 100}
    assert shape["families"] == 20
    dataset = read_dataset(tmp_path)
    assert {q.query_set for q in dataset.queries} == {"orb_extractive", "orb_abstractive"}
    assert all(q.in_domain for q in dataset.queries)
    assert {r.corpusid for r in dataset.qrels} == {q.doc_id for q in queries}
    assert all(r.paper_id is None for r in dataset.qrels)
    sections = [json.loads(line) for line in (tmp_path / "sections.jsonl").read_text().splitlines()]
    assert len(sections) == 100 and sections[0].keys() == {"query_id", "doc_id", "section_id"}
    qa = read_qa_assignment(tmp_path)
    assert set(qa.values()) <= {"frozen", "development", "retrieval_only"}
    assert len(qa) == 100


def test_the_hydratable_copy_carries_text_and_answers_and_paper_ids(tmp_path):
    queries = text_slice(_queries())
    ids = {q.doc_id: f"paper-{q.doc_id}" for q in queries}
    write_orb_dataset(queries, sample_qa(queries), tmp_path, include_text=True, paper_ids=ids)
    dataset = read_dataset(tmp_path)
    assert all(q.query == "t" for q in dataset.queries)
    assert all(r.paper_id == f"paper-{r.corpusid}" for r in dataset.qrels)
    answers = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert answers[0] == {"query_id": answers[0]["query_id"], "answer": "a"}


def _benchmark(root):
    labels = root / "pdf" / "arxiv"
    (labels / "corpus").mkdir(parents=True)
    docs = {
        "2401.00001v2": "https://arxiv.org/pdf/2401.00001v2",
        "2401.00002v1": "https://arxiv.org/pdf/2401.00002v1",
    }
    queries = {
        "b": {"query": "second question", "type": "extractive", "source": "text"},
        "a": {"query": "first question", "type": "abstractive", "source": "text-image"},
    }
    qrels = {
        "b": {"doc_id": "2401.00002v1", "section_id": 3},
        "a": {"doc_id": "2401.00001v2", "section_id": 0},
    }
    answers = {"b": "two", "a": "one"}
    (labels / "queries.json").write_text(json.dumps(queries))
    (labels / "qrels.json").write_text(json.dumps(qrels))
    (labels / "answers.json").write_text(json.dumps(answers))
    (labels / "pdf_urls.json").write_text(json.dumps(docs))
    for doc in docs:
        (labels / "corpus" / f"{doc}.json").write_text(
            json.dumps({"id": doc, "title": "T", "sections": []})
        )
    return OrbPaths(root)


def test_load_reads_the_four_label_files_joined_by_query_id_in_id_order(tmp_path):
    paths = _benchmark(tmp_path)
    queries = load_orb(paths)
    assert [q.query_id for q in queries] == ["a", "b"]
    assert queries[1] == OrbQuery(
        "b", "second question", "extractive", "text", "2401.00002v1", 3, "two"
    )
    assert paths.corpus_file("2401.00002v1").exists()


def test_fetch_pins_the_revision_and_checksums_every_file(tmp_path, monkeypatch):
    seen: list[tuple[str, str]] = []
    paths = _benchmark(tmp_path)

    def fake(repo, revision, member, root):
        seen.append((revision, member))
        return root / member

    monkeypatch.setattr(orb, "hf_fetch", fake)
    checksums = orb.fetch_orb(paths)
    assert {revision for revision, _ in seen} == {ORB_REVISION}
    assert {member for _, member in seen} == {
        "pdf/arxiv/queries.json",
        "pdf/arxiv/qrels.json",
        "pdf/arxiv/answers.json",
        "pdf/arxiv/pdf_urls.json",
        "pdf/arxiv/corpus/2401.00001v2.json",
        "pdf/arxiv/corpus/2401.00002v1.json",
    }
    assert set(checksums) == {
        "pdf/arxiv/queries.json",
        "pdf/arxiv/qrels.json",
        "pdf/arxiv/answers.json",
        "pdf/arxiv/pdf_urls.json",
        "corpus",
    }
    assert checksums["corpus"]["files"] == 2 and len(checksums["corpus"]["sha256"]) == 64


def test_a_floating_revision_is_refused(tmp_path):
    for revision in ("main", "refs/convert/parquet", ""):
        with pytest.raises(UnpinnedRevisionError):
            orb.fetch_orb(OrbPaths(tmp_path), revision=revision)


def test_the_committed_fixture_carries_only_ids_labels_and_the_sample():
    """CC-BY-NC-4.0: nothing of the benchmark's text may sit in the repository."""

    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "data" / "fixtures" / "orb"
    allowed = {
        "queries.jsonl": {
            "author",
            "in_domain",
            "quality",
            "query_id",
            "query_set",
            "source",
            "specificity",
            "workflow",
        },
        "qrels.jsonl": {"corpusid", "grade", "paper_id", "query_id"},
        "splits.jsonl": {"family_id", "query_id", "split"},
        "sections.jsonl": {"doc_id", "query_id", "section_id"},
        "qa.jsonl": {"qa_split", "query_id"},
    }
    assert sorted(p.name for p in root.iterdir()) == sorted(allowed)
    for name, keys in allowed.items():
        rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
        assert len(rows) == 1914, name
        assert {k for row in rows for k in row} == keys, name
    shape = validate_dataset(root)
    assert shape["splits"] == {"retrieval": 1914} and shape["families"] == 387
    qa = read_qa_assignment(root)
    assert sum(1 for v in qa.values() if v == "frozen") == 400
    assert sum(1 for v in qa.values() if v == "development") == 100
