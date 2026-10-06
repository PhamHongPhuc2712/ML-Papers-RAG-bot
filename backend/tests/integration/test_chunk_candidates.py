"""Chunk-level candidates over a real release: grouping, the union branch and a recorded run."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from copilot.evaluation.datasets import (
    Dataset,
    QrelRecord,
    QueryRecord,
    SplitRecord,
    write_dataset,
)
from copilot.evaluation.retrieval import run_retrieval
from copilot.models.embeddings import FixtureEmbedding
from copilot.search.chunks import UnionRetriever
from copilot.search.dense import DenseRetriever
from copilot.search.index import CHUNKS, PAPERS, IndexBuildError
from copilot.search.rerank import FixtureReranker
from copilot.search.sparse import SparseRetriever

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def corpus(search_corpus):
    return search_corpus


def test_grouped_chunk_search_answers_with_papers_scored_by_their_best_chunk(corpus):
    paper_ids = set(corpus.ids.values())
    sparse = SparseRetriever(corpus.engine, corpus.client, kind=CHUNKS, group_papers=True)
    dense = DenseRetriever(
        corpus.engine, corpus.client, FixtureEmbedding(), kind=CHUNKS, group_papers=True
    )
    for branch in (sparse, dense):
        hits = branch.search("Message passing on molecular graphs.", None, 3, corpus.release)
        found = [paper_id for paper_id, _ in hits]
        assert 0 < len(found) <= 3
        assert set(found) <= paper_ids, "a grouped search returns paper ids, never chunk ids"
        assert len(found) == len(set(found))
        assert found[0] == corpus.ids["Graph neural networks for molecules"]
        scores = [score for _, score in hits]
        assert scores == sorted(scores, reverse=True)


def test_grouping_is_only_for_the_chunk_collection(corpus):
    with pytest.raises(IndexBuildError, match="group_papers_requires_chunks"):
        SparseRetriever(corpus.engine, corpus.client, kind=PAPERS, group_papers=True)
    with pytest.raises(IndexBuildError, match="group_papers_requires_chunks"):
        DenseRetriever(
            corpus.engine, corpus.client, FixtureEmbedding(), kind=PAPERS, group_papers=True
        )


def test_a_union_branch_finds_what_either_collection_finds(corpus):
    papers = SparseRetriever(corpus.engine, corpus.client)
    chunks = SparseRetriever(corpus.engine, corpus.client, kind=CHUNKS, group_papers=True)
    union = UnionRetriever((papers, chunks))
    query = "Message passing on molecular graphs."
    fused = {paper_id for paper_id, _ in union.search(query, None, 8, corpus.release)}
    only_papers = {paper_id for paper_id, _ in papers.search(query, None, 8, corpus.release)}
    only_chunks = {paper_id for paper_id, _ in chunks.search(query, None, 8, corpus.release)}
    assert fused == only_papers | only_chunks
    assert fused <= set(corpus.ids.values())


class FixtureModels:
    def embedder(self, models: Path):
        return FixtureEmbedding(), None

    def reranker(self, models: Path):
        return FixtureReranker()

    def listwise(self, model_key: str, ledger, cache):  # pragma: no cover - no LLM variant
        raise AssertionError("no LLM variant here")


def _experiment(root: Path, corpus) -> tuple[Path, Path]:
    """Two known-item queries whose text is one chunk each, labelled by paper id."""

    golds = {
        "q-0001": ("Message passing on molecular graphs.", "Graph neural networks for molecules"),
        "q-0002": ("Denoising generates images.", "Diffusion models for image synthesis"),
    }
    queries = tuple(
        QueryRecord(
            query_id=qid,
            query=text,
            query_set="fixture",
            workflow="known_item",
            specificity=1,
            quality=2,
            in_domain=True,
            source="fixture",
            author="fixture",
        )
        for qid, (text, _) in golds.items()
    )
    qrels = tuple(
        QrelRecord(query_id=qid, corpusid=qid, grade=1, paper_id=corpus.ids[title])
        for qid, (_, title) in golds.items()
    )
    splits = tuple(SplitRecord(query_id=qid, family_id=qid, split="development") for qid in golds)
    dataset = root / "dataset"
    write_dataset(Dataset(queries, qrels, splits), dataset, include_query_text=False)
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "queries.jsonl").write_text(
        "".join(json.dumps({"query_id": q.query_id, "query": q.query}) + "\n" for q in queries)
    )
    config = root / "chunks.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "chunk-candidates",
                "dataset": {
                    "path": str(dataset),
                    "query_text": "queries.jsonl",
                    "slice": "in_domain",
                    "labels": "paper_id",
                },
                "corpus": {"release": corpus.release, "papers": "database"},
                "search": "configs/search.yaml",
                "models": str(root / "fixture.yaml"),
                "metrics": {"recall_at": [10], "ndcg_at": [10], "mrr_at": [10], "depth": 10},
                "bootstrap": {"resamples": 100, "seed": 42, "confidence": 0.95},
                "variants": [
                    {"name": "hybrid", "mode": "hybrid", "candidates": "papers"},
                    {"name": "hybrid_chunks", "mode": "hybrid", "candidates": "chunks"},
                    # No source named: follows configs/search.yaml, which says chunks.
                    {"name": "hybrid_default", "mode": "hybrid"},
                    {"name": "hybrid_rerank_both", "mode": "hybrid_rerank", "candidates": "both"},
                ],
                "decision": {"primary": "ndcg@10", "order": ["hybrid"]},
            }
        )
    )
    return config, data_dir


def test_a_run_records_each_variant_s_candidate_source_and_finds_gold_through_chunks(
    corpus, test_settings, tmp_path
):
    config, data_dir = _experiment(tmp_path, corpus)
    metrics = run_retrieval(
        config,
        "development",
        tmp_path / "out",
        data_dir=data_dir,
        database_url=test_settings.database_url,
        qdrant_url=test_settings.qdrant_url,
        models=FixtureModels(),
    )
    variants = metrics["manifest"]["variants"]
    assert variants["hybrid"]["candidates"] == "papers"
    assert "candidates_source" not in variants["hybrid"]["search"]
    assert variants["hybrid_chunks"]["search"]["candidates_source"] == "chunks"
    assert variants["hybrid_rerank_both"]["search"]["candidates_source"] == "both"
    # Unset, a variant follows the shared file; the manifest records what it resolved to.
    assert variants["hybrid_default"]["candidates"] == "chunks"
    assert variants["hybrid_default"]["search_sha256"] == variants["hybrid_chunks"]["search_sha256"]
    # The source is part of what was searched: three sources, three digests.
    assert len({entry["search_sha256"] for entry in variants.values()}) == 3
    rows = pq.read_table(tmp_path / "out" / "development" / "per_query.parquet").to_pylist()
    assert {row["failure"] for row in rows} == {None}
    for row in rows:
        assert row["candidate_recall"] == 1.0
        assert row["ranked"][0] == row["relevant"][0], (row["variant"], row["query_id"])
        assert set(row["ranked"]) <= set(corpus.ids.values())


def test_the_api_builds_its_branches_from_the_configured_source(
    corpus, test_settings, tmp_path, monkeypatch
):
    """configs/search.yaml says `chunks`: a mock-mode service finds papers through them."""

    import yaml

    from copilot.config import Settings
    from copilot.contracts import PaperFilters, SearchRequest
    from copilot.search.api import default_search_service

    shipped = yaml.safe_load(Path("configs/search.yaml").read_text(encoding="utf-8"))
    assert shipped["candidates"]["source"] == "chunks"
    settings = Settings(
        database_url=test_settings.database_url,
        qdrant_url=test_settings.qdrant_url,
        qdrant_collection_prefix=test_settings.qdrant_collection_prefix,
        data_dir=test_settings.data_dir,
        model_mode="mock",
    )
    service = default_search_service(settings, corpus.engine, corpus.client)
    try:
        assert service.config.candidates_source == "chunks"
        assert service.identity("hybrid")["candidates_source"] == "chunks"
        request = SearchRequest(
            query="Message passing on molecular graphs.",
            mode="hybrid",
            filters=PaperFilters(),
            limit=5,
        )
        ranking = service.rank(request)
        assert ranking.ordering.items[0][0] == corpus.ids["Graph neural networks for molecules"]
        assert {paper_id for paper_id, _ in ranking.ordering.items} <= set(corpus.ids.values())
    finally:
        service.close()
    # At paper level the identity carries no source, so orderings cached before this
    # decision never collide with the new ones.
    plain = tmp_path / "papers.yaml"
    shipped["candidates"]["source"] = "papers"
    plain.write_text(yaml.safe_dump(shipped), encoding="utf-8")
    service = default_search_service(settings, corpus.engine, corpus.client, search_path=plain)
    try:
        assert "candidates_source" not in service.identity("hybrid")
    finally:
        service.close()
