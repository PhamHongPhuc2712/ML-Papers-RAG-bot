"""The LangChain retriever over the real search service and evidence retriever (fixture models)."""

from __future__ import annotations

from pathlib import Path

import pytest

from copilot.chat.retriever import CorpusRetriever, ServicePaperSearch, from_document
from copilot.corpus.releases import load_release
from copilot.evidence.retrieve import EvidenceConfig, EvidenceRetriever
from copilot.models.embeddings import FixtureEmbedding
from copilot.search.api import default_search_service
from copilot.search.rerank import FixtureReranker

pytestmark = pytest.mark.integration


def test_the_retriever_returns_passages_from_the_papers_search_found(search_corpus, test_settings):
    service = default_search_service(test_settings, search_corpus.engine, search_corpus.client)
    evidence = EvidenceRetriever(
        engine=search_corpus.engine,
        client=search_corpus.client,
        embedder=FixtureEmbedding(),
        reranker=FixtureReranker(),
        config=EvidenceConfig(tokenizer_kind="whitespace"),
        data_dir=Path("/nonexistent"),
    )
    retriever = CorpusRetriever(
        search=ServicePaperSearch(
            service, lambda: load_release(search_corpus.engine, search_corpus.release), papers=3
        ),
        evidence=evidence.retrieve,
    )
    documents = retriever.invoke("graph neural networks message passing")
    assert documents, "the fixture corpus has graph papers"
    ids = set(search_corpus.ids.values())
    for document in documents:
        item = from_document(document)
        assert str(item.source_id) in ids and item.chunk_id is not None
    _, hits = retriever.search_documents("graph neural networks message passing")
    assert hits.release_id == search_corpus.release and len(hits.paper_ids) <= 3


def test_the_search_returns_its_ranked_references_the_first_ones_read(
    search_corpus, test_settings
):
    service = default_search_service(test_settings, search_corpus.engine, search_corpus.client)
    search = ServicePaperSearch(
        service,
        lambda: load_release(search_corpus.engine, search_corpus.release),
        papers=2,
        references=4,
    )
    hits = search("graph neural networks message passing")
    references = hits.references
    assert 2 < len(references) <= 4
    assert [r.rank for r in references] == sorted(r.rank for r in references)
    assert [str(r.paper_id) for r in references if r.read] == hits.paper_ids
    titles = {paper_id: title for title, paper_id in search_corpus.ids.items()}
    for reference in references:
        assert reference.title == titles[str(reference.paper_id)]
