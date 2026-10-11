"""The corpus as a LangChain retriever: papers by the search service, passages inside them.

``CorpusRetriever`` is a ``BaseRetriever``, so any LangChain chain can use it; the chat graph
calls ``search_documents``, which also returns which release answered and how the search was
degraded, facts a list of documents cannot carry when it is empty.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ..contracts import Evidence, PaperFilters, SearchRequest
from ..corpus.releases import ReleaseRecord


@dataclass(frozen=True)
class PaperHits:
    release_id: str
    paper_ids: list[str]
    warnings: list[str]


@dataclass
class ServicePaperSearch:
    """One ``hybrid_rerank`` search over the active release, its top ``papers`` ids."""

    service: Any
    release: Callable[[], ReleaseRecord]
    papers: int

    def __call__(self, query: str) -> PaperHits:
        release = self.release()
        request = SearchRequest(
            query=query, mode="hybrid_rerank", filters=PaperFilters(), limit=self.papers
        )
        ranking = self.service.rank(request, release=release)
        ids = [paper_id for paper_id, _ in ranking.ordering.items[: self.papers]]
        return PaperHits(release.id, ids, list(ranking.ordering.warnings))


def to_document(item: Evidence) -> Document:
    return Document(page_content=item.text, metadata=item.model_dump(mode="json"))


def from_document(document: Document) -> Evidence:
    return Evidence.model_validate({**document.metadata, "text": document.page_content})


class CorpusRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    search: Callable[[str], PaperHits]
    evidence: Callable[[str, list[str], str], list[Evidence]]

    def search_documents(self, query: str) -> tuple[list[Document], PaperHits]:
        hits = self.search(query)
        if not hits.paper_ids:
            return [], hits
        items = self.evidence(query, hits.paper_ids, hits.release_id)
        return [to_document(item) for item in items], hits

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.search_documents(query)[0]
