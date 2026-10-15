"""The corpus as a LangChain retriever: papers by the search service, passages inside them.

``CorpusRetriever`` is a ``BaseRetriever``, so any LangChain chain can use it; the chat graph
calls ``search_documents``, which also returns which release answered and how the search was
degraded, facts a list of documents cannot carry when it is empty.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ..contracts import Evidence, PaperFilters, Reference, SearchRequest
from ..corpus.releases import ReleaseRecord


@dataclass(frozen=True)
class PaperHits:
    release_id: str
    paper_ids: list[str]
    warnings: list[str]
    # The ranking's head with metadata, the answer's References; ``paper_ids`` were read.
    references: list[Reference] = field(default_factory=list)


@dataclass
class ServicePaperSearch:
    """One ``hybrid_rerank`` search over the active release: its top ``papers`` ids are read,
    and its top ``references`` papers, with metadata, are the answer's References."""

    service: Any
    release: Callable[[], ReleaseRecord]
    papers: int
    references: int = 20

    def __call__(self, query: str) -> PaperHits:
        release = self.release()
        request = SearchRequest(
            query=query,
            mode="hybrid_rerank",
            filters=PaperFilters(),
            limit=max(self.papers, self.references),
        )
        ranking = self.service.rank(request, release=release)
        ids = [paper_id for paper_id, _ in ranking.ordering.items[: self.papers]]
        # The search API's own page: canonical metadata, and a paper gone from the database
        # left out with ``metadata_missing`` rather than shown without a title.
        page = self.service.page(
            ranking.ordering,
            offset=0,
            limit=request.limit,
            request_id=ranking.request_id,
            rows=ranking.rows,
        )
        read = set(ids)
        references = [
            Reference(
                paper_id=item.paper_id,
                rank=item.rank,
                read=str(item.paper_id) in read,
                title=page.papers[str(item.paper_id)].title,
                venue=page.papers[str(item.paper_id)].venue or None,
                year=page.papers[str(item.paper_id)].year or None,
            )
            for item in page.items
        ]
        return PaperHits(release.id, ids, list(page.warnings), references)


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
