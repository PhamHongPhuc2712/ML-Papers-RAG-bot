"""Search papers, then retrieve evidence inside them for one chat turn."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class CorpusRetriever:
    """Return domain evidence from one paper search and its selected paper IDs."""

    search: Callable[[str], PaperHits]
    evidence: Callable[[str, list[str], str], list[Evidence]]

    def search_evidence(self, query: str) -> tuple[list[Evidence], PaperHits]:
        hits = self.search(query)
        if not hits.paper_ids:
            return [], hits
        return self.evidence(query, hits.paper_ids, hits.release_id), hits
