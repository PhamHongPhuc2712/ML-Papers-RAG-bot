"""Paper candidates found through their chunks (P2.6 step 4, E4's chunk-level retrieval).

A paper's title and abstract is one short text; its body is hundreds of chunks that
say what the abstract only names. Searching the chunk collection and keeping each
paper's best chunk gives the first stage a second way to find a paper, measured as
a candidate source in the retrieval experiments and nowhere else yet.

Only chunks that are evidence by default are searched: a references section names
every paper the author cited, so a query describing one paper would otherwise pull in
everything that cites it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

from qdrant_client import QdrantClient, models

from ..contracts import PaperFilters
from .fusion import RRF_K, rrf
from .index import qdrant_filter

PAPER_FIELD = "paper_id"
EVIDENCE_FIELD = "evidence_default"


def evidence_filter(filters: PaperFilters | None) -> models.Filter:
    """The branches' shared filter, restricted to chunks that are evidence by default."""

    base = qdrant_filter(filters)
    must: list[models.Condition] = list(cast(list[models.Condition], base.must or []))
    must.append(models.FieldCondition(key=EVIDENCE_FIELD, match=models.MatchValue(value=True)))
    return models.Filter(must=must, must_not=base.must_not)


GroupedHits = tuple[list[tuple[str, float]], dict[str, str]]


def paper_groups(
    client: QdrantClient,
    collection: str,
    *,
    query: Any,
    using: str,
    query_filter: models.Filter,
    limit: int,
) -> GroupedHits:
    """The ``limit`` best papers of a chunk collection, each scored by its best chunk.

    Returns the ranked (paper id, score) pairs and, beside them, the id of the chunk
    each paper was found by: the evidence the reranker may read in place of the
    abstract (``rerank.text`` in ``configs/search.yaml``), and what the evaluation
    harness asks whether the hit fell in the right section.
    """

    groups = client.query_points_groups(
        collection,
        query=query,
        using=using,
        query_filter=query_filter,
        group_by=PAPER_FIELD,
        limit=limit,
        group_size=1,
        with_payload=False,
    ).groups
    hits = [(str(group.id), group.hits[0]) for group in groups if group.hits]
    ranked = [(paper_id, float(hit.score)) for paper_id, hit in hits]
    return ranked, {paper_id: str(hit.id) for paper_id, hit in hits}


class CandidateRetriever(Protocol):
    def search(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> list[tuple[str, float]]: ...


class ChunkCandidateRetriever(CandidateRetriever, Protocol):
    """A branch that can also say which chunk each paper was found by."""

    def search_with_chunks(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> GroupedHits: ...


class UnionRetriever:
    """One branch made of several retrievers, fused by RRF: a paper either finds.

    The fused value is the RRF score, so a branch built this way ranks by it;
    the inner retrievers' own scores are not comparable with each other.
    """

    def __init__(self, parts: Sequence[CandidateRetriever], *, k: int = RRF_K) -> None:
        if not parts:
            raise ValueError("union_requires_retrievers")
        self._parts = tuple(parts)
        self._k = k

    def search(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> list[tuple[str, float]]:
        rankings = [
            [paper_id for paper_id, _ in part.search(query, filters, limit, release_id)]
            for part in self._parts
        ]
        return rrf(rankings, self._k)[:limit]


def lexical_branch(
    engine: Any,
    client: QdrantClient,
    *,
    source: str,
    k: int = RRF_K,
    track_best_chunks: bool = False,
) -> Any:
    """The BM25 branch for a candidate source (``configs/search.yaml``, ``candidates.source``)."""

    from .index import CHUNKS
    from .sparse import SparseRetriever

    papers = SparseRetriever(engine, client)
    if source == "papers":
        return papers
    chunks = SparseRetriever(
        engine, client, kind=CHUNKS, group_papers=True, track_best_chunks=track_best_chunks
    )
    if source == "chunks":
        return chunks
    if source == "both":
        return UnionRetriever((papers, chunks), k=k)
    raise ValueError(f"unknown candidate source {source}")


def dense_branch(
    engine: Any,
    client: QdrantClient,
    model: Any,
    *,
    source: str,
    max_tokens: int | None = None,
    k: int = RRF_K,
    track_best_chunks: bool = False,
) -> Any:
    """The dense branch for a candidate source; the query is encoded once per retriever."""

    from .dense import DenseRetriever
    from .index import CHUNKS

    papers = DenseRetriever(engine, client, model, max_tokens=max_tokens)
    if source == "papers":
        return papers
    chunks = DenseRetriever(
        engine,
        client,
        model,
        kind=CHUNKS,
        max_tokens=max_tokens,
        group_papers=True,
        track_best_chunks=track_best_chunks,
    )
    if source == "chunks":
        return chunks
    if source == "both":
        return UnionRetriever((papers, chunks), k=k)
    raise ValueError(f"unknown candidate source {source}")
