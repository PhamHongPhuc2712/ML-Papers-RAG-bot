"""Exact-BM25 candidate retrieval served from a release's sparse vectors.

The stored vectors already hold BM25 document weights, and the query vector
carries the IDF from the same pinned statistics, so Qdrant's plain sparse dot
product is the P2.1 oracle's score — no server-side IDF modifier, which would
count it twice (spec §7). The statistics file is checksum-verified on load: a
query weighted by other statistics would be BM25 against a different corpus.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient, models
from sqlalchemy import Engine

from ..contracts import PaperFilters
from ..corpus.releases import ReleaseRecord, load_release
from .index import PAPERS, SPARSE, IndexBuildError, qdrant_filter
from .lexical import BM25Vocabulary


class SparseRetriever:
    """BM25 over a release's paper or chunk collection."""

    def __init__(self, engine: Engine, client: QdrantClient, *, kind: str = PAPERS) -> None:
        self._engine = engine
        self._client = client
        self._kind = kind
        self._releases: dict[str, tuple[ReleaseRecord, BM25Vocabulary]] = {}

    def _release(self, release_id: str) -> tuple[ReleaseRecord, BM25Vocabulary]:
        cached = self._releases.get(release_id)
        if cached is None:
            release = load_release(self._engine, release_id)
            build = release.counts.get(self._kind)
            if not isinstance(build, dict) or not isinstance(build.get("vocabulary"), dict):
                raise IndexBuildError("collection_not_built", f"{release_id}:{self._kind}")
            vocabulary = BM25Vocabulary.load(
                Path(str(build["vocabulary"]["path"])), sha256=str(build["vocabulary"]["sha256"])
            )
            cached = (release, vocabulary)
            self._releases[release_id] = cached
        return cached

    def search(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> list[tuple[str, float]]:
        if limit < 1:
            raise IndexBuildError("invalid_limit", str(limit))
        release, vocabulary = self._release(release_id)
        encoded = vocabulary.encode_query(query)
        if not encoded.indices:
            # No query term occurs in this corpus: BM25 scores every document 0.
            return []
        hits = self._client.query_points(
            release.collection(self._kind),
            query=models.SparseVector(indices=list(encoded.indices), values=list(encoded.values)),
            using=SPARSE,
            query_filter=qdrant_filter(filters),
            limit=limit,
            with_payload=False,
        ).points
        return [(str(hit.id), float(hit.score)) for hit in hits]
