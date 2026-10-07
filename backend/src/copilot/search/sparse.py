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
from .chunks import evidence_filter, paper_groups
from .index import CHUNKS, PAPERS, SPARSE, IndexBuildError, qdrant_filter
from .lexical import BM25Vocabulary


class SparseRetriever:
    """BM25 over a release's paper or chunk collection.

    With ``group_papers`` the chunk collection answers with papers: each paper's
    best evidence chunk stands for it (P2.6 step 4).
    """

    def __init__(
        self,
        engine: Engine,
        client: QdrantClient,
        *,
        kind: str = PAPERS,
        group_papers: bool = False,
        track_best_chunks: bool = False,
    ) -> None:
        if group_papers and kind != CHUNKS:
            raise IndexBuildError("group_papers_requires_chunks", kind)
        self._engine = engine
        self._client = client
        self._kind = kind
        self._group_papers = group_papers
        # Diagnostic, opt-in and for one caller at a time: the last grouped search's
        # best chunk per paper. The evaluation harness reads it; the API never does.
        self._track_best_chunks = track_best_chunks
        self.best_chunks: dict[str, str] = {}
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
        vector = models.SparseVector(indices=list(encoded.indices), values=list(encoded.values))
        if self._group_papers:
            if self._track_best_chunks:
                self.best_chunks = {}
            return paper_groups(
                self._client,
                release.collection(self._kind),
                query=vector,
                using=SPARSE,
                query_filter=evidence_filter(filters),
                limit=limit,
                best_chunks=self.best_chunks if self._track_best_chunks else None,
            )
        hits = self._client.query_points(
            release.collection(self._kind),
            query=vector,
            using=SPARSE,
            query_filter=qdrant_filter(filters),
            limit=limit,
            with_payload=False,
        ).points
        return [(str(hit.id), float(hit.score)) for hit in hits]
