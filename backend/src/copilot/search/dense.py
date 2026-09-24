"""Dense candidate retrieval against one named release (spec §7).

The retriever is given a release id rather than reading the active pointer, so
a request that captured its release once keeps reading that release's pair
even if the pointer moves mid-request. A query is encoded only by the model the
release was built with: a vector from any other model is in another space, and
scoring it against these points would rank confidently and meaninglessly.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from sqlalchemy import Engine

from ..contracts import PaperFilters
from ..corpus.releases import ReleaseRecord, load_release
from ..models.embeddings import VectorModel, validate_vectors
from .index import DENSE, PAPERS, IndexBuildError, qdrant_filter


class DenseRetriever:
    """Nearest neighbours by cosine over a release's paper or chunk collection."""

    def __init__(
        self,
        engine: Engine,
        client: QdrantClient,
        model: VectorModel,
        *,
        kind: str = PAPERS,
        max_tokens: int | None = None,
    ) -> None:
        self._engine = engine
        self._client = client
        self._model = model
        self._kind = kind
        self._max_tokens = max_tokens
        # A release's collection pair and model never change after staging.
        self._releases: dict[str, ReleaseRecord] = {}

    def _release(self, release_id: str) -> ReleaseRecord:
        release = self._releases.get(release_id)
        if release is None:
            release = load_release(self._engine, release_id)
            self._releases[release_id] = release
        return release

    def search(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> list[tuple[str, float]]:
        release = self._release(release_id)
        if release.model_revision != self._model.identity:
            raise IndexBuildError(
                "model_revision_mismatch", f"{self._model.identity} != {release.model_revision}"
            )
        if limit < 1:
            raise IndexBuildError("invalid_limit", str(limit))
        encoded = self._model.encode_array([query], max_tokens=self._max_tokens)
        validate_vectors(encoded.vectors, self._model.dimensions)
        hits = self._client.query_points(
            release.collection(self._kind),
            query=[float(value) for value in encoded.vectors[0]],
            using=DENSE,
            query_filter=qdrant_filter(filters),
            limit=limit,
            with_payload=False,
        ).points
        return [(str(hit.id), float(hit.score)) for hit in hits]
