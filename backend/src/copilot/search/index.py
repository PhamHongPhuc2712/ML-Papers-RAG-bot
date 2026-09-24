"""Index lifecycle: build a release's collection pair, then prove it before serving.

A release is one corpus snapshot embedded by one pinned model into two Qdrant
collections, ``paper_abstracts_<release>`` and ``paper_chunks_<release>``. Each
point carries a named dense vector and a named exact-BM25 sparse vector whose
weights are already final, so Qdrant applies no IDF modifier and its sparse dot
product is the P2.1 oracle's score (spec §7).

Papers are read from the snapshot itself. Chunk text is withheld from the
snapshot for rights reasons, so chunks are read back from the database — and
the build digests every (id, text sha256) pair it indexes, in id order, which is
the same digest the snapshot manifest recorded at export. Activation compares
the two, so text that drifted in the database between export and build refuses
to serve instead of serving silently.

The collections of one release are built one at a time — abstracts first — but
the release activates only when both validate: count, dimensions, distance,
digests, the pinned BM25 statistics, and canary rankings recomputed with exact
search. A half-built release therefore cannot be served, and whatever was
active keeps serving.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text

from ..contracts import PaperFilters
from ..corpus.export import EXPORTED_CHUNKS, shard_paths, validate_manifest
from ..corpus.releases import (
    ReleaseError,
    ReleaseRecord,
    Validator,
    load_release,
    record_collection_build,
    stage_release,
)
from ..models.embeddings import (
    EmbeddingCache,
    EncodedBatch,
    Matrix,
    VectorModel,
    validate_vectors,
)
from .lexical import BM25Vocabulary, SparseVector, paper_text

PAPERS = "papers"
CHUNKS = "chunks"
KINDS = (PAPERS, CHUNKS)
DENSE = "dense"
SPARSE = "bm25"
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
_COLLECTION_STEMS = {PAPERS: "paper_abstracts", CHUNKS: "paper_chunks"}
# Two canary scores closer than this are a tie whose order may legitimately
# differ after a restore reorders segments.
_TIE = 1e-5
# Extra results fetched when re-checking a ranking, so a tied item that moved
# across the recorded limit is found rather than reported missing.
CANARY_SLACK = 20


class IndexBuildError(RuntimeError):
    """An index cannot be built, validated or queried as asked."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class CollectionSettings:
    """Storage layout for one collection kind; none of it changes a ranking."""

    max_tokens: int
    on_disk: bool = False
    quantization: str | None = None
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    defer_indexing: bool = False


@dataclass(frozen=True)
class IndexConfig:
    collections: Mapping[str, CollectionSettings]
    batch_points: int = 256
    canary_queries: int = 5
    canary_limit: int = 10
    canary_terms: int = 12


def load_index_config(path: str | Path) -> IndexConfig:
    """Read the ``index`` section of ``configs/models.yaml``."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise IndexBuildError("models_config_invalid", "root")
    index = raw.get("index")
    embedding = raw.get("embedding")
    if not isinstance(index, Mapping) or not isinstance(embedding, Mapping):
        raise IndexBuildError("models_config_invalid", "index")
    max_tokens = embedding.get("max_tokens") or {}
    collections: dict[str, CollectionSettings] = {}
    for kind in KINDS:
        settings = index.get("collections", {}).get(kind)
        if not isinstance(settings, Mapping) or kind not in max_tokens:
            raise IndexBuildError("models_config_invalid", f"collections.{kind}")
        hnsw = settings.get("hnsw") or {}
        collections[kind] = CollectionSettings(
            max_tokens=int(max_tokens[kind]),
            on_disk=bool(settings.get("on_disk", False)),
            quantization=settings.get("quantization"),
            hnsw_m=int(hnsw.get("m", 16)),
            hnsw_ef_construct=int(hnsw.get("ef_construct", 128)),
            defer_indexing=bool(settings.get("defer_indexing", False)),
        )
    canaries = index.get("canaries") or {}
    return IndexConfig(
        collections=collections,
        batch_points=int(index.get("batch_points", 256)),
        canary_queries=int(canaries.get("queries", 5)),
        canary_limit=int(canaries.get("limit", 10)),
        canary_terms=int(canaries.get("terms", 12)),
    )


def collection_names(prefix: str, release_id: str) -> dict[str, str]:
    """The pair a release names, inside the configured namespace (``dev_``, ``test_``)."""

    if not _RELEASE_ID.match(release_id):
        raise IndexBuildError("invalid_release_id", release_id)
    return {kind: f"{prefix}{stem}_{release_id}" for kind, stem in _COLLECTION_STEMS.items()}


def qdrant_filter(filters: PaperFilters | None) -> models.Filter:
    """The payload filter both candidate branches apply identically (spec §7)."""

    must: list[models.Condition] = []
    if filters is not None:
        if filters.year_from is not None or filters.year_to is not None:
            must.append(
                models.FieldCondition(
                    key="year", range=models.Range(gte=filters.year_from, lte=filters.year_to)
                )
            )
        if filters.venues:
            must.append(
                models.FieldCondition(key="venue", match=models.MatchAny(any=filters.venues))
            )
        if filters.fulltext_only:
            must.append(models.FieldCondition(key="fulltext", match=models.MatchValue(value=True)))
    return models.Filter(
        must=must or None,
        must_not=[models.FieldCondition(key="deleted", match=models.MatchValue(value=True))],
    )


@dataclass(frozen=True)
class IndexDocument:
    id: str
    text: str
    text_sha256: str
    payload: dict[str, Any]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot_papers(
    manifest_path: str | Path, *, limit: int | None = None
) -> Iterator[IndexDocument]:
    """Paper documents straight from a validated snapshot, in paper-id order."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    directory = Path(manifest_path).parent
    emitted = 0
    for shard in shard_paths(manifest, PAPERS):
        for batch in pq.ParquetFile(directory / shard).iter_batches(batch_size=10_000):
            for row in batch.to_pylist():
                if limit is not None and emitted >= limit:
                    return
                document = paper_text(str(row["title"]), row.get("abstract"))
                digest = _sha256_text(document)
                if row.get("text_sha256") and row["text_sha256"] != digest:
                    raise IndexBuildError("snapshot_text_mismatch", str(row["paper_id"]))
                yield IndexDocument(
                    id=str(row["paper_id"]),
                    text=document,
                    text_sha256=digest,
                    payload={
                        "paper_id": str(row["paper_id"]),
                        "year": row.get("year"),
                        "venue": row.get("venue"),
                        "track": row.get("track"),
                        "fulltext": row.get("parse_status") == "parsed",
                        "version": row.get("content_sha256"),
                        "deleted": False,
                    },
                )
                emitted += 1


_CHUNK_DOCUMENTS = f"""
select c.id::text as chunk_id,
       pv.paper_id::text as paper_id,
       c.paper_version_id::text as paper_version_id,
       c.text,
       c.kind,
       c.section_path,
       c.ordinal,
       c.page_start,
       c.page_end,
       c.evidence_default,
       p.publication_year as year,
       v.name as venue
{EXPORTED_CHUNKS}
left join venues v on v.id = p.venue_id
"""


def database_chunks(
    engine: Engine, *, limit: int | None = None, ordered: bool = True, batch_rows: int = 5_000
) -> Iterator[IndexDocument]:
    """Chunk documents from the database, in the snapshot's own chunk-id order.

    Ordering 3.4 M rows of text by id is a multi-gigabyte sort. The BM25
    statistics do not depend on order, so their pass asks for ``ordered=False``
    and reads the table sequentially; a limited pass must stay ordered, because
    "the first N" only means something in id order.
    """

    statement = _CHUNK_DOCUMENTS
    if ordered or limit is not None:
        statement += "order by c.id\n"
    if limit is not None:
        statement += f"limit {int(limit)}\n"
    with engine.connect() as connection:
        result = connection.execution_options(stream_results=True, yield_per=batch_rows).execute(
            text(statement)
        )
        for partition in result.mappings().partitions(batch_rows):
            for row in partition:
                yield IndexDocument(
                    id=str(row["chunk_id"]),
                    text=str(row["text"]),
                    text_sha256=_sha256_text(str(row["text"])),
                    payload={
                        "chunk_id": str(row["chunk_id"]),
                        "paper_id": str(row["paper_id"]),
                        "paper_version_id": str(row["paper_version_id"]),
                        "year": row["year"],
                        "venue": row["venue"],
                        "section": row["section_path"],
                        "kind": row["kind"],
                        "ordinal": row["ordinal"],
                        "page_start": row["page_start"],
                        "page_end": row["page_end"],
                        "evidence_default": bool(row["evidence_default"]),
                        "fulltext": True,
                        "deleted": False,
                    },
                )


def _batched(items: Iterable[IndexDocument], size: int) -> Iterator[list[IndexDocument]]:
    batch: list[IndexDocument] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


_PAYLOAD_INDEXES: dict[str, dict[str, models.PayloadSchemaType]] = {
    PAPERS: {
        "paper_id": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
        "venue": models.PayloadSchemaType.KEYWORD,
        "fulltext": models.PayloadSchemaType.BOOL,
        "deleted": models.PayloadSchemaType.BOOL,
    },
    CHUNKS: {
        "paper_id": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
        "venue": models.PayloadSchemaType.KEYWORD,
        "kind": models.PayloadSchemaType.KEYWORD,
        "evidence_default": models.PayloadSchemaType.BOOL,
        "fulltext": models.PayloadSchemaType.BOOL,
        "deleted": models.PayloadSchemaType.BOOL,
    },
}


def ensure_collection(
    client: QdrantClient, name: str, kind: str, settings: CollectionSettings, dimensions: int
) -> None:
    """Create the collection once; an existing one must already have this shape."""

    if client.collection_exists(name):
        _check_shape(client, name, dimensions)
        return
    quantization = None
    if settings.quantization == "int8":
        quantization = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
        )
    elif settings.quantization is not None:
        raise IndexBuildError("quantization_unsupported", str(settings.quantization))
    client.create_collection(
        name,
        vectors_config={
            DENSE: models.VectorParams(
                size=dimensions, distance=models.Distance.COSINE, on_disk=settings.on_disk
            )
        },
        # No modifier: the stored weights are final BM25 document weights, and
        # the query side carries the IDF (spec §7).
        sparse_vectors_config={
            SPARSE: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=settings.on_disk)
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=settings.hnsw_m, ef_construct=settings.hnsw_ef_construct
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0)
        if settings.defer_indexing
        else None,
        quantization_config=quantization,
        on_disk_payload=settings.on_disk,
    )
    # Queued on the collection's update log ahead of every upsert, so they apply
    # in order without a round trip each; a filter works before they finish.
    for field, schema in _PAYLOAD_INDEXES[kind].items():
        client.create_payload_index(name, field_name=field, field_schema=schema, wait=False)


def _check_shape(client: QdrantClient, name: str, dimensions: int) -> list[str]:
    problems: list[str] = []
    params = client.get_collection(name).config.params
    vectors = params.vectors if isinstance(params.vectors, Mapping) else {}
    dense = vectors.get(DENSE)
    if dense is None:
        problems.append(f"{name}: no dense vector")
    else:
        if dense.size != dimensions:
            problems.append(f"{name}: dense dimension {dense.size} != {dimensions}")
        if dense.distance != models.Distance.COSINE:
            problems.append(f"{name}: distance {dense.distance} is not cosine")
    sparse = (params.sparse_vectors or {}).get(SPARSE)
    if sparse is None:
        problems.append(f"{name}: no bm25 sparse vector")
    elif sparse.modifier not in (None, models.Modifier.NONE):
        problems.append(f"{name}: sparse modifier {sparse.modifier} double-counts IDF")
    return problems


def _point(
    document: IndexDocument, dense: Sequence[float], sparse: SparseVector
) -> models.PointStruct:
    vector: dict[str, Any] = {DENSE: [float(value) for value in dense]}
    # A document with no terms has no sparse vector; it is still findable densely.
    if sparse.indices:
        vector[SPARSE] = models.SparseVector(
            indices=list(sparse.indices), values=list(sparse.values)
        )
    return models.PointStruct(id=document.id, vector=vector, payload=document.payload)


def _encode(
    model: VectorModel,
    batch: Sequence[IndexDocument],
    max_tokens: int,
    cache: EmbeddingCache | None,
) -> EncodedBatch:
    key = EmbeddingCache.key([document.text_sha256 for document in batch], max_tokens)
    if cache is not None:
        cached = cache.load(key)
        if cached is not None:
            return cached
    encoded = model.encode_array([document.text for document in batch], max_tokens=max_tokens)
    # Checked before the cache or the index sees it: nothing nonfinite is stored.
    validate_vectors(encoded.vectors, model.dimensions)
    if cache is not None:
        cache.store(key, encoded)
    return encoded


def _canary_query(document: IndexDocument, terms: int) -> str:
    return " ".join(document.text.split()[:terms])


def _hits(points: Sequence[models.ScoredPoint]) -> list[list[Any]]:
    return [[str(point.id), float(point.score)] for point in points]


def _exact_search(
    client: QdrantClient,
    collection: str,
    query: list[float] | models.SparseVector,
    using: str,
    limit: int,
) -> list[list[Any]]:
    return _hits(
        client.query_points(
            collection,
            query=query,
            using=using,
            limit=limit,
            search_params=models.SearchParams(exact=True),
            with_payload=False,
        ).points
    )


def same_ranking(expected: Sequence[Sequence[Any]], actual: Sequence[Sequence[Any]]) -> bool:
    """Whether ``actual`` reproduces ``expected``, allowing only tied items to trade places.

    Scores must agree rank by rank, and every expected item must be present
    with its own score. ``actual`` may run past ``expected`` — callers fetch
    ``CANARY_SLACK`` extra results — so an item that a tie pushed across the
    limit is still found rather than excused. Nothing is exempt from the
    identity check: an all-tied result with the wrong documents fails.
    """

    if len(actual) < len(expected):
        return False
    for (_, want), (_, got) in zip(expected, actual[: len(expected)], strict=True):
        if not math.isclose(float(want), float(got), rel_tol=_TIE, abs_tol=_TIE):
            return False
    served = {str(item): float(score) for item, score in actual}
    return all(
        str(item) in served
        and math.isclose(served[str(item)], float(score), rel_tol=_TIE, abs_tol=_TIE)
        for item, score in expected
    )


def build_collection(
    kind: str,
    documents: Callable[[], Iterable[IndexDocument]],
    *,
    client: QdrantClient,
    collection: str,
    statistics_documents: Callable[[], Iterable[IndexDocument]] | None = None,
    model: VectorModel,
    settings: CollectionSettings,
    config: IndexConfig,
    vocabulary_path: Path,
    cache: EmbeddingCache | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Two streaming passes: fit the BM25 statistics, then encode and upsert.

    Upserts are idempotent by point id, so rerunning after a crash rewrites the
    same points; the embedding cache makes the rerun cheap.
    """

    started = time.monotonic()
    vocabulary = BM25Vocabulary.fit(
        document.text for document in (statistics_documents or documents)()
    )
    vocabulary_sha256 = vocabulary.save(vocabulary_path)
    total = vocabulary.statistics.documents
    ensure_collection(client, collection, kind, settings, model.dimensions)

    step = max(1, total // max(1, config.canary_queries))
    canary_positions = set(range(0, total, step)[: config.canary_queries])
    canary_sources: list[IndexDocument] = []
    digest = hashlib.sha256()
    points = 0
    truncated = 0
    encode_seconds = 0.0
    upsert_wait = 0.0

    def upload(batch: list[IndexDocument], vectors: Matrix) -> None:
        # Sparse encoding is pure Python and holds the GIL, while the model's
        # forward pass releases it, so both overlap the next batch's encoding.
        client.upsert(
            collection,
            points=[
                _point(document, vector, vocabulary.encode_document(document.text))
                for document, vector in zip(batch, vectors, strict=True)
            ],
            wait=True,
        )

    # One upload in flight at a time: the GPU encodes batch n+1 while batch n
    # is written. A failed upload surfaces at the next wait and fails the build.
    with ThreadPoolExecutor(max_workers=1) as uploader:
        pending: Future[None] | None = None
        for batch in _batched(documents(), config.batch_points):
            tick = time.monotonic()
            encoded = _encode(model, batch, settings.max_tokens, cache)
            encode_seconds += time.monotonic() - tick
            tick = time.monotonic()
            if pending is not None:
                pending.result()
            upsert_wait += time.monotonic() - tick
            pending = uploader.submit(upload, batch, encoded.vectors)
            for offset, document in enumerate(batch):
                digest.update(f"{document.id}\t{document.text_sha256}\n".encode())
                if points + offset in canary_positions:
                    canary_sources.append(document)
            points += len(batch)
            truncated += encoded.truncated
            if progress is not None:
                progress(
                    {
                        "kind": kind,
                        "points": points,
                        "total": total,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                )
        tick = time.monotonic()
        if pending is not None:
            pending.result()
        upsert_wait += time.monotonic() - tick
    if points != total:
        # The statistics pass and this one read the source separately; if they
        # disagree the BM25 weights describe a corpus that was not indexed.
        raise IndexBuildError("source_changed_between_passes", f"{total} then {points}")

    if settings.defer_indexing:
        client.update_collection(
            collection, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20_000)
        )

    canaries = []
    for source in canary_sources:
        query = _canary_query(source, config.canary_terms)
        dense = [float(value) for value in model.encode_array([query]).vectors[0]]
        sparse = vocabulary.encode_query(query)
        sparse_query = models.SparseVector(indices=list(sparse.indices), values=list(sparse.values))
        canaries.append(
            {
                "source_id": source.id,
                "dense_query": dense,
                "sparse_query": {"indices": list(sparse.indices), "values": list(sparse.values)},
                "dense": _exact_search(client, collection, dense, DENSE, config.canary_limit),
                "sparse": _exact_search(
                    client, collection, sparse_query, SPARSE, config.canary_limit
                )
                if sparse.indices
                else [],
            }
        )

    statistics = vocabulary.statistics
    return {
        "collection": collection,
        "points": points,
        "digest": digest.hexdigest(),
        "dimensions": model.dimensions,
        "model": model.identity,
        "max_tokens": settings.max_tokens,
        "truncated": truncated,
        "vocabulary": {
            "path": str(vocabulary_path),
            "sha256": vocabulary_sha256,
            "terms": statistics.vocabulary_size,
            "documents": statistics.documents,
            "average_length": statistics.average_length,
            "k1": statistics.k1,
            "b": statistics.b,
        },
        "canaries": canaries,
        "seconds": {
            "total": round(time.monotonic() - started, 1),
            "encode": round(encode_seconds, 1),
            # Time the encoder waited on the uploader; zero means fully overlapped.
            "upsert_wait": round(upsert_wait, 1),
        },
        "cache": {"hits": cache.hits, "misses": cache.misses} if cache is not None else None,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(
    manifest: str | Path,
    model: VectorModel,
    *,
    engine: Engine,
    client: QdrantClient,
    prefix: str,
    data_dir: str | Path,
    config: IndexConfig,
    kinds: Sequence[str] = KINDS,
    release_id: str | None = None,
    limit: int | None = None,
    cache: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Stage a release for a snapshot and build the requested collections into it.

    Never activates: activation is a separate, explicit step that validates the
    whole pair first. ``limit`` builds a measurement index over the first N
    documents of each collection, in the id order a full build uses, so its
    batches are a prefix of the full build's and the embedding cache carries
    them over. A limited build is marked as such and can never activate.
    """

    manifest_path = Path(manifest).resolve()
    snapshot = validate_manifest(manifest_path)
    if int(snapshot.get("schema_version", 0)) < 2:
        # Version 1 snapshots carry no chunk identities to bind a chunk index to.
        raise IndexBuildError("snapshot_too_old", str(snapshot.get("schema_version")))
    unknown = [kind for kind in kinds if kind not in KINDS]
    if unknown or not kinds:
        raise IndexBuildError("collection_kind_unknown", ",".join(unknown))
    release_id = release_id or str(snapshot["run_id"])
    names = collection_names(prefix, release_id)
    stage_release(
        engine,
        release_id,
        paper_collection=names[PAPERS],
        chunk_collection=names[CHUNKS],
        model_revision=model.identity,
        manifest_sha256=_file_sha256(manifest_path),
        counts={"manifest": str(manifest_path)},
    )
    release = load_release(engine, release_id)
    if release.status != "staged":
        raise ReleaseError("release_not_staged", release_id)

    root = Path(data_dir)
    embedding_cache = (
        EmbeddingCache(root / "indexes" / "cache" / "embeddings", model.identity) if cache else None
    )
    factories: dict[str, Callable[[], Iterable[IndexDocument]]] = {
        PAPERS: lambda: snapshot_papers(manifest_path, limit=limit),
        CHUNKS: lambda: database_chunks(engine, limit=limit),
    }
    statistics: dict[str, Callable[[], Iterable[IndexDocument]]] = {
        PAPERS: factories[PAPERS],
        CHUNKS: lambda: database_chunks(engine, limit=limit, ordered=False),
    }
    for kind in kinds:
        details = build_collection(
            kind,
            factories[kind],
            client=client,
            collection=names[kind],
            statistics_documents=statistics[kind],
            model=model,
            settings=config.collections[kind],
            config=config,
            vocabulary_path=root / "indexes" / release_id / f"{kind}-bm25.parquet",
            cache=embedding_cache,
            progress=progress,
        )
        record_collection_build(engine, release_id, kind, {**details, "limit": limit})
    return release_id


_EXPECTED_COUNTS = {PAPERS: "papers", CHUNKS: "chunks_total"}


def release_problems(
    release: ReleaseRecord, *, client: QdrantClient, model_identity: str, dimensions: int
) -> list[str]:
    """Everything that stops this release serving; an empty list means it may."""

    problems: list[str] = []
    if release.model_revision != model_identity:
        problems.append(f"model {release.model_revision} != configured {model_identity}")
    manifest_path = Path(str(release.counts.get("manifest", "")))
    try:
        snapshot = validate_manifest(manifest_path)
    except Exception as error:  # noqa: BLE001 - every failure is a reason not to serve
        return [*problems, f"snapshot unusable: {error}"]
    if _file_sha256(manifest_path) != release.manifest_sha256:
        problems.append("snapshot manifest changed since staging")

    for kind in KINDS:
        name = release.collection(kind)
        build = release.counts.get(kind)
        if not isinstance(build, Mapping):
            problems.append(f"{kind} collection not built")
            continue
        if build.get("limit") is not None:
            problems.append(f"{name}: measurement build of the first {build['limit']} documents")
        if not client.collection_exists(name):
            problems.append(f"{name} missing")
            continue
        problems.extend(_check_shape(client, name, dimensions))
        expected = int(snapshot["counts"][_EXPECTED_COUNTS[kind]])
        stored = client.count(name, exact=True).count
        if stored != expected or build.get("points") != expected:
            problems.append(
                f"{name}: {stored} points, built {build.get('points')}, snapshot {expected}"
            )
        if build.get("digest") != snapshot.get("digests", {}).get(kind):
            problems.append(f"{name}: indexed content does not match the snapshot digest")
        vocabulary = build.get("vocabulary") or {}
        vocabulary_path = Path(str(vocabulary.get("path", "")))
        if not vocabulary_path.is_file() or _file_sha256(vocabulary_path) != vocabulary.get(
            "sha256"
        ):
            problems.append(f"{name}: BM25 statistics missing or altered")
        for number, canary in enumerate(build.get("canaries") or []):
            dense = _exact_search(
                client, name, canary["dense_query"], DENSE, len(canary["dense"]) + CANARY_SLACK
            )
            if not same_ranking(canary["dense"], dense):
                problems.append(f"{name}: dense canary {number} ranking changed")
            if canary["sparse"]:
                query = models.SparseVector(**canary["sparse_query"])
                sparse = _exact_search(
                    client, name, query, SPARSE, len(canary["sparse"]) + CANARY_SLACK
                )
                if not same_ranking(canary["sparse"], sparse):
                    problems.append(f"{name}: sparse canary {number} ranking changed")
    return problems


def index_validator(client: QdrantClient, *, model_identity: str, dimensions: int) -> Validator:
    """The activation gate: refuse, with every reason, unless the pair validates."""

    def validate(row: Mapping[str, Any]) -> bool:
        release = ReleaseRecord(
            id=str(row["id"]),
            manifest_sha256=str(row.get("manifest_sha256", "")),
            paper_collection=str(row["paper_collection"]),
            chunk_collection=str(row["chunk_collection"]),
            model_revision=str(row["model_revision"]),
            status=str(row.get("status", "")),
            counts=dict(row.get("counts") or {}),
        )
        problems = release_problems(
            release, client=client, model_identity=model_identity, dimensions=dimensions
        )
        if problems:
            raise ReleaseError("index_validation_failed", "; ".join(problems))
        return True

    return validate


def wait_until_indexed(client: QdrantClient, collection: str, *, timeout: float = 7200.0) -> str:
    """Block until Qdrant's optimizers finish, so measurements see the final layout."""

    deadline = time.monotonic() + timeout
    while True:
        status = client.get_collection(collection).status
        if status == models.CollectionStatus.GREEN or time.monotonic() > deadline:
            return str(status)
        time.sleep(2.0)
