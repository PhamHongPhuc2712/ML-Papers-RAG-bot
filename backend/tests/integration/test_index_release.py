"""Index lifecycle against real test services: build, validate, switch, roll back, restore."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text

from copilot.app import create_app
from copilot.config import Settings
from copilot.contracts import PaperFilters
from copilot.corpus.export import export_snapshot
from copilot.corpus.releases import (
    ReleaseError,
    activate_release,
    capture_release,
    load_release,
)
from copilot.db.session import assert_safe_test_database, cleanup_test_vectors
from copilot.models.embeddings import EncodedBatch, FixtureEmbedding
from copilot.search.dense import DenseRetriever
from copilot.search.index import (
    CHUNKS,
    KINDS,
    PAPERS,
    CollectionSettings,
    IndexBuildError,
    IndexConfig,
    build_index,
    index_validator,
    release_problems,
    same_ranking,
)
from copilot.search.lexical import BM25, paper_text
from copilot.search.sparse import SparseRetriever

pytestmark = pytest.mark.integration

TABLES = (
    "chunks",
    "paper_versions",
    "paper_identifiers",
    "paper_authors",
    "authors",
    "papers",
    "venues",
)
CHUNK_TOKENS = 65
CONFIG = IndexConfig(
    collections={
        PAPERS: CollectionSettings(max_tokens=64),
        # The production chunk layout: on disk, int8 in RAM, HNSW deferred.
        CHUNKS: CollectionSettings(
            max_tokens=CHUNK_TOKENS, on_disk=True, quantization="int8", defer_indexing=True
        ),
    },
    batch_points=2,
    canary_queries=3,
    canary_limit=3,
    canary_terms=4,
)
# (venue, year, title, abstract, parse status, chunk texts)
PAPERS_SEED = (
    ("ICLR", 2024, "Graph neural networks for molecules", "Message passing on molecular graphs.",
     "parsed", ("We pass messages along bonds.", "Results on molecule benchmarks.")),
    ("ICLR", 2023, "Contrastive learning of visual representations", "Augmented views agree.",
     "parsed", ("Two augmented views of an image.", "Linear evaluation protocol.")),
    ("ACL", 2024, "Graph contrastive learning for text", "Graphs of words, contrasted.",
     "parsed", ("Word graphs are built from co-occurrence.",)),
    ("ACL", 2023, "Retrieval augmented generation", "Retrieve passages, then generate.",
     "oversized", ()),
    ("NeurIPS", 2024, "Diffusion models for image synthesis", "Denoising generates images.",
     "parsed", ("A forward noising process.", "A learned reverse process.")),
    ("NeurIPS", 2025, "Scaling laws for language models", None, "pending", ()),
)


@dataclass
class Corpus:
    engine: Engine
    client: QdrantClient
    settings: Settings
    ids: dict[str, str]
    manifest: Path
    root: Path


def _reset(engine: Engine, settings: Settings, client: QdrantClient) -> None:
    assert_safe_test_database(settings)
    with engine.begin() as connection:
        connection.execute(text("delete from active_release"))
        connection.execute(text("delete from corpus_releases"))
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
    cleanup_test_vectors(settings, client)


def _add_paper(
    engine: Engine,
    venue: str,
    year: int,
    title: str,
    abstract: str | None,
    status: str,
    chunks: tuple[str, ...],
) -> str:
    paper_id = str(uuid4())
    with engine.begin() as connection:
        venue_id = connection.execute(
            text(
                "insert into venues (id, name, track) values (gen_random_uuid(), :name, 'main')"
                " on conflict (name, track) do update set name = excluded.name returning id"
            ),
            {"name": venue},
        ).scalar_one()
        connection.execute(
            text(
                "insert into papers (id, title, abstract, publication_year, venue_id,"
                " first_seen_at, acceptance_type, metadata_status) values (:id, :title,"
                " :abstract, :year, :venue, now(), 'poster', 'active')"
            ),
            {"id": paper_id, "title": title, "abstract": abstract, "year": year, "venue": venue_id},
        )
        version = connection.execute(
            text(
                "insert into paper_versions (id, paper_id, source, source_url, source_revision,"
                " content_sha256, redistribution, parser_version, parse_status) values"
                " (gen_random_uuid(), :paper, 'papercli', 'u', 'rev', :sha, 'unknown',"
                " 'pypdf-text-v2', :status) returning id"
            ),
            {"paper": paper_id, "sha": paper_id.replace("-", "")[:32] * 2, "status": status},
        ).scalar_one()
        for ordinal, chunk in enumerate(chunks):
            connection.execute(
                text(
                    "insert into chunks (id, paper_version_id, section_path, section_ordinal,"
                    " kind, ordinal, text, token_count, parser_version, chunker_version) values"
                    " (gen_random_uuid(), :version, 'Method', 0, 'body', :ordinal, :text, 8,"
                    " 'pypdf-text-v2', 'paragraph-pack-v1')"
                ),
                {"version": version, "ordinal": ordinal, "text": chunk},
            )
    return paper_id


def _export(corpus: Corpus, run_id: str) -> Path:
    directory = corpus.root / "exports" / run_id
    export_snapshot(run_id, directory, engine=corpus.engine)
    return directory / "manifest.json"


@pytest.fixture
def corpus(migrated_database, test_settings, test_qdrant, tmp_path) -> Iterator[Corpus]:
    # The shared client carries the readiness probe's 2 s bound; creating and
    # snapshotting on-disk collections is build work and legitimately slower.
    client = QdrantClient(url=test_settings.qdrant_url, timeout=60)
    _reset(migrated_database, test_settings, client)
    ids = {
        seed[2]: _add_paper(migrated_database, *seed)  # type: ignore[arg-type]
        for seed in PAPERS_SEED
    }
    state = Corpus(migrated_database, client, test_settings, ids, Path(), tmp_path)
    state.manifest = _export(state, "rel-a")
    try:
        yield state
    finally:
        client.close()


def _build(
    corpus: Corpus,
    *,
    release: str = "rel-a",
    manifest: Path | None = None,
    kinds: tuple[str, ...] = KINDS,
    model: FixtureEmbedding | None = None,
    limit: int | None = None,
) -> str:
    return build_index(
        manifest or corpus.manifest,
        model or FixtureEmbedding(),
        engine=corpus.engine,
        client=corpus.client,
        prefix=corpus.settings.qdrant_collection_prefix,
        data_dir=corpus.root / "data",
        config=CONFIG,
        kinds=kinds,
        release_id=release,
        limit=limit,
    )


def _validator(corpus: Corpus):
    model = FixtureEmbedding()
    return index_validator(corpus.client, model_identity=model.identity, dimensions=2)


def _activate(corpus: Corpus, release: str) -> None:
    activate_release(corpus.engine, release, validator=_validator(corpus))


def _problems(corpus: Corpus, release: str) -> list[str]:
    return release_problems(
        load_release(corpus.engine, release),
        client=corpus.client,
        model_identity=FixtureEmbedding().identity,
        dimensions=2,
    )


def test_a_built_pair_validates_activates_and_makes_the_api_ready(corpus):
    release = _build(corpus)
    record = load_release(corpus.engine, release)
    assert record.status == "staged"
    # Building never activates; a release serves only after explicit validation.
    assert capture_release(corpus.engine) is None
    assert corpus.client.count(record.paper_collection, exact=True).count == 6
    assert corpus.client.count(record.chunk_collection, exact=True).count == 7

    _activate(corpus, release)

    assert capture_release(corpus.engine).id == release
    app = create_app(
        corpus.settings, {"db_engine": corpus.engine, "qdrant_client": corpus.client}
    )
    with TestClient(app) as client:
        ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["corpus_release_id"] == release


def test_rebuilding_the_same_input_upserts_idempotently(corpus):
    release = _build(corpus)
    first = load_release(corpus.engine, release).counts
    _build(corpus)
    second = load_release(corpus.engine, release).counts
    for kind in KINDS:
        assert second[kind]["points"] == first[kind]["points"]
        assert second[kind]["digest"] == first[kind]["digest"]
        assert second[kind]["canaries"] == first[kind]["canaries"]
    assert _problems(corpus, release) == []


def test_filters_exclude_the_wrong_venue_year_and_availability(corpus):
    release = _build(corpus)
    dense = DenseRetriever(corpus.engine, corpus.client, FixtureEmbedding())
    sparse = SparseRetriever(corpus.engine, corpus.client)
    ids = corpus.ids
    query = "graph learning models"

    for retriever in (dense, sparse):
        acl = {pid for pid, _ in retriever.search(query, PaperFilters(venues=["ACL"]), 10, release)}
        assert acl and acl <= {ids["Graph contrastive learning for text"],
                               ids["Retrieval augmented generation"]}
        recent = {
            pid for pid, _ in retriever.search(query, PaperFilters(year_from=2024), 10, release)
        }
        assert recent and all(
            pid not in recent
            for pid in (ids["Contrastive learning of visual representations"],
                        ids["Retrieval augmented generation"])
        )
        fulltext = {
            pid
            for pid, _ in retriever.search(query, PaperFilters(fulltext_only=True), 10, release)
        }
        assert ids["Retrieval augmented generation"] not in fulltext
        assert ids["Scaling laws for language models"] not in fulltext


def test_a_query_from_another_model_is_refused(corpus):
    release = _build(corpus)
    other = FixtureEmbedding(identity="fixture/hashed-2d@v2#l2")
    with pytest.raises(IndexBuildError, match="model_revision_mismatch"):
        DenseRetriever(corpus.engine, corpus.client, other).search("graph", None, 5, release)
    # Nor can another model be built into a release staged under this one.
    with pytest.raises(ReleaseError, match="release_conflict"):
        _build(corpus, model=other)
    assert "model" in " ".join(
        release_problems(
            load_release(corpus.engine, release),
            client=corpus.client,
            model_identity=other.identity,
            dimensions=2,
        )
    )


class _NaNModel(FixtureEmbedding):
    def encode_array(self, texts, *, max_tokens=None) -> EncodedBatch:
        batch = super().encode_array(texts, max_tokens=max_tokens)
        return EncodedBatch(np.full_like(batch.vectors, np.nan), batch.truncated)


def test_nonfinite_vectors_are_refused_before_any_write(corpus):
    with pytest.raises(ValueError, match="nonfinite"):
        _build(corpus, kinds=(PAPERS,), model=_NaNModel())
    record = load_release(corpus.engine, "rel-a")
    assert corpus.client.count(record.paper_collection, exact=True).count == 0
    assert PAPERS not in record.counts


class _CrashOnChunks(FixtureEmbedding):
    def encode_array(self, texts, *, max_tokens=None) -> EncodedBatch:
        if max_tokens == CHUNK_TOKENS:
            raise RuntimeError("worker died mid-build")
        return super().encode_array(texts, max_tokens=max_tokens)


def test_a_crash_between_collection_builds_leaves_the_previous_release_serving(corpus):
    _build(corpus)
    _activate(corpus, "rel-a")
    _add_paper(corpus.engine, "ICLR", 2025, "Graph transformers", "Attention on graphs.",
               "parsed", ("Attention over nodes.",))
    manifest_b = _export(corpus, "rel-b")

    with pytest.raises(RuntimeError, match="worker died"):
        _build(corpus, release="rel-b", manifest=manifest_b, model=_CrashOnChunks())

    assert capture_release(corpus.engine).id == "rel-a"
    with pytest.raises(ReleaseError, match="chunks collection not built"):
        _activate(corpus, "rel-b")
    assert capture_release(corpus.engine).id == "rel-a"


def test_requests_in_flight_keep_the_pair_they_captured(corpus):
    _build(corpus)
    _activate(corpus, "rel-a")
    captured = capture_release(corpus.engine)
    new_paper = _add_paper(corpus.engine, "ICLR", 2025, "Graph transformers",
                           "Attention on graphs.", "parsed", ("Attention over nodes.",))
    manifest_b = _export(corpus, "rel-b")
    _build(corpus, release="rel-b", manifest=manifest_b)
    _activate(corpus, "rel-b")

    dense = DenseRetriever(corpus.engine, corpus.client, FixtureEmbedding())
    query = paper_text("Graph transformers", "Attention on graphs.")
    # The request that captured rel-a finishes on rel-a's collections.
    assert new_paper not in {pid for pid, _ in dense.search(query, None, 10, captured.id)}
    assert dense.search(query, None, 1, "rel-b")[0][0] == new_paper


def test_rollback_restores_both_collections_together(corpus):
    _build(corpus)
    _activate(corpus, "rel-a")
    _add_paper(corpus.engine, "ICLR", 2025, "Graph transformers", "Attention on graphs.",
               "parsed", ("Attention over nodes.",))
    _build(corpus, release="rel-b", manifest=_export(corpus, "rel-b"))
    _activate(corpus, "rel-b")
    assert load_release(corpus.engine, "rel-a").status == "superseded"

    _activate(corpus, "rel-a")

    active = capture_release(corpus.engine)
    original = load_release(corpus.engine, "rel-a")
    assert (active.paper_collection, active.chunk_collection) == (
        original.paper_collection,
        original.chunk_collection,
    )
    assert load_release(corpus.engine, "rel-b").status == "superseded"
    assert _problems(corpus, "rel-a") == []


def test_restored_collections_reproduce_the_canary_rankings(corpus):
    release = _build(corpus)
    record = load_release(corpus.engine, release)
    for name in (record.paper_collection, record.chunk_collection):
        snapshot = corpus.client.create_snapshot(name, wait=True)
        corpus.client.delete_collection(name)
        corpus.client.recover_snapshot(
            name, location=f"file:///qdrant/snapshots/{name}/{snapshot.name}", wait=True
        )
        corpus.client.delete_snapshot(name, snapshot.name, wait=True)

    assert _problems(corpus, release) == []
    _activate(corpus, release)


def test_sparse_ranking_reproduces_the_bm25_oracle(corpus):
    release = _build(corpus, kinds=(PAPERS,))
    oracle = BM25()
    with corpus.engine.connect() as connection:
        oracle.fit(
            {
                str(row.id): paper_text(row.title, row.abstract)
                for row in connection.execute(text("select id, title, abstract from papers"))
            }
        )
    sparse = SparseRetriever(corpus.engine, corpus.client)
    for query in ("graph learning", "contrastive views", "diffusion image synthesis", "zzz"):
        served = sparse.search(query, None, 10, release)
        expected = oracle.search(query, 10) if query != "zzz" else []
        assert same_ranking([list(pair) for pair in expected], [list(pair) for pair in served])


def test_text_that_drifted_since_the_snapshot_refuses_activation(corpus):
    with corpus.engine.begin() as connection:
        connection.execute(
            text("update chunks set text = 'silently edited' where ordinal = 0 and text like 'A%'")
        )
    _build(corpus)
    with pytest.raises(ReleaseError, match="does not match the snapshot digest"):
        _activate(corpus, "rel-a")


def test_a_measurement_build_can_never_activate(corpus):
    _build(corpus, limit=2)
    record = load_release(corpus.engine, "rel-a")
    assert record.counts[PAPERS]["points"] == 2
    assert record.counts[CHUNKS]["points"] == 2
    with pytest.raises(ReleaseError, match="measurement build"):
        _activate(corpus, "rel-a")


def test_a_collection_that_gained_points_fails_validation(corpus):
    release = _build(corpus)
    record = load_release(corpus.engine, release)
    corpus.client.upsert(
        record.paper_collection,
        points=[models.PointStruct(id=str(uuid4()), vector={"dense": [1.0, 0.0]}, payload={})],
        wait=True,
    )
    assert any("7 points" in problem for problem in _problems(corpus, release))


def test_a_measurement_build_is_a_prefix_of_the_full_build(corpus):
    """Its cached batches are the full build's first batches, so none is encoded twice."""

    _build(corpus, release="pilot", limit=4)
    _build(corpus)
    for kind in KINDS:
        cache = load_release(corpus.engine, "rel-a").counts[kind]["cache"]
        # Two full batches of CONFIG.batch_points each were encoded by the pilot.
        assert cache["hits"] >= 2


def test_a_failed_upload_fails_the_build_rather_than_vanishing(corpus, monkeypatch):
    """Uploads run beside the encoder; an error there must still stop the build."""

    real = corpus.client.upsert
    calls: list[int] = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("qdrant went away")
        return real(*args, **kwargs)

    monkeypatch.setattr(corpus.client, "upsert", flaky)
    with pytest.raises(RuntimeError, match="qdrant went away"):
        _build(corpus, kinds=(PAPERS,))
    assert PAPERS not in load_release(corpus.engine, "rel-a").counts
