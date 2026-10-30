"""Fixtures shared by the search suites: one small release, built and active."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import Engine, text

from copilot.corpus.export import export_snapshot
from copilot.corpus.releases import activate_release
from copilot.db.session import assert_safe_test_database, cleanup_test_vectors
from copilot.models.embeddings import FixtureEmbedding
from copilot.search.index import (
    CHUNKS,
    PAPERS,
    CollectionSettings,
    IndexConfig,
    build_index,
    index_validator,
)

SEARCH_RELEASE = "search-a"
TABLES = (
    "chunks",
    "paper_versions",
    "paper_identifiers",
    "paper_authors",
    "authors",
    "papers",
    "venues",
)
# (venue, year, title, abstract)
CORPUS = (
    ("ICLR", 2024, "Graph neural networks for molecules", "Message passing on molecular graphs."),
    ("ICLR", 2023, "Contrastive learning of visual representations", "Augmented views agree."),
    ("ACL", 2024, "Graph contrastive learning for text", "Word graphs, contrasted."),
    ("ACL", 2023, "Retrieval augmented generation", "Retrieve passages, then generate."),
    ("NeurIPS", 2024, "Diffusion models for image synthesis", "Denoising generates images."),
    ("NeurIPS", 2023, "Scaling laws for language models", "Loss falls as a power law."),
    ("ICLR", 2024, "Graph transformers at scale", "Attention over graph nodes."),
    ("ACL", 2024, "Contrastive sentence embeddings", "Sentence pairs, contrasted."),
)


@dataclass
class SearchCorpus:
    engine: Engine
    client: QdrantClient
    ids: dict[str, str]
    release: str = SEARCH_RELEASE


def _seed(engine: Engine) -> dict[str, str]:
    ids: dict[str, str] = {}
    with engine.begin() as connection:
        author = connection.execute(
            text(
                "insert into authors (id, name, normalized_name) values"
                " (gen_random_uuid(), 'Ada Lovelace', 'ada lovelace') returning id"
            )
        ).scalar_one()
        for venue, year, title, abstract in CORPUS:
            paper_id = str(uuid4())
            ids[title] = paper_id
            venue_id = connection.execute(
                text(
                    "insert into venues (id, name, track) values (gen_random_uuid(), :name,"
                    " 'main') on conflict (name, track) do update set name = excluded.name"
                    " returning id"
                ),
                {"name": venue},
            ).scalar_one()
            connection.execute(
                text(
                    "insert into papers (id, title, abstract, publication_year, venue_id,"
                    " first_seen_at, acceptance_type, metadata_status) values (:id, :title,"
                    " :abstract, :year, :venue, now(), 'poster', 'active')"
                ),
                {
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "venue": venue_id,
                },
            )
            connection.execute(
                text(
                    "insert into paper_authors (paper_id, position, author_id) values"
                    " (:paper, 0, :author)"
                ),
                {"paper": paper_id, "author": author},
            )
            version = connection.execute(
                text(
                    "insert into paper_versions (id, paper_id, source, source_url,"
                    " source_revision, content_sha256, redistribution, parser_version,"
                    " parse_status) values (gen_random_uuid(), :paper, 'papercli', 'u', 'rev',"
                    " :sha, 'unknown', 'pypdf-text-v2', 'parsed') returning id"
                ),
                {"paper": paper_id, "sha": paper_id.replace("-", "")[:32] * 2},
            ).scalar_one()
            connection.execute(
                text(
                    "insert into chunks (id, paper_version_id, section_path, section_ordinal,"
                    " kind, ordinal, text, token_count, parser_version, chunker_version) values"
                    " (gen_random_uuid(), :version, 'Method', 0, 'body', 0, :text, 8,"
                    " 'pypdf-text-v2', 'paragraph-pack-v1')"
                ),
                {"version": version, "text": abstract},
            )
    return ids


@pytest.fixture(scope="module")
def search_corpus(migrated_database, test_settings, tmp_path_factory) -> Iterator[SearchCorpus]:
    """Seeded, exported, indexed and activated once per module that asks for it."""

    assert_safe_test_database(test_settings)
    client = QdrantClient(url=test_settings.qdrant_url, timeout=60)
    with migrated_database.begin() as connection:
        connection.execute(text("delete from active_release"))
        connection.execute(text("delete from corpus_releases"))
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
    cleanup_test_vectors(test_settings, client)
    ids = _seed(migrated_database)
    root = tmp_path_factory.mktemp("search")
    snapshot = root / "exports" / SEARCH_RELEASE
    export_snapshot(SEARCH_RELEASE, snapshot, engine=migrated_database)
    model = FixtureEmbedding()
    build_index(
        Path(snapshot / "manifest.json"),
        model,
        engine=migrated_database,
        client=client,
        prefix=test_settings.qdrant_collection_prefix,
        data_dir=root / "data",
        config=IndexConfig(
            collections={
                PAPERS: CollectionSettings(max_tokens=64),
                CHUNKS: CollectionSettings(max_tokens=64),
            },
            batch_points=4,
            canary_queries=2,
            canary_limit=3,
            canary_terms=3,
        ),
        release_id=SEARCH_RELEASE,
    )
    activate_release(
        migrated_database,
        SEARCH_RELEASE,
        validator=index_validator(client, model_identity=model.identity, dimensions=2),
    )
    try:
        yield SearchCorpus(migrated_database, client, ids)
    finally:
        client.close()
