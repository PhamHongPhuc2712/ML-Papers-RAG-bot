"""Health endpoint acceptance tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Distance, VectorParams

from copilot.app import create_app
from copilot.config import Settings, SettingsError
from copilot.db.models import ActiveRelease, CorpusRelease
from copilot.db.session import session_scope


@pytest.mark.integration
def test_liveness_and_missing_release(api_client):
    assert api_client.get("/health/live").json()["status"] == "ok"
    response = api_client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "corpus_not_ready"


@pytest.mark.integration
def test_liveness_does_not_imply_readiness_when_database_is_unavailable(test_settings):
    settings = test_settings.model_copy(
        update={"database_url": "postgresql+psycopg://invalid:invalid@127.0.0.1:1/test_db"}
    )
    client = create_app(settings, {"qdrant_client": None})

    with TestClient(client) as api:
        assert api.get("/health/live").json() == {"status": "ok"}
        response = api.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_unavailable"
    assert "invalid" not in response.text
    assert "127.0.0.1" not in response.text


@pytest.mark.integration
def test_readiness_requires_vector_collections_after_active_release(
    api_client, migrated_database
):
    release_id = "release-vector-check"
    with session_scope(migrated_database) as session:
        session.add(
            CorpusRelease(
                id=release_id,
                manifest_sha256="a" * 64,
                paper_collection="test_papers_missing",
                chunk_collection="test_chunks_missing",
                model_revision="test-model",
                status="ready",
                counts={"papers": 0, "chunks": 0},
            )
        )
        session.add(ActiveRelease(singleton_key="active", corpus_release_id=release_id))
    response = api_client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] in {"vector_unavailable", "vector_not_ready"}


@pytest.mark.integration
def test_readiness_is_ok_when_active_release_vectors_exist(
    api_client, migrated_database, test_qdrant
):
    release_id = "release-vector-ready"
    paper_collection = "test_papers_ready"
    chunk_collection = "test_chunks_ready"
    for name in (paper_collection, chunk_collection):
        test_qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
    with session_scope(migrated_database) as session:
        session.add(
            CorpusRelease(
                id=release_id,
                manifest_sha256="b" * 64,
                paper_collection=paper_collection,
                chunk_collection=chunk_collection,
                model_revision="test-model",
                status="ready",
                counts={"papers": 1, "chunks": 1},
            )
        )
        session.add(ActiveRelease(singleton_key="active", corpus_release_id=release_id))

    response = api_client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "corpus_release_id": release_id}


def test_production_rejects_mock_model_mode_and_default_secret():
    with pytest.raises(SettingsError):
        Settings(environment="production", model_mode="mock", secret_key="change-me")


def test_production_rejects_default_secret_even_with_real_models():
    with pytest.raises(SettingsError):
        Settings(environment="production", model_mode="real", secret_key="change-me")


@pytest.mark.integration
def test_migration_creates_foundation_tables(migrated_database):
    from sqlalchemy import inspect

    names = set(inspect(migrated_database).get_table_names())
    assert {"alembic_version", "corpus_releases", "active_release"} <= names


@pytest.mark.integration
def test_cleanup_guard_rejects_non_test_database(test_settings):
    from copilot.db.session import assert_safe_test_database

    unsafe = test_settings.model_copy(
        update={"database_url": "postgresql+psycopg://u:p@localhost:5432/copilot"}
    )
    with pytest.raises(ValueError, match="test_"):
        assert_safe_test_database(unsafe)


@pytest.mark.integration
def test_cleanup_guard_rejects_non_test_collection_namespace(test_settings):
    from copilot.db.session import assert_safe_test_database

    unsafe = test_settings.model_copy(update={"qdrant_collection_prefix": "papers_"})
    with pytest.raises(ValueError, match="test_"):
        assert_safe_test_database(unsafe)
