"""Shared fixtures for the backend integration suite."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from copilot.app import create_app
from copilot.config import Settings
from copilot.db.session import (
    cleanup_test_namespace,
    cleanup_test_vectors,
    make_engine,
    migrate_database,
)


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Load the explicitly configured test services; never silently skip them."""

    database_url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    qdrant_url = os.environ.get("TEST_QDRANT_URL") or os.environ.get("QDRANT_URL")
    if not database_url:
        pytest.fail("TEST_DATABASE_URL or DATABASE_URL is required for integration tests")
    if not qdrant_url:
        pytest.fail("TEST_QDRANT_URL or QDRANT_URL is required for integration tests")
    return Settings(
        environment="test",
        database_url=database_url,
        qdrant_url=qdrant_url,
        qdrant_collection_prefix=os.environ.get("TEST_QDRANT_COLLECTION_PREFIX", "test_"),
        secret_key="test-only-secret",
        model_mode="mock",
    )


@pytest.fixture(scope="session")
def migrated_database(test_settings: Settings) -> Iterator[object]:
    """Run the foundation migration once and dispose its engine after the suite."""

    engine = make_engine(test_settings)
    migrate_database(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="session")
def test_qdrant(test_settings: Settings) -> Iterator[QdrantClient]:
    """Connect to the configured Qdrant service and fail if it is unavailable."""

    client = QdrantClient(
        url=test_settings.qdrant_url,
        api_key=test_settings.qdrant_api_key,
        timeout=max(1, int(test_settings.ready_timeout_seconds)),
    )
    client.get_collections()
    try:
        yield client
    finally:
        cleanup_test_vectors(test_settings, client)


@pytest.fixture
def api_client(
    test_settings: Settings,
    migrated_database: object,
    test_qdrant: QdrantClient,
) -> Iterator[TestClient]:
    """Yield a fresh app/client and clean only the configured test namespace."""

    app = create_app(
        test_settings,
        {"db_engine": migrated_database, "qdrant_client": test_qdrant},
    )
    with TestClient(app) as client:
        yield client
    cleanup_test_namespace(test_settings, migrated_database)
    cleanup_test_vectors(test_settings, test_qdrant)
