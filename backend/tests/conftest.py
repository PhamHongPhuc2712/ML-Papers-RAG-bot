"""Shared fixtures for the backend integration suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import text

from copilot.app import create_app
from copilot.config import Settings
from copilot.db.session import (
    assert_safe_test_database,
    cleanup_test_namespace,
    cleanup_test_vectors,
    make_engine,
    migrate_database,
)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required for integration tests; missing services never skip")
    return value


@pytest.fixture(scope="session")
def test_settings() -> Iterator[Settings]:
    """Load the explicitly configured test services; never silently skip them."""

    database_url = _required("TEST_DATABASE_URL")
    qdrant_url = _required("TEST_QDRANT_URL")
    collection_prefix = _required("TEST_QDRANT_COLLECTION_PREFIX")
    # Test files live only under <DATA_DIR>/test, the same subdirectory the Compose
    # test profile bind-mounts, so destructive cleanup can never reach developer data.
    test_root = Path(_required("DATA_DIR").rstrip("\\/")) / "test"
    test_root.mkdir(parents=True, exist_ok=True)
    # pydantic-settings lets an environment alias win over a same-named init keyword,
    # so DATA_DIR itself must name the test subdirectory for the whole session.
    patch = pytest.MonkeyPatch()
    patch.setenv("DATA_DIR", str(test_root))
    patch.delenv("COPILOT_DATA_DIR", raising=False)
    try:
        yield Settings(
            _env_file=None,
            environment="test",
            data_dir=test_root,
            database_url=database_url,
            qdrant_url=qdrant_url,
            qdrant_collection_prefix=collection_prefix,
            secret_key="test-only-secret",
            model_mode="mock",
        )
    finally:
        patch.undo()


@pytest.fixture(scope="session")
def migrated_database(test_settings: Settings) -> Iterator[object]:
    """Run the migrations once and dispose the engine after the suite."""

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
def mailbox() -> object:
    """Mail the app would have sent (sign-in codes), kept instead of sent."""

    from copilot.identity.mail import RecordingSender

    return RecordingSender()


@pytest.fixture
def sign_in_by_code(mailbox) -> object:
    """Ask for a code, read it from the mailbox, and exchange it: the token response."""

    import re

    def sign_in(client: TestClient, email: str) -> object:
        asked = client.post("/v1/auth/code", json={"email": email})
        assert asked.status_code == 202, asked.text
        code = re.search(r"\b(\d{6})\b", mailbox.sent[-1].body).group(1)
        return client.post("/v1/auth/token", json={"email": email, "code": code})

    return sign_in


@pytest.fixture
def api_client(
    test_settings: Settings,
    migrated_database: object,
    test_qdrant: QdrantClient,
    mailbox: object,
) -> Iterator[TestClient]:
    """Yield a fresh app/client and clean only the configured test namespace."""

    app = create_app(
        test_settings,
        {"db_engine": migrated_database, "qdrant_client": test_qdrant, "mail_sender": mailbox},
    )
    with TestClient(app) as client:
        yield client
    cleanup_test_namespace(test_settings, migrated_database)
    cleanup_test_vectors(test_settings, test_qdrant)
    with migrated_database.begin() as connection:  # type: ignore[attr-defined]
        connection.execute(text("delete from login_codes"))


PASSWORD = "correct horse battery staple"


@pytest.fixture
def make_user(test_settings: Settings, migrated_database) -> Iterator[object]:
    """Create accounts for one test and delete them (and their chats) afterwards."""

    from copilot.identity.users import UserStore

    store = UserStore(migrated_database)
    created: list[UUID] = []

    def make(email: str | None = None, password: str = PASSWORD) -> tuple[UUID, str]:
        address = email or f"user-{uuid4().hex[:12]}@example.org"
        user_id = store.create(address, password)
        created.append(user_id)
        return user_id, address.strip().lower()

    yield make
    assert_safe_test_database(test_settings)
    with migrated_database.begin() as connection:
        connection.execute(text("delete from users where id = any(:ids)"), {"ids": created})


@pytest.fixture
def auth_headers(test_settings: Settings) -> object:
    from copilot.identity.tokens import TokenIssuer

    issuer = TokenIssuer(test_settings.secret_key, ttl_seconds=600)

    def headers(user_id: UUID) -> dict[str, str]:
        return {"Authorization": f"Bearer {issuer.issue(user_id).token}"}

    return headers
