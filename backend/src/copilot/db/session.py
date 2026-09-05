"""Engine, migration, and isolated test namespace helpers."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .models import ActiveRelease, CorpusRelease

if TYPE_CHECKING:
    from copilot.config import Settings


def normalize_database_url(value: str) -> str:
    """Ensure PostgreSQL URLs use the installed psycopg 3 driver."""

    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value.removeprefix("postgres://")
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value.removeprefix("postgresql://")
    if value.startswith("postgresql+psycopg://"):
        return value
    raise ValueError("database_url_must_use_postgresql")


def make_engine(settings_or_url: Settings | str, *, echo: bool = False) -> Engine:
    """Create a synchronous PostgreSQL engine for the API or test harness."""

    if isinstance(settings_or_url, str):
        database_url = settings_or_url
        timeout = 2.0
    else:
        database_url = settings_or_url.database_url
        timeout = settings_or_url.ready_timeout_seconds

    return create_engine(
        normalize_database_url(database_url),
        echo=echo,
        future=True,
        pool_pre_ping=True,
        pool_timeout=timeout,
        connect_args={"connect_timeout": max(1, int(timeout))},
    )


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a caller-owned session factory bound to ``engine``."""

    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine) -> Generator[Session, None, None]:
    """Commit successful work and roll back failed work for a caller-owned session."""

    factory = session_factory(engine)
    with factory() as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise


def _alembic_config() -> Config:
    backend_root = Path(__file__).resolve().parents[3]
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "migrations"))
    return config


def migrate_database(engine: Engine, revision: str = "head") -> None:
    """Apply or roll back migrations using an existing connection and transaction."""

    config = _alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        if revision in {"base", "0"}:
            command.downgrade(config, revision)
        else:
            command.upgrade(config, revision)


def get_active_release(session: Session) -> CorpusRelease | None:
    """Load the release selected by the singleton pointer, if one exists."""

    statement = (
        select(CorpusRelease)
        .join(ActiveRelease, ActiveRelease.corpus_release_id == CorpusRelease.id)
        .where(ActiveRelease.singleton_key == "active")
    )
    return session.execute(statement).scalar_one_or_none()


def assert_safe_test_database(settings_or_url: Settings | str) -> None:
    """Reject destructive cleanup unless both database and namespace are test-scoped."""

    if isinstance(settings_or_url, str):
        database_url = settings_or_url
        collection_prefix = None
    else:
        database_url = settings_or_url.database_url
        collection_prefix = settings_or_url.qdrant_collection_prefix

    database_name = urlparse(database_url).path.removeprefix("/").split("?", 1)[0]
    if not database_name.startswith("test_"):
        raise ValueError("refusing destructive cleanup outside a test_ database")
    if collection_prefix is not None and not collection_prefix.startswith("test_"):
        raise ValueError("refusing destructive cleanup outside a test_ collection namespace")


def cleanup_test_namespace(settings_or_url: Settings | str, engine: Engine) -> None:
    """Delete foundation rows only from a database explicitly named with ``test_``."""

    assert_safe_test_database(settings_or_url)
    with session_scope(engine) as session:
        session.execute(delete(ActiveRelease))
        session.execute(delete(CorpusRelease))


def cleanup_test_vectors(settings: Settings, client: object) -> None:
    """Delete only Qdrant collections in the configured test namespace."""

    assert_safe_test_database(settings)
    get_collections = getattr(client, "get_collections", None)
    delete_collection = getattr(client, "delete_collection", None)
    if not callable(get_collections) or not callable(delete_collection):
        raise TypeError("qdrant_client_missing_collection_methods")

    result = get_collections()
    collections = getattr(result, "collections", result)
    if not isinstance(collections, list | tuple | set):
        raise TypeError("qdrant_client_returned_invalid_collections")
    for item in collections:
        name: object
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name")
        else:
            name = getattr(item, "name", None)
        if isinstance(name, str) and name.startswith(settings.qdrant_collection_prefix):
            delete_collection(name)


def safe_error_code(error: BaseException) -> str:
    """Classify DB errors without exposing connection details to API callers."""

    if isinstance(error, SQLAlchemyError):
        return "database_unavailable"
    return "database_unavailable"
