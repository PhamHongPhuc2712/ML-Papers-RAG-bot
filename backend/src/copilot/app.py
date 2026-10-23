"""FastAPI application composition and health endpoints."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from .chat.api import ChatApi
from .chat.api import router as chat_router
from .chat.config import load_chat_config
from .chat.limits import usage as answer_usage
from .chat.pipeline import TurnPipeline
from .chat.store import ChatStore
from .config import Settings
from .corpus.api import router as corpus_router
from .db.models import CorpusRelease
from .db.session import get_active_release, make_engine
from .identity.api import Identity
from .identity.api import router as identity_router
from .identity.codes import CodeStore
from .identity.mail import MailSender, SmtpSender
from .identity.tokens import TokenIssuer
from .identity.users import UserStore
from .observability import (
    install_request_log,
    send_app_logs_to_stderr,
    send_request_log_to_stdout,
)
from .search.api import SearchApi, install_error_handlers
from .search.api import router as search_router
from .search.client import qdrant_client
from .search.service import SearchService

logger = logging.getLogger(__name__)


def _typed_error(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": {"code": code, "message": message}})


def _check_database(engine: Engine, settings: Settings) -> CorpusRelease | None:
    """Run a bounded connectivity probe and active-release lookup."""

    timeout_ms = max(1, int(settings.ready_timeout_seconds * 1000))
    with engine.connect() as connection:
        with connection.begin():
            # The integer is derived from validated local configuration, never user input.
            connection.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))
            connection.execute(text("SELECT 1"))
            with Session(bind=connection, autoflush=False, expire_on_commit=False) as session:
                return get_active_release(session)


def _qdrant_client(settings: Settings) -> QdrantClient:
    return qdrant_client(
        settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=max(1, int(settings.ready_timeout_seconds)),
    )


def _collection_names(result: object) -> set[str]:
    collections = getattr(result, "collections", result)
    names: set[str] = set()
    if isinstance(collections, list | tuple | set):
        for item in collections:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, Mapping):
                name = item.get("name")
                if isinstance(name, str):
                    names.add(name)
            else:
                name = getattr(item, "name", None)
                if isinstance(name, str):
                    names.add(name)
    return names


def _check_vectors(client: object, release: CorpusRelease) -> bool:
    """Check Qdrant health and the two collections named by the active release."""

    collection_exists = getattr(client, "collection_exists", None)
    if callable(collection_exists):
        return bool(collection_exists(release.paper_collection)) and bool(
            collection_exists(release.chunk_collection)
        )

    get_collections = getattr(client, "get_collections", None)
    if callable(get_collections):
        return {
            release.paper_collection,
            release.chunk_collection,
        }.issubset(_collection_names(get_collections()))

    healthz = getattr(client, "healthz", None)
    if callable(healthz):
        healthz()
        return True
    return False


def create_app(settings: Settings, overrides: dict[str, object] | None = None) -> FastAPI:
    """Create a fresh API instance with optional test/service dependency overrides."""

    app_overrides: dict[str, object] = dict(overrides or {})
    engine = cast(Engine | None, app_overrides.get("db_engine") or app_overrides.get("engine"))
    owns_engine = engine is None
    if engine is None:
        engine = make_engine(settings)

    clock = app_overrides.get("clock")
    search_api = SearchApi(
        settings=settings,
        engine=engine,
        client=cast(QdrantClient | None, app_overrides.get("qdrant_client")),
        service=cast(SearchService | None, app_overrides.get("search_service")),
        models_path=str(app_overrides.get("models_config") or "configs/models.yaml"),
        search_path=str(app_overrides.get("search_config") or "configs/search.yaml"),
        clock=cast(Callable[[], int], clock) if callable(clock) else None,
    )
    chat_api = ChatApi(
        settings=settings,
        engine=engine,
        search=search_api,
        pipeline=cast(TurnPipeline | None, app_overrides.get("chat_pipeline")),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Real models load here, once, before the first request is accepted.
        search_api.start()
        chat_api.start()
        try:
            yield
        finally:
            chat_api.close()
            search_api.close()
            if owns_engine:
                engine.dispose()

    app = FastAPI(
        title="ML Research Copilot",
        version="0.1.0",
        lifespan=lifespan,
    )
    artifacts = app_overrides.get("artifacts_config") or "configs/artifacts.yaml"
    app.include_router(corpus_router(engine, str(artifacts)))
    app.include_router(search_router())
    app.include_router(identity_router())
    # Tests set the identity clock (code expiry) and record mail instead of sending it.
    identity_clock = cast(
        Callable[[], datetime],
        app_overrides.get("identity_clock") or (lambda: datetime.now(UTC)),
    )
    mail = cast(MailSender | None, app_overrides.get("mail_sender"))
    if mail is None and settings.smtp_host:
        mail = SmtpSender(settings.smtp_host, settings.smtp_port, settings.mail_from)
    app.state.identity = Identity(
        UserStore(engine, identity_clock),
        TokenIssuer(settings.secret_key, settings.access_token_ttl_seconds),
        codes=CodeStore(engine, settings.secret_key, identity_clock),
        mail=mail,
        usage=partial(answer_usage, engine, limit=load_chat_config().daily_answers),
    )
    app.include_router(chat_router())
    app.state.chat_store = ChatStore(engine)
    install_error_handlers(app)
    install_request_log(app)
    app.state.search_api = search_api
    app.state.chat_api = chat_api
    app.state.settings = settings
    app.state.overrides = app_overrides
    app.state.db_engine = engine

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", response_model=None)
    def ready() -> JSONResponse | dict[str, str]:
        try:
            checker = app_overrides.get("database_checker")
            if callable(checker):
                active_release = cast(CorpusRelease | None, checker(engine))
            else:
                active_release = _check_database(engine, settings)
        except Exception as error:  # noqa: BLE001 - readiness must become a typed 503
            logger.debug("database readiness check failed", exc_info=error)
            return _typed_error("database_unavailable", "database is unavailable")

        if active_release is None:
            return _typed_error("corpus_not_ready", "no active corpus release")

        try:
            vector_checker = app_overrides.get("vector_checker")
            if callable(vector_checker):
                vectors_ready = bool(vector_checker(active_release))
            else:
                qdrant = app_overrides.get("qdrant_client")
                if qdrant is None:
                    qdrant = _qdrant_client(settings)
                vectors_ready = _check_vectors(qdrant, active_release)
        except Exception as error:  # noqa: BLE001 - readiness must become a typed 503
            logger.debug("vector readiness check failed", exc_info=error)
            return _typed_error("vector_unavailable", "vector service is unavailable")

        if not vectors_ready:
            return _typed_error("vector_not_ready", "active corpus vectors are unavailable")
        return {"status": "ok", "corpus_release_id": active_release.id}

    return app


def openapi_schema() -> dict[str, Any]:
    """The public HTTP contract, generated without a database, model or secret.

    ``openapi()`` never runs the lifespan, so nothing connects or loads; the
    settings exist only because the app is built from them.
    """

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, data_dir=Path(tempfile.gettempdir()).resolve() / "copilot-schema"
    )
    return create_app(settings).openapi()


def create_default_app() -> FastAPI:
    """Uvicorn factory entry point using environment-backed settings."""

    # Required fields such as DATA_DIR arrive from the environment at runtime; mypy
    # cannot see pydantic-settings sources, so the bare call is unsatisfiable statically.
    send_request_log_to_stdout()
    send_app_logs_to_stderr()
    return create_app(Settings())  # type: ignore[call-arg]
