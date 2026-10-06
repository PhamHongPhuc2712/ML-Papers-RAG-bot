"""Public search, paper metadata and related-paper endpoints (spec §10).

``POST /v1/search`` ranks through the P2.3 service, keeps the ordering for ten
minutes and pages through it with signed cursors. ``GET /v1/papers/{id}``
returns canonical metadata, following merges to the surviving paper.
``GET /v1/papers/{id}/related`` ranks the seed's dense neighbours in the active
release, the seed itself excluded.

Every success and every error carries a request id, and errors share one
shape: ``{"error": {"code", "message", "retryable"}, "request_id"}``. The
endpoints read only the active release's public collections inside this
deployment's namespace; a release pointing anywhere else is refused, so a test
or development process can never serve another environment's index.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

from ..config import Settings
from ..contracts import ContractModel, PaperFilters, PaperSummary, SearchRequest, SearchResponse
from ..corpus.releases import ReleaseRecord, capture_release
from .cache import CursorError, OrderingCache, decode_cursor, encode_cursor, query_key, ranking_key
from .index import CHUNKS, DENSE, PAPERS, IndexBuildError, collection_names, qdrant_filter
from .service import (
    CandidateRetriever,
    ListwiseReranker,
    Ordering,
    PaperRow,
    PaperStore,
    SearchService,
    SearchUnavailable,
    load_search_config,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS_PATH = Path("configs/models.yaml")
DEFAULT_SEARCH_PATH = Path("configs/search.yaml")
DEFAULT_LLM_PATH = Path("configs/llm.yaml")
DEFAULT_LISTWISE_PROMPT = Path("prompts/rerank/listwise-v1.yaml")


# --- Errors -----------------------------------------------------------------


class ErrorBody(ContractModel):
    code: str
    message: str
    retryable: bool


class ErrorResponse(ContractModel):
    error: ErrorBody
    request_id: UUID


class ApiError(Exception):
    """A typed failure the handler turns into the spec's error shape."""

    def __init__(self, status: int, code: str, message: str, *, retryable: bool = False) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(code)


def request_id(request: Request) -> UUID:
    """This HTTP request's id: made once, shared by the endpoint and any error handler."""

    current = getattr(request.state, "request_id", None)
    if not isinstance(current, UUID):
        current = uuid4()
        request.state.request_id = current
    return current


def error_response(
    request: Request, status: int, code: str, message: str, *, retryable: bool
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, retryable=retryable),
        request_id=request_id(request),
    )
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def _describe(errors: Sequence[Mapping[str, Any]]) -> str:
    """Where and why validation failed — never the submitted value, which may be private."""

    parts = []
    for error in errors[:5]:
        where = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        parts.append(f"{where}: {error.get('msg', 'invalid')}" if where else str(error.get("msg")))
    return "; ".join(parts) or "invalid request"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return error_response(
            request, error.status, error.code, error.message, retryable=error.retryable
        )

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        return error_response(
            request, 422, "validation_error", _describe(error.errors()), retryable=False
        )

    @app.exception_handler(OperationalError)
    async def database(request: Request, error: OperationalError) -> JSONResponse:
        logger.warning("database unavailable during a request", exc_info=error)
        return error_response(
            request, 503, "database_unavailable", "database is unavailable", retryable=True
        )


def _errors(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """The error statuses a route can return, documented with their one shape."""

    return {status: {"model": ErrorResponse} for status in statuses}


# --- Paper metadata ---------------------------------------------------------


class PaperIdentifier(ContractModel):
    namespace: str
    value: str


class PaperSource(ContractModel):
    """Where the paper can be read; ``available`` is false when no link is known."""

    pdf_url: str | None
    landing_url: str | None
    available: bool


class PaperDetail(ContractModel):
    request_id: UUID
    paper_id: UUID
    # The id asked for, when a merge redirected it to ``paper_id``.
    redirected_from: UUID | None = None
    paper: PaperSummary
    acceptance_type: str
    identifiers: list[PaperIdentifier]
    source: PaperSource
    parse_status: str | None


_RESOLVE = """
with recursive chain(id, merged_into, depth) as (
    select id, merged_into, 0 from papers where id = :id
    union all
    select p.id, p.merged_into, c.depth + 1
    from papers p join chain c on p.id = c.merged_into
    where c.depth < 32
)
select id, merged_into from chain order by depth desc limit 1
"""

_DETAIL = """
select p.acceptance_type,
       coalesce((
           select json_agg(json_build_object('namespace', i.namespace, 'value', i.value)
                           order by i.namespace, i.value)
           from paper_identifiers i where i.paper_id = p.id
       ), '[]') as identifiers,
       v.source_url as landing_url,
       v.parse_status
from papers p
left join lateral (
    select pv.source_url, pv.parse_status from paper_versions pv
    where pv.paper_id = p.id
    order by (pv.parse_status = 'parsed') desc, pv.created_at desc
    limit 1
) v on true
where p.id = :id
"""


class PaperDirectory:
    """Canonical paper identity and detail, straight from PostgreSQL."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def resolve(self, paper_id: UUID) -> UUID | None:
        """The surviving paper an id now names, following merges; None if unknown."""

        with self._engine.connect() as connection:
            row = connection.execute(text(_RESOLVE), {"id": paper_id}).mappings().one_or_none()
        if row is None:
            return None
        if row["merged_into"] is not None:
            # Merges refuse cycles, so a chain this long is corrupt, not deep.
            raise ApiError(500, "redirect_unresolved", "paper redirects do not terminate")
        return UUID(str(row["id"]))

    def detail(self, paper_id: UUID) -> Mapping[Any, Any]:
        with self._engine.connect() as connection:
            return connection.execute(text(_DETAIL), {"id": paper_id}).mappings().one()


# --- The default service ----------------------------------------------------


class UnavailableBranch:
    """A candidate branch whose model could not load: it fails typed, every time."""

    def __init__(self, code: str) -> None:
        self.code = code

    def search(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> list[tuple[str, float]]:
        raise SearchUnavailable(self.code)


def default_listwise(
    settings: Settings,
    engine: Engine,
    *,
    llm_path: str | Path = DEFAULT_LLM_PATH,
    prompt_path: str | Path = DEFAULT_LISTWISE_PROMPT,
) -> ListwiseReranker | None:
    """The deep-search LLM reranker, or None: then ``hybrid_rerank_llm`` degrades.

    Built only when ``LLM_RERANK_MODEL`` names a priced model whose key is set
    and a daily cap is configured. Nothing here makes a call, and a missing
    piece is logged by its code alone, never with a key.
    """

    if settings.llm_rerank_model is None:
        return None
    try:
        from ..models.llm import build_client
        from ..models.spend import SpendLedger
        from .llm_rerank import LlmListwiseReranker, load_prompt

        ledger = SpendLedger(engine, settings.llm_daily_spend_cap_usd)
        client = build_client(settings.llm_rerank_model, settings, ledger, llm_config=llm_path)
        return LlmListwiseReranker(client, load_prompt(prompt_path))
    except Exception as error:  # noqa: BLE001 - any gap degrades the opt-in mode, typed
        code = getattr(error, "code", None) or str(error).split(":")[0] or type(error).__name__
        logger.warning("LLM reranker unavailable (%s); hybrid_rerank_llm will degrade", code)
        return None


def default_search_service(
    settings: Settings,
    engine: Engine,
    client: QdrantClient,
    *,
    models_path: str | Path = DEFAULT_MODELS_PATH,
    search_path: str | Path = DEFAULT_SEARCH_PATH,
) -> SearchService:
    """The service from configuration: fixture models in mock mode, pinned models otherwise.

    A model that cannot load degrades search rather than stopping the API
    (spec §12): without the embedder the dense branch fails typed and search
    serves BM25 with ``dense_unavailable``; without the reranker
    ``hybrid_rerank`` serves the RRF order with ``rerank_unavailable``. Mock mode
    never builds the hosted-LLM reranker, so no test can spend money.
    """

    from ..models.embeddings import FixtureEmbedding, load_embedding_spec
    from .chunks import dense_branch, lexical_branch
    from .rerank import FixtureReranker, load_reranker_spec

    config = load_search_config(search_path)
    # Both branches read the configured candidate source (spec §7): the chunk
    # collection, each paper scored by its best evidence chunk, as decided in P2.6.
    lexical = lexical_branch(engine, client, source=config.candidates_source, k=config.rrf_k)
    dense: CandidateRetriever
    if settings.model_mode == "mock" or settings.mock_mode:
        return SearchService(
            engine=engine,
            lexical=lexical,
            dense=dense_branch(
                engine, client, FixtureEmbedding(), source=config.candidates_source, k=config.rrf_k
            ),
            reranker=FixtureReranker(),
            config=config,
        )
    try:
        from ..models.embeddings import TransformerEmbedding

        spec = load_embedding_spec(models_path)
        dense = dense_branch(
            engine,
            client,
            TransformerEmbedding(spec, settings.data_dir),
            source=config.candidates_source,
            max_tokens=spec.max_tokens["papers"],
            k=config.rrf_k,
        )
    except Exception as error:  # noqa: BLE001 - any load failure degrades, typed
        logger.warning("embedding model unavailable; dense search will degrade", exc_info=error)
        dense = UnavailableBranch("model_unavailable")
    reranker = None
    try:
        from .rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(load_reranker_spec(models_path), settings.data_dir)
    except Exception as error:  # noqa: BLE001 - any load failure degrades, typed
        logger.warning("reranker unavailable; hybrid_rerank will degrade", exc_info=error)
    return SearchService(
        engine=engine,
        lexical=lexical,
        dense=dense,
        reranker=reranker,
        config=config,
        listwise=default_listwise(settings, engine),
    )


def cursor_secret(secret_key: str) -> bytes:
    """A key used only for cursors, derived so the application secret is never used raw."""

    return hmac.new(secret_key.encode(), b"copilot:search-cursor:v1", hashlib.sha256).digest()


# --- Endpoint logic ---------------------------------------------------------


class SearchApi:
    """What the search routes need, built with the app and started by its lifespan."""

    def __init__(
        self,
        *,
        settings: Settings,
        engine: Engine,
        client: QdrantClient | None = None,
        service: SearchService | None = None,
        models_path: str | Path = DEFAULT_MODELS_PATH,
        search_path: str | Path = DEFAULT_SEARCH_PATH,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._client = client
        self._owns_client = client is None
        self._service = service
        self._owns_service = service is None
        self._models_path = models_path
        self._search_path = search_path
        self._clock = clock or (lambda: int(time.time()))
        self._secret = cursor_secret(settings.secret_key)
        self._prefix = settings.qdrant_collection_prefix
        self._papers = PaperStore(engine)
        self._directory = PaperDirectory(engine)
        self._cache: OrderingCache | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._client is None:
                self._client = QdrantClient(
                    url=self._settings.qdrant_url, api_key=self._settings.qdrant_api_key, timeout=10
                )
            if self._service is None:
                self._service = default_search_service(
                    self._settings,
                    self._engine,
                    self._client,
                    models_path=self._models_path,
                    search_path=self._search_path,
                )
            self._cache = OrderingCache(self._engine, self._service.config.cache_ttl_seconds)
            service = self._service
        try:
            release = capture_release(self._engine)
        except Exception as error:  # noqa: BLE001 - readiness reports the database, not startup
            logger.warning("no release to warm search against", exc_info=error)
            return
        if release is not None:
            logger.info("search warmed on %s: %s", release.id, service.warm(release))

    def close(self) -> None:
        with self._lock:
            if self._owns_service and self._service is not None:
                self._service.close()
                self._service = None
            if self._owns_client and self._client is not None:
                self._client.close()
                self._client = None
            self._cache = None

    def _ready(self) -> tuple[SearchService, OrderingCache, QdrantClient]:
        if self._service is None or self._cache is None or self._client is None:
            raise ApiError(503, "search_unavailable", "search is not started", retryable=True)
        return self._service, self._cache, self._client

    def _release(self) -> ReleaseRecord:
        """The active release, if its collections are this deployment's public ones."""

        release = capture_release(self._engine)
        if release is None:
            raise ApiError(503, "corpus_not_ready", "no active corpus release", retryable=True)
        try:
            expected = collection_names(self._prefix, release.id)
        except IndexBuildError:
            expected = {}
        if (release.paper_collection, release.chunk_collection) != (
            expected.get(PAPERS),
            expected.get(CHUNKS),
        ):
            logger.error("active release %s names collections outside this namespace", release.id)
            raise ApiError(
                503,
                "release_outside_namespace",
                "the active release is not servable by this deployment",
                retryable=False,
            )
        return release

    def search(self, body: SearchRequest, rid: UUID) -> SearchResponse:
        service, cache, _ = self._ready()
        now = self._clock()
        rows: dict[str, PaperRow] | None = None
        if body.cursor is not None:
            try:
                claims = decode_cursor(body.cursor, self._secret, now)
            except CursorError as error:
                raise _cursor_error(error.code) from error
            entry = cache.get(UUID(claims["request_id"]), now)
            if entry is None:
                raise _cursor_error("cursor_expired")
            if entry.query_key != query_key(body):
                raise ApiError(422, "cursor_mismatch", "the cursor belongs to a different search")
            offset = int(claims["offset"])
        else:
            release = self._release()
            key = ranking_key(body, release.id, release.model_revision, service.identity(body.mode))
            entry = cache.find(key, now)
            if entry is None:
                try:
                    ranking = service.rank(body, release=release, request_id=rid)
                except SearchUnavailable as error:
                    raise ApiError(
                        503, error.code, "search is unavailable", retryable=error.retryable
                    ) from error
                ordering = ranking.ordering
                entry = cache.put(
                    entry_id=ranking.request_id,
                    cache_key=key,
                    query_key=query_key(body),
                    release_id=ordering.release_id,
                    mode=body.mode,
                    items=ordering.items,
                    scores=ordering.scores,
                    warnings=ordering.warnings,
                    now=now,
                )
                rows = ranking.rows
            offset = 0
        ordering = Ordering(
            release_id=entry.release_id,
            items=entry.items,
            scores=entry.scores,
            warnings=entry.warnings,
        )
        response = service.page(
            ordering, offset=offset, limit=body.limit, request_id=rid, rows=rows
        )
        following = offset + body.limit
        if following < len(ordering.items):
            cursor = encode_cursor(str(entry.id), following, entry.expires_at, self._secret)
            response = response.model_copy(update={"next_cursor": cursor})
        return response

    def paper(self, paper_id: UUID, rid: UUID) -> PaperDetail:
        canonical = self._directory.resolve(paper_id)
        row = None if canonical is None else self._papers.load([str(canonical)]).get(str(canonical))
        if canonical is None or row is None:
            raise ApiError(404, "paper_not_found", "no paper has this id")
        detail = self._directory.detail(canonical)
        landing = detail["landing_url"]
        return PaperDetail(
            request_id=rid,
            paper_id=canonical,
            redirected_from=paper_id if canonical != paper_id else None,
            paper=row.summary(),
            acceptance_type=str(detail["acceptance_type"]),
            identifiers=[PaperIdentifier(**item) for item in detail["identifiers"]],
            source=PaperSource(
                pdf_url=row.pdf_url,
                landing_url=landing,
                available=bool(row.pdf_url or landing),
            ),
            parse_status=detail["parse_status"],
        )

    def related(
        self, paper_id: UUID, filters: PaperFilters, limit: int, rid: UUID
    ) -> SearchResponse:
        service, _, client = self._ready()
        canonical = self._directory.resolve(paper_id)
        if canonical is None:
            raise ApiError(404, "paper_not_found", "no paper has this id")
        release = self._release()
        seed = str(canonical)
        try:
            points = client.retrieve(
                release.paper_collection, ids=[seed], with_vectors=[DENSE], with_payload=False
            )
            vector = _dense_vector(points[0].vector) if points else None
            hits: list[models.ScoredPoint] = []
            if vector is not None:
                condition = qdrant_filter(filters)
                condition.must_not = [
                    *(condition.must_not or []),
                    models.HasIdCondition(has_id=[seed]),
                ]
                hits = client.query_points(
                    release.paper_collection,
                    query=vector,
                    using=DENSE,
                    query_filter=condition,
                    limit=limit,
                    with_payload=False,
                ).points
        except Exception as error:  # noqa: BLE001 - the vector store failing is a typed 503
            logger.warning("related search failed against Qdrant", exc_info=error)
            raise ApiError(
                503, "vector_unavailable", "vector service is unavailable", retryable=True
            ) from error
        items = tuple((str(hit.id), float(hit.score)) for hit in hits if str(hit.id) != seed)
        ordering = Ordering(
            release_id=release.id,
            items=items,
            scores={item: {"dense": score} for item, score in items},
            # A paper added after this release was built has no stored vector.
            warnings=() if vector is not None else ("seed_not_indexed",),
        )
        return service.page(ordering, offset=0, limit=limit, request_id=rid)


def _dense_vector(stored: object) -> list[float] | None:
    if isinstance(stored, Mapping):
        stored = stored.get(DENSE)
    if isinstance(stored, list) and all(isinstance(value, int | float) for value in stored):
        return [float(value) for value in stored]
    return None


def _cursor_error(code: str) -> ApiError:
    if code == "cursor_expired":
        return ApiError(
            410, "cursor_expired", "this search has expired; run it again for fresh results"
        )
    return ApiError(422, "cursor_invalid", "the cursor was not issued by this service")


def _state(request: Request) -> SearchApi:
    api = getattr(request.app.state, "search_api", None)
    if not isinstance(api, SearchApi):
        raise ApiError(503, "search_unavailable", "search is not configured", retryable=True)
    return api


def router() -> APIRouter:
    api = APIRouter(prefix="/v1", tags=["search"])

    @api.post("/search", response_model=SearchResponse, responses=_errors(410, 422, 503))
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        return _state(request).search(body, request_id(request))

    @api.get("/papers/{paper_id}", response_model=PaperDetail, responses=_errors(404, 422, 503))
    def paper(paper_id: UUID, request: Request) -> PaperDetail:
        return _state(request).paper(paper_id, request_id(request))

    @api.get(
        "/papers/{paper_id}/related",
        response_model=SearchResponse,
        responses=_errors(404, 422, 503),
    )
    def related(
        paper_id: UUID,
        request: Request,
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
        year_from: int | None = None,
        year_to: int | None = None,
        venues: Annotated[list[str] | None, Query()] = None,
        fulltext_only: bool = False,
    ) -> SearchResponse:
        try:
            filters = PaperFilters(
                year_from=year_from,
                year_to=year_to,
                venues=venues or [],
                fulltext_only=fulltext_only,
            )
        except ValidationError as error:
            raise ApiError(422, "validation_error", _describe(error.errors())) from error
        return _state(request).related(paper_id, filters, limit, request_id(request))

    return api
