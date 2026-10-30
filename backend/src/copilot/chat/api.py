"""The chat endpoints (spec §10): history, and one turn streamed as server-sent events."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import StreamingResponse

from ..config import Settings
from ..contracts import ChatDetail, ChatList, ChatRequest, FeedbackRequest
from ..identity.api import ConfirmRequest, CurrentPrincipal, confirm_code
from ..search.api import ApiError, request_id
from .agent import ChatAgent, build_chat_agent
from .config import load_chat_config
from .limits import DailyLimitReached
from .service import ChatService
from .store import ChatNotFound, ChatStore, MessageNotRatable, TurnConflict

logger = logging.getLogger(__name__)
_CONFLICTS = {
    "request_in_progress": "this request is still being answered",
    "chat_busy": "this chat is answering another message",
    "request_id_reused": "this request id was used for a different message",
}


class ChatApi:
    """The chat turn's dependencies, built at startup once search has loaded its models."""

    def __init__(
        self, *, settings: Settings, engine: Any, search: Any, agent: ChatAgent | None = None
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._search = search
        self._agent = agent
        self._store = ChatStore(engine)
        config = load_chat_config()
        self._stale_after = timedelta(seconds=2 * config.turn_timeout)
        self._daily_answers = config.daily_answers
        self._service: ChatService | None = None
        self._unavailable = "chat_not_started"

    def start(self) -> None:
        try:
            # The database may be down at startup: readiness reports that, and the API
            # still starts, so this sweep is guarded with the rest.
            interrupted = self._store.interrupt_stale(stale_after=self._stale_after)
            if interrupted:
                logger.info("interrupted %d chat turns left pending", interrupted)
            agent = self._agent or build_chat_agent(self._settings, self._engine, self._search)
        except Exception as error:  # noqa: BLE001 - chat failing to build must not stop search
            # LlmError (no key or cap), ApiError (search not started), a missing tokenizer
            # under DATA_DIR, an unreachable database: each leaves chat unavailable, typed,
            # and search serving.
            self._unavailable = getattr(error, "code", None) or type(error).__name__
            logger.warning("chat unavailable: %s", self._unavailable, exc_info=error)
            return
        self._service = ChatService(
            self._store, agent, stale_after=self._stale_after, daily_answers=self._daily_answers
        )

    def close(self) -> None:
        self._service = None

    def service(self) -> ChatService:
        if self._service is None:
            raise ApiError(503, "chat_unavailable", f"chat is unavailable ({self._unavailable})")
        return self._service


def _store(request: Request) -> ChatStore:
    store = getattr(request.app.state, "chat_store", None)
    if not isinstance(store, ChatStore):
        raise ApiError(503, "chat_unavailable", "chat is not configured", retryable=True)
    return store


def _not_found() -> ApiError:
    return ApiError(404, "chat_not_found", "no such chat")


def router() -> APIRouter:
    api = APIRouter()

    @api.get("/v1/chats", response_model=ChatList)
    def chats(
        principal: CurrentPrincipal,
        request: Request,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> ChatList:
        items = _store(request).list_chats(principal, limit=limit)
        return ChatList(request_id=request_id(request), items=items)

    @api.get("/v1/chats/{chat_id}", response_model=ChatDetail)
    def chat(chat_id: UUID, principal: CurrentPrincipal, request: Request) -> ChatDetail:
        try:
            summary, messages = _store(request).get_chat(principal, chat_id)
        except ChatNotFound as error:
            raise _not_found() from error
        return ChatDetail(request_id=request_id(request), chat=summary, messages=messages)

    @api.delete("/v1/chats", status_code=204, response_class=Response)
    def delete_all(body: ConfirmRequest, principal: CurrentPrincipal, request: Request) -> Response:
        """Delete every chat of the caller, confirmed by a code from POST /v1/me/code (A4).
        Today's answers stay counted."""

        confirm_code(request, principal, body.code)
        _store(request).delete_all_chats(principal)
        return Response(status_code=204)

    @api.delete("/v1/chats/{chat_id}", status_code=204, response_class=Response)
    def delete(chat_id: UUID, principal: CurrentPrincipal, request: Request) -> Response:
        try:
            _store(request).delete_chat(principal, chat_id)
        except ChatNotFound as error:
            raise _not_found() from error
        return Response(status_code=204)

    @api.put(
        "/v1/chats/{chat_id}/messages/{message_id}/feedback",
        status_code=204,
        response_class=Response,
    )
    def feedback(
        chat_id: UUID,
        message_id: UUID,
        body: FeedbackRequest,
        principal: CurrentPrincipal,
        request: Request,
    ) -> Response:
        """Helpful / Not helpful on an answer of the caller's chat; null clears it (A5)."""

        try:
            _store(request).set_feedback(principal, chat_id, message_id, body.rating)
        except ChatNotFound as error:
            raise _not_found() from error
        except MessageNotRatable as error:
            raise ApiError(
                409, "message_not_ratable", "only a finished answer in this chat can be rated"
            ) from error
        return Response(status_code=204)

    @api.post(
        "/v1/chats/stream",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}, "description": "A chat turn"}},
    )
    def stream(
        body: ChatRequest, principal: CurrentPrincipal, request: Request
    ) -> StreamingResponse:
        chat_api = getattr(request.app.state, "chat_api", None)
        if not isinstance(chat_api, ChatApi):
            raise ApiError(503, "chat_unavailable", "chat is not configured")
        try:
            events = chat_api.service().stream(principal, body)
        except ChatNotFound as error:
            raise _not_found() from error
        except DailyLimitReached as error:
            wait = max(1, math.ceil((error.resets_at - datetime.now(UTC)).total_seconds()))
            raise ApiError(
                429,
                "daily_limit",
                f"you have used today's {error.limit} answers; they reset at 00:00 UTC",
                retryable=True,
                retry_after=wait,
            ) from error
        except TurnConflict as error:
            raise ApiError(
                409,
                error.code,
                _CONFLICTS.get(error.code, "conflict"),
                retryable=error.code != "request_id_reused",
            ) from error
        return StreamingResponse(
            events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return api
