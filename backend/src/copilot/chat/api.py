"""The chat endpoints (spec §10): history here; the streaming turn is added in C6."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response

from ..contracts import ChatDetail, ChatList
from ..identity.api import CurrentPrincipal
from ..search.api import ApiError, request_id
from .store import ChatNotFound, ChatStore


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

    @api.delete("/v1/chats/{chat_id}", status_code=204, response_class=Response)
    def delete(chat_id: UUID, principal: CurrentPrincipal, request: Request) -> Response:
        try:
            _store(request).delete_chat(principal, chat_id)
        except ChatNotFound as error:
            raise _not_found() from error
        return Response(status_code=204)

    return api
