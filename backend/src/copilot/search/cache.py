"""Cached orderings and the signed cursors that page through them (spec §7).

A search's ordering — its top 200, or for ``hybrid_rerank`` its reranked head —
is kept in PostgreSQL for ten minutes. A cursor names that ordering and an
offset, signed with HMAC-SHA256 so a client can neither forge an offset into
someone else's ordering nor extend a cursor's life. Pages come from the one
stored ordering, so their boundaries are stable and nothing repeats, even if
the active release switches while a client is paging.

Only hashes of the query are stored: the cache key and the cursor binding are
enough to find an ordering again, and a raw query never lands in the table.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text

from ..contracts import PaperFilters, SearchRequest

CURSOR_VERSION = 1


class CursorError(ValueError):
    """A cursor that cannot be used: ``cursor_invalid`` or ``cursor_expired``."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sign(payload: str, secret: bytes) -> str:
    return _b64encode(hmac.new(secret, payload.encode("ascii"), hashlib.sha256).digest())


def encode_cursor(request_id: str, offset: int, expires_at: int, secret: bytes) -> str:
    """``<payload>.<signature>``, both URL-safe base64; the payload is compact JSON."""

    if offset < 0:
        raise ValueError("invalid_offset")
    body = {"v": CURSOR_VERSION, "rid": str(UUID(request_id)), "off": offset, "exp": expires_at}
    payload = _b64encode(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())
    return f"{payload}.{_sign(payload, secret)}"


def decode_cursor(token: str, secret: bytes, now: int) -> dict[str, Any]:
    """The cursor's request id, offset and expiry, or a typed refusal.

    The signature is checked first and in constant time, so nothing about an
    unsigned payload — including whether it claims to be expired — is trusted.
    """

    payload, dot, signature = token.partition(".")
    if not dot or not payload or not signature:
        raise CursorError("cursor_invalid")
    try:
        expected = _sign(payload, secret)
    except UnicodeEncodeError as error:
        raise CursorError("cursor_invalid") from error
    if not hmac.compare_digest(expected.encode("ascii"), signature.encode("utf-8")):
        raise CursorError("cursor_invalid")
    try:
        body = json.loads(_b64decode(payload))
        request_id = str(UUID(str(body["rid"])))
        offset, expires_at, version = body["off"], body["exp"], body["v"]
    except (ValueError, KeyError, TypeError, binascii.Error) as error:
        raise CursorError("cursor_invalid") from error
    if version != CURSOR_VERSION or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in (offset, expires_at)
    ):
        raise CursorError("cursor_invalid")
    if offset < 0:
        raise CursorError("cursor_invalid")
    if now >= expires_at:
        raise CursorError("cursor_expired")
    return {"request_id": request_id, "offset": offset, "expires_at": expires_at}


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _filters(filters: PaperFilters) -> dict[str, Any]:
    # Venues are a set: the same venues in another order are the same search.
    return {**filters.model_dump(mode="json"), "venues": sorted(set(filters.venues))}


def query_key(request: SearchRequest) -> str:
    """What a cursor is bound to: the question, not the release that answered it."""

    return _digest(
        {"query": request.query, "mode": request.mode, "filters": _filters(request.filters)}
    )


def ranking_key(
    request: SearchRequest, release_id: str, embedding: str, identity: Mapping[str, Any]
) -> str:
    """What makes two orderings interchangeable: the question, the release and the rankers."""

    return _digest(
        {
            "query": query_key(request),
            "release": release_id,
            "embedding": embedding,
            "ranker": dict(identity),
        }
    )


@dataclass(frozen=True)
class CachedOrdering:
    id: UUID
    cache_key: str
    query_key: str
    release_id: str
    mode: str
    items: tuple[tuple[str, float], ...]
    scores: Mapping[str, Mapping[str, float]]
    warnings: tuple[str, ...]
    expires_at: int


def _at(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, UTC)


_COLUMNS = "id, cache_key, query_key, corpus_release_id, mode, items, warnings, expires_at"


def _entry(row: Mapping[Any, Any]) -> CachedOrdering:
    items = row["items"]
    return CachedOrdering(
        id=UUID(str(row["id"])),
        cache_key=str(row["cache_key"]),
        query_key=str(row["query_key"]),
        release_id=str(row["corpus_release_id"]),
        mode=str(row["mode"]),
        items=tuple((str(item["id"]), float(item["score"])) for item in items),
        scores={str(item["id"]): dict(item["scores"]) for item in items},
        warnings=tuple(str(warning) for warning in row["warnings"]),
        expires_at=int(row["expires_at"].timestamp()),
    )


class OrderingCache:
    """Orderings in PostgreSQL, found again by id (a cursor) or by key (a repeated search)."""

    def __init__(self, engine: Engine, ttl_seconds: int) -> None:
        if ttl_seconds < 1:
            raise ValueError("invalid_cache_ttl")
        self._engine = engine
        self.ttl_seconds = ttl_seconds

    def put(
        self,
        *,
        entry_id: UUID,
        cache_key: str,
        query_key: str,
        release_id: str,
        mode: str,
        items: Sequence[tuple[str, float]],
        scores: Mapping[str, Mapping[str, float]],
        warnings: Sequence[str],
        now: int,
    ) -> CachedOrdering:
        expires_at = now + self.ttl_seconds
        stored = [
            {"id": paper_id, "score": score, "scores": dict(scores.get(paper_id, {}))}
            for paper_id, score in items
        ]
        with self._engine.begin() as connection:
            # Expired orderings are unreachable by any cursor; clear them as we go.
            connection.execute(
                text("delete from search_orderings where expires_at <= :now"), {"now": _at(now)}
            )
            row = (
                connection.execute(
                    text(
                        "insert into search_orderings (id, cache_key, query_key,"
                        " corpus_release_id, mode, items, warnings, created_at, expires_at)"
                        " values (:id, :cache_key, :query_key, :release, :mode,"
                        " cast(:items as jsonb), cast(:warnings as jsonb), :created, :expires)"
                        f" returning {_COLUMNS}"
                    ),
                    {
                        "id": entry_id,
                        "cache_key": cache_key,
                        "query_key": query_key,
                        "release": release_id,
                        "mode": mode,
                        "items": json.dumps(stored),
                        "warnings": json.dumps(list(warnings)),
                        "created": _at(now),
                        "expires": _at(expires_at),
                    },
                )
                .mappings()
                .one()
            )
        return _entry(row)

    def get(self, entry_id: UUID, now: int) -> CachedOrdering | None:
        """The ordering a cursor names, while it lives."""

        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        f"select {_COLUMNS} from search_orderings"
                        " where id = :id and expires_at > :now"
                    ),
                    {"id": entry_id, "now": _at(now)},
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else _entry(row)

    def find(self, cache_key: str, now: int) -> CachedOrdering | None:
        """A live, undegraded ordering for the same key, newest first.

        A degraded ordering stays reachable by its own cursors but is never
        handed to a new search: the reranker that timed out may be back.
        """

        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        f"select {_COLUMNS} from search_orderings"
                        " where cache_key = :key and expires_at > :now"
                        " and jsonb_array_length(warnings) = 0"
                        " order by created_at desc limit 1"
                    ),
                    {"key": cache_key, "now": _at(now)},
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else _entry(row)
