"""Signed search cursors: what they carry, and every way a bad one is refused."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from uuid import uuid4

import pytest

from copilot.contracts import PaperFilters, SearchRequest
from copilot.search.cache import (
    CursorError,
    decode_cursor,
    encode_cursor,
    query_key,
    ranking_key,
)

SECRET = b"k" * 32
NOW = 1_800_000_000


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _signed(body: object, secret: bytes = SECRET) -> str:
    """A correctly signed token around any payload: the format, written out by hand."""

    payload = _b64(json.dumps(body).encode())
    signature = _b64(hmac.new(secret, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}"


def _code(token: str, now: int = NOW) -> str:
    with pytest.raises(CursorError) as refused:
        decode_cursor(token, SECRET, now)
    return refused.value.code


def test_a_cursor_round_trips_its_request_offset_and_expiry():
    request_id = str(uuid4())
    token = encode_cursor(request_id, 40, NOW + 600, SECRET)
    assert decode_cursor(token, SECRET, NOW) == {
        "request_id": request_id,
        "offset": 40,
        "expires_at": NOW + 600,
    }
    # URL-safe as issued: no padding, nothing a query string would mangle.
    assert "=" not in token and "+" not in token and "/" not in token


def test_a_changed_offset_breaks_the_signature():
    token = encode_cursor(str(uuid4()), 20, NOW + 600, SECRET)
    payload, signature = token.split(".")
    body = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    body["off"] = 180
    forged = _b64(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())
    assert _code(f"{forged}.{signature}") == "cursor_invalid"


def test_a_changed_signature_or_secret_is_refused():
    token = encode_cursor(str(uuid4()), 20, NOW + 600, SECRET)
    payload, signature = token.split(".")
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    assert _code(f"{payload}.{flipped}") == "cursor_invalid"
    with pytest.raises(CursorError, match="cursor_invalid"):
        decode_cursor(token, b"another secret", NOW)


def test_expiry_is_exclusive_and_typed():
    token = encode_cursor(str(uuid4()), 20, NOW + 600, SECRET)
    assert decode_cursor(token, SECRET, NOW + 599)["offset"] == 20
    assert _code(token, NOW + 600) == "cursor_expired"
    assert _code(token, NOW + 10_000) == "cursor_expired"


def test_an_unsigned_payload_is_never_trusted_even_to_say_it_expired():
    body = {"v": 1, "rid": str(uuid4()), "off": 0, "exp": NOW - 1}
    assert _code(_signed(body, b"wrong")) == "cursor_invalid"


@pytest.mark.parametrize(
    "token",
    ["", "no-dot", "payload.", ".signature", "a.b.c", "été.signature"],
)
def test_malformed_tokens_are_invalid(token):
    assert _code(token) == "cursor_invalid"


@pytest.mark.parametrize(
    "body",
    [
        "not an object",
        {"v": 1, "off": 0, "exp": NOW + 60},
        {"v": 1, "rid": "not-a-uuid", "off": 0, "exp": NOW + 60},
        {"v": 1, "rid": str(uuid4()), "off": -1, "exp": NOW + 60},
        {"v": 1, "rid": str(uuid4()), "off": "20", "exp": NOW + 60},
        {"v": 1, "rid": str(uuid4()), "off": True, "exp": NOW + 60},
        {"v": 1, "rid": str(uuid4()), "off": 0, "exp": float(NOW + 60)},
        {"v": 2, "rid": str(uuid4()), "off": 0, "exp": NOW + 60},
    ],
)
def test_a_signed_payload_of_the_wrong_shape_is_invalid(body):
    assert _code(_signed(body)) == "cursor_invalid"


def test_a_cursor_cannot_be_issued_for_a_negative_offset_or_a_non_uuid():
    with pytest.raises(ValueError):
        encode_cursor(str(uuid4()), -1, NOW, SECRET)
    with pytest.raises(ValueError):
        encode_cursor("request-7", 0, NOW, SECRET)


def _request(query="graph learning", mode="hybrid", **filters) -> SearchRequest:
    return SearchRequest(query=query, mode=mode, filters=PaperFilters(**filters))


def test_the_query_key_ignores_venue_order_and_page_size():
    assert query_key(_request(venues=["ICLR", "ACL"])) == query_key(
        _request(venues=["ACL", "ICLR", "ACL"])
    )
    base = query_key(_request())
    assert base != query_key(_request(query="graph learning!"))
    assert base != query_key(_request(mode="bm25"))
    assert base != query_key(_request(year_from=2024))
    # The page size is not part of the question: pages may differ in length.
    assert base == query_key(_request().model_copy(update={"limit": 7}))


def test_the_ranking_key_changes_with_release_embedder_and_ranker():
    request = _request(mode="hybrid_rerank")
    identity = {"reranker": "bge@abc#pair1024", "rrf_k": 60}
    base = ranking_key(request, "rel-a", "bge-m3@1", identity)
    assert base == ranking_key(request, "rel-a", "bge-m3@1", dict(identity))
    assert base != ranking_key(request, "rel-b", "bge-m3@1", identity)
    assert base != ranking_key(request, "rel-a", "bge-m3@2", identity)
    assert base != ranking_key(request, "rel-a", "bge-m3@1", {**identity, "reranker": "x#pair512"})
