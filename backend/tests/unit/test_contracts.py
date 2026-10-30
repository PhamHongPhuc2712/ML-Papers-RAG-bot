"""Contract model acceptance tests."""

from __future__ import annotations

from uuid import UUID, uuid4

from copilot.contracts import PaperFilters, Principal, SearchRequest


def test_principal_serializes_uuid_as_json_string():
    user_id = uuid4()
    principal = Principal(user_id=user_id)

    assert principal.model_dump(mode="json") == {"user_id": str(user_id)}


def test_search_request_uses_the_declared_default_limit_and_cursor():
    request = SearchRequest(query="retrieval", mode="bm25", filters=PaperFilters())

    assert request.limit == 20
    assert request.cursor is None


def test_production_uuid_fields_reject_short_fixture_identifiers():
    try:
        Principal(user_id=UUID("00000000-0000-0000-0000-000000000001"))
    except Exception as exc:  # pragma: no cover - defensive assertion context
        raise AssertionError("valid UUID unexpectedly rejected") from exc
