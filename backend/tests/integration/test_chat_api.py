"""The chat history API: only the owner sees, lists and deletes a chat."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from copilot.chat.store import ChatStore
from copilot.contracts import ChatRequest, Principal

pytestmark = pytest.mark.integration


def _chat_for(engine, user_id, message="What is message passing?"):
    store = ChatStore(engine)
    request = ChatRequest(request_id=uuid4(), message=message)
    return store.begin_turn(Principal(user_id=user_id), request, stale_after=timedelta(minutes=4))


def test_the_owner_lists_and_reads_a_chat(api_client, make_user, auth_headers, migrated_database):
    alice, _ = make_user()
    turn = _chat_for(migrated_database, alice)
    listing = api_client.get("/v1/chats", headers=auth_headers(alice))
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()["items"]] == [str(turn.chat_id)]
    detail = api_client.get(f"/v1/chats/{turn.chat_id}", headers=auth_headers(alice))
    assert detail.status_code == 200
    body = detail.json()
    assert body["chat"]["title"] == "What is message passing?"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]


def test_another_user_gets_404_and_an_empty_list(
    api_client, make_user, auth_headers, migrated_database
):
    alice, _ = make_user()
    bob, _ = make_user()
    turn = _chat_for(migrated_database, alice)
    assert api_client.get("/v1/chats", headers=auth_headers(bob)).json()["items"] == []
    for method in ("get", "delete"):
        response = getattr(api_client, method)(
            f"/v1/chats/{turn.chat_id}", headers=auth_headers(bob)
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "chat_not_found"


def test_delete_then_404(api_client, make_user, auth_headers, migrated_database):
    alice, _ = make_user()
    turn = _chat_for(migrated_database, alice)
    path = f"/v1/chats/{turn.chat_id}"
    assert api_client.delete(path, headers=auth_headers(alice)).status_code == 204
    assert api_client.get(path, headers=auth_headers(alice)).status_code == 404


def test_history_needs_a_token(api_client):
    assert api_client.get("/v1/chats").status_code == 401
    assert api_client.get(f"/v1/chats/{uuid4()}").status_code == 401
