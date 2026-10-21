"""Deleting an account deletes its chats at once and ends its sessions."""

from __future__ import annotations

import re
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from copilot.chat.store import ChatStore
from copilot.contracts import ChatRequest, Principal

pytestmark = pytest.mark.integration


def _code(mailbox) -> str:
    return re.search(r"\b(\d{6})\b", mailbox.sent[-1].body).group(1)


def _wrong(code: str) -> str:
    return f"{(int(code) + 1) % 1_000_000:06d}"


def _begin(store: ChatStore, user_id, message="hi"):
    return store.begin_turn(
        Principal(user_id=user_id),
        ChatRequest(request_id=uuid4(), message=message),
        stale_after=timedelta(minutes=4),
    )


def test_the_signed_in_account_gets_a_code_at_its_own_email(
    api_client, make_user, auth_headers, mailbox
):
    user_id, email = make_user()
    asked = api_client.post("/v1/me/code", headers=auth_headers(user_id))
    assert asked.status_code == 202
    assert mailbox.sent[-1].to == email
    assert api_client.post("/v1/me/code").status_code == 401


def test_delete_me_takes_a_code_removes_chats_and_ends_the_session(
    api_client, make_user, auth_headers, migrated_database, mailbox
):
    user_id, _ = make_user()
    _begin(ChatStore(migrated_database), user_id)
    headers = auth_headers(user_id)
    api_client.post("/v1/me/code", headers=headers)
    code = _code(mailbox)
    wrong = api_client.request("DELETE", "/v1/me", json={"code": _wrong(code)}, headers=headers)
    assert wrong.status_code == 401 and wrong.json()["error"]["code"] == "invalid_code"
    assert api_client.get("/v1/me", headers=headers).status_code == 200
    deleted = api_client.request("DELETE", "/v1/me", json={"code": code}, headers=headers)
    assert deleted.status_code == 204
    assert api_client.get("/v1/me", headers=headers).status_code == 401
    with migrated_database.connect() as connection:
        left = connection.execute(
            text("select count(*) from chats where user_id = :id"), {"id": user_id}
        ).scalar_one()
        messages = connection.execute(
            text("select count(*) from chat_messages where user_id = :id"), {"id": user_id}
        ).scalar_one()
    assert left == 0 and messages == 0


def test_a_password_no_longer_deletes_an_account(api_client, make_user, auth_headers):
    user_id, _ = make_user()
    response = api_client.request(
        "DELETE", "/v1/me", json={"password": "correct horse battery staple"},
        headers=auth_headers(user_id),
    )
    assert response.status_code == 422


def test_delete_all_chats_needs_a_code_and_keeps_only_others_chats(
    api_client, make_user, auth_headers, migrated_database, mailbox
):
    store = ChatStore(migrated_database)
    alice, _ = make_user()
    bob, _ = make_user()
    for _ in range(2):
        _begin(store, alice)
    _begin(store, bob)
    headers = auth_headers(alice)
    api_client.post("/v1/me/code", headers=headers)
    code = _code(mailbox)
    wrong = api_client.request("DELETE", "/v1/chats", json={"code": _wrong(code)}, headers=headers)
    assert wrong.status_code == 401
    deleted = api_client.request("DELETE", "/v1/chats", json={"code": code}, headers=headers)
    assert deleted.status_code == 204
    assert api_client.get("/v1/me", headers=headers).status_code == 200
    with migrated_database.connect() as connection:
        counts = dict(
            connection.execute(
                text(
                    "select user_id::text, count(*) from chats"
                    " where user_id = any(:ids) group by user_id"
                ),
                {"ids": [alice, bob]},
            ).all()
        )
    assert counts == {str(bob): 1}


def test_a_deleted_account_leaves_its_ledger_rows(make_user, migrated_database):
    from decimal import Decimal

    from copilot.identity.users import UserStore
    from copilot.models.spend import SpendLedger, load_llm_config

    user_id, _ = make_user()
    ledger = SpendLedger(migrated_database, daily_cap_usd=Decimal("100"))
    model = load_llm_config("configs/llm.yaml").models["deepseek-flash"]
    call = ledger.reserve(
        model=model, estimated_usd=0.01, purpose="search", request_id=None, run_id=None
    )
    assert UserStore(migrated_database).delete(user_id) is True
    assert UserStore(migrated_database).delete(user_id) is False
    with migrated_database.connect() as connection:
        kept = connection.execute(
            text("select count(*) from llm_calls where id = :id"), {"id": call}
        ).scalar_one()
    assert kept == 1
