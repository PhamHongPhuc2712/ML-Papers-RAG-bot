"""Deleting an account deletes its chats at once and ends its sessions."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from copilot.chat.store import ChatStore
from copilot.contracts import ChatRequest, Principal

pytestmark = pytest.mark.integration
PASSWORD = "correct horse battery staple"


def test_delete_me_removes_chats_and_ends_the_session(
    api_client, make_user, auth_headers, migrated_database
):
    user_id, _ = make_user()
    store = ChatStore(migrated_database)
    store.begin_turn(
        Principal(user_id=user_id),
        ChatRequest(request_id=uuid4(), message="hi"),
        stale_after=timedelta(minutes=4),
    )
    headers = auth_headers(user_id)
    wrong = api_client.request(
        "DELETE", "/v1/me", json={"password": "not the password!"}, headers=headers
    )
    assert wrong.status_code == 401
    assert wrong.json()["error"]["code"] == "invalid_credentials"
    deleted = api_client.request("DELETE", "/v1/me", json={"password": PASSWORD}, headers=headers)
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
