"""Helpful / Not helpful on an answer (sign-up plan, A5): the owner rates an answered message,
the rating comes back with the history, is counted by ``chat stats`` and is backed up."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from copilot.backup import create_backup, restore_backup
from copilot.chat.store import ChatNotFound, ChatStore, MessageNotRatable, chat_stats
from copilot.chat.turns import TurnResult
from copilot.contracts import ChatRequest, Claim, Evidence, GroundedAnswer, Principal

pytestmark = pytest.mark.integration
STALE = timedelta(minutes=4)
NOW = datetime(2026, 1, 5, 12, tzinfo=UTC)


def _result(paper_id: str) -> TurnResult:
    evidence = Evidence(
        id="E1",
        source_kind="paper",
        source_id=paper_id,
        version_id=str(uuid4()),
        chunk_id=uuid4(),
        text="Message passing.",
        section="Method",
        page_start=2,
        page_end=3,
    )
    return TurnResult(
        answer=GroundedAnswer(
            claims=[Claim(text="It passes messages.", evidence_ids=["E1"])],
            insufficient_evidence=False,
        ),
        evidence=[evidence],
        standalone_query="How do graph networks work?",
        corpus_release_id="search-a",
        warnings=[],
        identity={"chat_config": "test"},
        trace={"retrieve": 0.1},
        cost_usd=0.0004,
    )


@pytest.fixture
def store(search_corpus):
    return ChatStore(search_corpus.engine, clock=lambda: NOW)


def _answered(store, principal, search_corpus):
    turn = store.begin_turn(
        principal,
        ChatRequest(request_id=uuid4(), message="How do graph networks work?"),
        stale_after=STALE,
    )
    store.complete_turn(turn, _result(search_corpus.ids["Graph neural networks for molecules"]))
    return turn


def _feedback(store, principal, turn):
    _, messages = store.get_chat(principal, turn.chat_id)
    return messages[1].feedback


def test_the_owner_rates_changes_and_clears_and_history_returns_it(store, make_user, search_corpus):
    alice = Principal(user_id=make_user()[0])
    turn = _answered(store, alice, search_corpus)
    assert _feedback(store, alice, turn) is None
    store.set_feedback(alice, turn.chat_id, turn.assistant_message_id, "helpful")
    assert _feedback(store, alice, turn) == "helpful"
    store.set_feedback(alice, turn.chat_id, turn.assistant_message_id, "not_helpful")
    assert _feedback(store, alice, turn) == "not_helpful"
    store.set_feedback(alice, turn.chat_id, turn.assistant_message_id, None)
    assert _feedback(store, alice, turn) is None
    with search_corpus.engine.connect() as connection:
        row = connection.execute(
            text("select feedback, feedback_at from chat_messages where id = :id"),
            {"id": turn.assistant_message_id},
        ).one()
    assert row.feedback is None and row.feedback_at is None


def test_only_the_owner_and_only_an_answered_assistant_message(store, make_user, search_corpus):
    alice = Principal(user_id=make_user()[0])
    bob = Principal(user_id=make_user()[0])
    turn = _answered(store, alice, search_corpus)
    with pytest.raises(ChatNotFound):
        store.set_feedback(bob, turn.chat_id, turn.assistant_message_id, "helpful")
    with pytest.raises(MessageNotRatable):
        store.set_feedback(alice, turn.chat_id, turn.user_message_id, "helpful")
    with pytest.raises(MessageNotRatable):
        store.set_feedback(alice, turn.chat_id, uuid4(), "helpful")
    pending = store.begin_turn(
        alice,
        ChatRequest(request_id=uuid4(), chat_id=turn.chat_id, message="And their limits?"),
        stale_after=STALE,
    )
    with pytest.raises(MessageNotRatable):
        store.set_feedback(alice, turn.chat_id, pending.assistant_message_id, "helpful")
    store.fail_turn(pending, status="failed", error_code="llm_timeout")
    with pytest.raises(MessageNotRatable):
        store.set_feedback(alice, turn.chat_id, pending.assistant_message_id, "not_helpful")
    assert _feedback(store, alice, turn) is None


def test_the_api_rates_and_refuses(api_client, make_user, auth_headers, search_corpus):
    alice, _ = make_user()
    bob, _ = make_user()
    store = ChatStore(search_corpus.engine)
    turn = _answered(store, Principal(user_id=alice), search_corpus)
    path = f"/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/feedback"
    rated = api_client.put(path, json={"rating": "helpful"}, headers=auth_headers(alice))
    assert rated.status_code == 204
    detail = api_client.get(f"/v1/chats/{turn.chat_id}", headers=auth_headers(alice)).json()
    assert [m["feedback"] for m in detail["messages"]] == [None, "helpful"]
    cleared = api_client.put(path, json={"rating": None}, headers=auth_headers(alice))
    assert cleared.status_code == 204
    other = api_client.put(path, json={"rating": "helpful"}, headers=auth_headers(bob))
    assert other.status_code == 404 and other.json()["error"]["code"] == "chat_not_found"
    user_path = f"/v1/chats/{turn.chat_id}/messages/{turn.user_message_id}/feedback"
    refused = api_client.put(user_path, json={"rating": "helpful"}, headers=auth_headers(alice))
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "message_not_ratable"
    bad = api_client.put(path, json={"rating": "great"}, headers=auth_headers(alice))
    assert bad.status_code == 422
    assert api_client.put(path, json={"rating": "helpful"}).status_code == 401


def test_chat_stats_counts_each_rating_and_backup_carries_it(
    store, make_user, search_corpus, tmp_path
):
    alice = Principal(user_id=make_user()[0])
    first = _answered(store, alice, search_corpus)
    second = _answered(store, alice, search_corpus)
    _answered(store, alice, search_corpus)
    store.set_feedback(alice, first.chat_id, first.assistant_message_id, "helpful")
    store.set_feedback(alice, second.chat_id, second.assistant_message_id, "not_helpful")
    stats = chat_stats(search_corpus.engine, day=date(2026, 1, 5))
    assert stats["feedback"] == {"helpful": 1, "not_helpful": 1, "unrated": 1}

    create_backup(search_corpus.engine, tmp_path / "b")
    restore_backup(search_corpus.engine, tmp_path / "b", replace=True)
    assert _feedback(store, alice, first) == "helpful"
    assert _feedback(store, alice, second) == "not_helpful"
