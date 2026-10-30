"""Chats, messages and the turn lifecycle: owned, ordered, idempotent, never stuck pending."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from copilot.chat.store import ChatNotFound, ChatStore, TurnConflict
from copilot.chat.turns import TurnResult
from copilot.contracts import ChatRequest, Claim, Evidence, GroundedAnswer, Principal

pytestmark = pytest.mark.integration
STALE = timedelta(minutes=4)


def _evidence(paper_id: str, label: str, text: str) -> Evidence:
    return Evidence(
        id=label,
        source_kind="paper",
        source_id=paper_id,
        version_id=str(uuid4()),
        chunk_id=uuid4(),
        text=text,
        section="Method",
        page_start=2,
        page_end=3,
    )


def _result(paper_id: str, *, insufficient: bool = False) -> TurnResult:
    evidence = [_evidence(paper_id, "E1", "Message passing."), _evidence(paper_id, "E2", "Other.")]
    claims = [] if insufficient else [Claim(text="It passes messages.", evidence_ids=["E1"])]
    return TurnResult(
        answer=GroundedAnswer(claims=claims, insufficient_evidence=insufficient),
        evidence=evidence,
        standalone_query="How do graph networks work?",
        corpus_release_id="search-a",
        warnings=[],
        identity={"chat_config": "test"},
        trace={"retrieve": 0.1},
        cost_usd=0.0004,
    )


@pytest.fixture
def store(search_corpus):
    return ChatStore(search_corpus.engine)


@pytest.fixture
def alice(make_user):
    return Principal(user_id=make_user()[0])


def _request(message="How do graph networks work?", chat_id=None, request_id=None):
    return ChatRequest(request_id=request_id or uuid4(), chat_id=chat_id, message=message)


def test_a_first_turn_creates_a_titled_chat_with_a_pending_answer(store, alice):
    turn = store.begin_turn(alice, _request(), stale_after=STALE)
    summary, messages = store.get_chat(alice, turn.chat_id)
    assert summary.title == "How do graph networks work?"
    assert [(m.ordinal, m.role, m.status) for m in messages] == [
        (0, "user", "complete"),
        (1, "assistant", "pending"),
    ]
    assert turn.ordinal == 1 and turn.replay is None


def test_completion_stores_claims_and_an_evidence_snapshot(store, alice, search_corpus):
    paper = search_corpus.ids["Graph neural networks for molecules"]
    turn = store.begin_turn(alice, _request(), stale_after=STALE)
    message = store.complete_turn(turn, _result(paper))
    assert message.status == "complete" and message.text == "It passes messages."
    assert [(e.id, e.cited) for e in message.evidence] == [("E1", True), ("E2", False)]
    first = message.evidence[0]
    assert first.title == "Graph neural networks for molecules"
    assert first.venue == "ICLR" and first.year == 2024 and first.page_start == 2


def test_an_insufficient_answer_is_stored_as_such(store, alice, search_corpus):
    paper = search_corpus.ids["Graph neural networks for molecules"]
    turn = store.begin_turn(alice, _request(), stale_after=STALE)
    message = store.complete_turn(turn, _result(paper, insufficient=True))
    assert message.status == "insufficient" and message.claims == []


def test_the_same_request_replays_once_complete(store, alice, search_corpus):
    paper = search_corpus.ids["Graph neural networks for molecules"]
    request = _request()
    turn = store.begin_turn(alice, request, stale_after=STALE)
    stored = store.complete_turn(turn, _result(paper))
    again = store.begin_turn(alice, request, stale_after=STALE)
    assert again.replay == stored and again.chat_id == turn.chat_id


def test_a_new_chat_request_retried_without_its_chat_id_finds_the_same_chat(store, alice):
    request = _request()
    first = store.begin_turn(alice, request, stale_after=STALE)
    with pytest.raises(TurnConflict) as caught:
        store.begin_turn(alice, request, stale_after=STALE)
    assert caught.value.code == "request_in_progress"
    assert len(store.list_chats(alice, limit=50)) == 1
    assert store.get_chat(alice, first.chat_id)


def test_one_turn_at_a_time_per_chat(store, alice):
    turn = store.begin_turn(alice, _request(), stale_after=STALE)
    with pytest.raises(TurnConflict) as caught:
        store.begin_turn(alice, _request("next", chat_id=turn.chat_id), stale_after=STALE)
    assert caught.value.code == "chat_busy"


def test_a_request_id_reused_with_another_message_is_refused(store, alice, search_corpus):
    paper = search_corpus.ids["Graph neural networks for molecules"]
    request = _request()
    turn = store.begin_turn(alice, request, stale_after=STALE)
    store.complete_turn(turn, _result(paper))
    reused = _request("something else", chat_id=turn.chat_id, request_id=request.request_id)
    with pytest.raises(TurnConflict) as caught:
        store.begin_turn(alice, reused, stale_after=STALE)
    assert caught.value.code == "request_id_reused"


def test_a_failed_turn_runs_again_under_the_same_request(store, alice):
    request = _request()
    turn = store.begin_turn(alice, request, stale_after=STALE)
    store.fail_turn(turn, status="failed", error_code="llm_spend_cap")
    again = store.begin_turn(alice, request, stale_after=STALE)
    assert again.replay is None and again.assistant_message_id == turn.assistant_message_id
    _, messages = store.get_chat(alice, turn.chat_id)
    assert messages[-1].status == "pending" and messages[-1].error_code is None


def test_a_pending_turn_past_the_deadline_becomes_interrupted(search_corpus, alice):
    now = [datetime(2026, 10, 11, 12, 0, tzinfo=UTC)]
    store = ChatStore(search_corpus.engine, clock=lambda: now[0])
    turn = store.begin_turn(alice, _request(), stale_after=STALE)
    now[0] += STALE + timedelta(seconds=1)
    assert store.interrupt_stale(stale_after=STALE) == 1
    _, messages = store.get_chat(alice, turn.chat_id)
    assert messages[-1].status == "interrupted"
    # The chat is free again: a new request starts a new turn after the interrupted one.
    nxt = store.begin_turn(alice, _request("again", chat_id=turn.chat_id), stale_after=STALE)
    assert nxt.ordinal == 3


def test_history_keeps_answered_turns_newest_last(store, alice, search_corpus):
    paper = search_corpus.ids["Graph neural networks for molecules"]
    chat_id = None
    for index in range(4):
        turn = store.begin_turn(alice, _request(f"q{index}", chat_id=chat_id), stale_after=STALE)
        chat_id = turn.chat_id
        if index == 2:
            store.fail_turn(turn, status="failed", error_code="llm_timeout")
        else:
            store.complete_turn(turn, _result(paper))
    history = store.history(chat_id, before_ordinal=99, turns=2)
    assert [(item.role, item.text) for item in history] == [
        ("user", "q1"),
        ("assistant", "It passes messages."),
        ("user", "q3"),
        ("assistant", "It passes messages."),
    ]


def test_another_users_chat_does_not_exist_for_them(store, alice, make_user):
    bob = Principal(user_id=make_user()[0])
    turn = store.begin_turn(alice, _request(), stale_after=STALE)
    for action in (
        lambda: store.get_chat(bob, turn.chat_id),
        lambda: store.delete_chat(bob, turn.chat_id),
        lambda: store.begin_turn(bob, _request("x", chat_id=turn.chat_id), stale_after=STALE),
    ):
        with pytest.raises(ChatNotFound):
            action()
    assert store.list_chats(bob, limit=50) == []


def test_deleting_a_chat_removes_its_messages(store, alice):
    turn = store.begin_turn(alice, _request(), stale_after=STALE)
    store.delete_chat(alice, turn.chat_id)
    with pytest.raises(ChatNotFound):
        store.get_chat(alice, turn.chat_id)
