"""Answers a day per account (sign-up plan, A3): counted, enforced before anything is spent,
not slipped past by two turns at once, and reported by GET /v1/me."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from copilot.chat.limits import DailyLimitReached, usage
from copilot.chat.store import ChatStore
from copilot.chat.turns import TurnResult
from copilot.contracts import ChatRequest, Claim, Evidence, GroundedAnswer, Principal

pytestmark = pytest.mark.integration
STALE = timedelta(minutes=4)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 21, 22, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _result(paper_id: str) -> TurnResult:
    evidence = Evidence(
        id="E1",
        source_kind="paper",
        source_id=paper_id,
        version_id=str(uuid4()),
        chunk_id=uuid4(),
        text="Message passing.",
        section="Method",
        page_start=1,
        page_end=1,
    )
    return TurnResult(
        answer=GroundedAnswer(
            claims=[Claim(text="It passes messages.", evidence_ids=["E1"])],
            insufficient_evidence=False,
        ),
        evidence=[evidence],
        standalone_query="q",
        corpus_release_id="search-a",
        warnings=[],
        identity={"chat_config": "test"},
        trace={},
        cost_usd=0.0,
    )


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(search_corpus, clock):
    return ChatStore(search_corpus.engine, clock)


@pytest.fixture
def paper(search_corpus):
    return search_corpus.ids["Graph neural networks for molecules"]


@pytest.fixture
def alice(make_user):
    return Principal(user_id=make_user()[0])


def _request(message="How do graph networks work?", request_id=None):
    return ChatRequest(request_id=request_id or uuid4(), message=message)


def _answer(store, principal, paper, limit, request=None):
    turn = store.begin_turn(
        principal, request or _request(), stale_after=STALE, daily_answers=limit
    )
    store.complete_turn(turn, _result(paper))
    return turn


def test_the_answer_past_the_limit_is_refused_with_the_reset_time(store, alice, paper):
    for _ in range(3):
        _answer(store, alice, paper, 3)
    with pytest.raises(DailyLimitReached) as caught:
        store.begin_turn(alice, _request(), stale_after=STALE, daily_answers=3)
    assert caught.value.resets_at == datetime(2026, 10, 22, tzinfo=UTC)


def test_failed_and_interrupted_turns_do_not_count(store, alice, paper):
    for status in ("failed", "interrupted"):
        turn = store.begin_turn(alice, _request(), stale_after=STALE, daily_answers=2)
        store.fail_turn(turn, status=status, error_code="generation_failed")
    _answer(store, alice, paper, 2)
    _answer(store, alice, paper, 2)
    with pytest.raises(DailyLimitReached):
        store.begin_turn(alice, _request(), stale_after=STALE, daily_answers=2)


def test_a_replayed_request_is_not_counted_again(store, alice, paper):
    request = _request()
    _answer(store, alice, paper, 1, request)
    replay = store.begin_turn(alice, request, stale_after=STALE, daily_answers=1)
    assert replay.replay is not None


def test_the_limit_resets_at_midnight_utc(store, alice, paper, clock):
    _answer(store, alice, paper, 1)
    with pytest.raises(DailyLimitReached):
        store.begin_turn(alice, _request(), stale_after=STALE, daily_answers=1)
    clock.now = datetime(2026, 10, 22, 0, 0, 1, tzinfo=UTC)
    _answer(store, alice, paper, 1)


def test_one_account_s_answers_do_not_count_against_another(store, alice, paper, make_user):
    _answer(store, alice, paper, 1)
    bob = Principal(user_id=make_user()[0])
    _answer(store, bob, paper, 1)


def _race(store, principal) -> list[str]:
    """Two new chats begin at once for one account; what each got."""

    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def begin(message: str) -> None:
        barrier.wait()
        try:
            store.begin_turn(principal, _request(message), stale_after=STALE, daily_answers=2)
            outcomes.append("began")
        except DailyLimitReached:
            outcomes.append("refused")
        except Exception as error:  # noqa: BLE001 - any other outcome is the failure
            outcomes.append(type(error).__name__)

    threads = [threading.Thread(target=begin, args=(f"question {n}",)) for n in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def test_two_turns_started_at_once_cannot_both_pass_the_last_answer(store, paper, make_user):
    """Each round: one answer used of 2, then two new chats at once. Exactly one may begin, and
    the other is refused cleanly (a deadlock between them was found 2026-10-21)."""

    for _ in range(8):
        principal = Principal(user_id=make_user()[0])
        _answer(store, principal, paper, 2)
        assert sorted(_race(store, principal)) == ["began", "refused"]


def test_usage_counts_today_and_this_month(store, alice, paper, clock, search_corpus):
    clock.now = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
    _answer(store, alice, paper, 20)
    clock.now = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
    _answer(store, alice, paper, 20)
    _answer(store, alice, paper, 20)
    now = usage(search_corpus.engine, alice.user_id, limit=20, now=clock.now)
    assert (now.today, now.month, now.limit) == (2, 3, 20)
    assert now.resets_at == datetime(2026, 10, 22, tzinfo=UTC)


def test_me_reports_usage(api_client, make_user, auth_headers):
    user_id, _ = make_user()
    body = api_client.get("/v1/me", headers=auth_headers(user_id)).json()
    assert body["usage"]["today"] == 0 and body["usage"]["month"] == 0
    assert body["usage"]["limit"] == 20
    assert body["usage"]["resets_at"].endswith("T00:00:00Z")
