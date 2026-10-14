"""A day's chat turns: counts by status and error, stage timings and cost."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest

from copilot.chat.store import ChatStore, chat_stats
from copilot.chat.turns import TurnResult
from copilot.contracts import ChatRequest, Claim, Evidence, GroundedAnswer, Principal

pytestmark = pytest.mark.integration
DAY = date(2026, 1, 5)
TRACE = {"condense": 0.1, "retrieve": 0.5, "generate": 2.0}


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
        trace=TRACE,
        cost_usd=0.001,
    )


def test_a_days_turns_by_status_error_stage_and_cost(search_corpus, make_user):
    store = ChatStore(search_corpus.engine, clock=lambda: datetime(2026, 1, 5, 12, tzinfo=UTC))
    principal = Principal(user_id=make_user()[0])
    paper = search_corpus.ids["Graph neural networks for molecules"]
    for _ in range(2):
        turn = store.begin_turn(
            principal,
            ChatRequest(request_id=uuid4(), message="How do graph networks work?"),
            stale_after=timedelta(minutes=4),
        )
        store.complete_turn(turn, _result(paper))
    failed = store.begin_turn(
        principal,
        ChatRequest(request_id=uuid4(), message="And their limits?"),
        stale_after=timedelta(minutes=4),
    )
    store.fail_turn(failed, status="failed", error_code="llm_timeout")

    stats = chat_stats(search_corpus.engine, day=DAY)
    assert stats["day"] == "2026-01-05"
    assert stats["turns"] == 3
    assert stats["statuses"] == {"complete": 2, "failed": 1}
    assert stats["error_codes"] == {"llm_timeout": 1}
    assert stats["stages"]["retrieve"]["p50"] == pytest.approx(0.5)
    assert stats["stages"]["generate"]["p95"] == pytest.approx(2.0)
    assert stats["cost_usd"] == pytest.approx(0.002)
    assert chat_stats(search_corpus.engine, day=date(2026, 1, 6))["turns"] == 0
