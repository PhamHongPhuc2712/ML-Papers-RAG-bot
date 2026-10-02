"""The PostgreSQL spend ledger: the daily cap holds across concurrent callers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import text

from copilot.models.spend import SpendLedger, Usage, load_llm_config

pytestmark = pytest.mark.integration
NOON = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
MODEL = load_llm_config("configs/llm.yaml").models["deepseek-flash"]


@pytest.fixture
def ledger(migrated_database):
    with migrated_database.begin() as connection:
        connection.execute(text("delete from llm_calls"))
    return SpendLedger(migrated_database, daily_cap_usd=Decimal("1.00"), clock=lambda: NOON)


def test_concurrent_reservations_never_exceed_the_daily_cap(ledger):
    def reserve(_):
        return ledger.reserve(model=MODEL, estimated_usd=0.10, purpose="search",
                              request_id=None, run_id=None)

    with ThreadPoolExecutor(max_workers=20) as pool:
        granted = [call for call in pool.map(reserve, range(20)) if call is not None]
    assert len(granted) == 10


def test_settling_replaces_the_estimate_with_the_actual_cost(ledger, migrated_database):
    call = ledger.reserve(model=MODEL, estimated_usd=0.50, purpose="evaluation",
                          request_id=None, run_id="run-1")
    cost = ledger.settle(call, status="succeeded",
                         usage=Usage(input_tokens=35_000, cached_input_tokens=0, output_tokens=600),
                         served_model="DeepSeek-V4.1-Flash", latency_ms=4200, error_code=None)
    assert cost == pytest.approx((35_000 * 0.30 + 600 * 1.20) / 1e6)
    # The freed headroom is available again the same day.
    assert ledger.reserve(model=MODEL, estimated_usd=0.95, purpose="search",
                          request_id=None, run_id=None) is not None
    assert ledger.run_spend("run-1") == pytest.approx(cost)
    with migrated_database.connect() as connection:
        row = connection.execute(
            text("select status, served_model, input_tokens, output_tokens, latency_ms"
                 " from llm_calls where id = :id"), {"id": call}
        ).one()
    assert tuple(row) == ("succeeded", "DeepSeek-V4.1-Flash", 35_000, 600, 4200)


def test_a_failed_call_without_usage_keeps_its_estimate(ledger):
    call = ledger.reserve(model=MODEL, estimated_usd=0.40, purpose="search",
                          request_id=None, run_id="run-2")
    assert ledger.settle(call, status="failed", usage=None, served_model=None,
                         latency_ms=30_000, error_code="llm_timeout") == pytest.approx(0.40)
    assert ledger.run_spend("run-2") == pytest.approx(0.40)


def test_the_ledger_stores_no_prompt_or_query_text(migrated_database):
    with migrated_database.connect() as connection:
        columns = set(connection.execute(text(
            "select column_name from information_schema.columns where table_name = 'llm_calls'"
        )).scalars())
    assert columns
    assert not columns & {"prompt", "query", "messages", "response", "text"}


def test_a_new_utc_day_starts_with_the_full_cap(migrated_database, ledger):
    assert ledger.reserve(model=MODEL, estimated_usd=1.00, purpose="search",
                          request_id=None, run_id=None)
    assert ledger.reserve(model=MODEL, estimated_usd=0.01, purpose="search",
                          request_id=None, run_id=None) is None
    tomorrow = SpendLedger(migrated_database, daily_cap_usd=Decimal("1.00"),
                           clock=lambda: NOON.replace(day=3))
    assert tomorrow.reserve(model=MODEL, estimated_usd=1.00, purpose="search",
                            request_id=None, run_id=None)
