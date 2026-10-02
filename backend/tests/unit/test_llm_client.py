"""The OpenAI-compatible chat client, against recorded responses; no network is touched."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from copilot.config import Settings
from copilot.models.llm import ChatClient, LlmError, ResponseCache, build_client
from copilot.models.spend import MemoryLedger, estimate_usd, load_llm_config

CONFIG = load_llm_config("configs/llm.yaml")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"
KEY = SecretStr("sk-test-must-never-appear")
MESSAGES = [{"role": "user", "content": "rank these"}]


def _estimate(model_key):
    """One attempt's reservation, as the ledger records it: rounded to six decimals."""

    model = CONFIG.models[model_key]
    return round(estimate_usd(model.price, len(json.dumps(MESSAGES)), model.max_output_tokens), 6)


def _client(model_key, handler, *, cap="1.00", cache=None, ledger=None):
    model = CONFIG.models[model_key]
    return ChatClient(model, CONFIG.providers[model.provider], KEY,
                      ledger or MemoryLedger(daily_cap_usd=Decimal(cap)),
                      transport=httpx.MockTransport(handler), cache=cache,
                      sleep=lambda _seconds: None)


def _recorded(name, status=200, headers=None):
    body = json.loads((FIXTURES / name).read_text())
    return lambda request: httpx.Response(status, json=body, headers=headers or {})


def test_openai_usage_counts_cached_prompt_tokens():
    result = _client("openai-gpt-6-luna", _recorded("openai_chat.json")).complete(
        MESSAGES, timeout=5.0, purpose="evaluation")
    assert result.usage.cached_input_tokens == 1024   # prompt_tokens_details.cached_tokens
    assert result.served_model.startswith("gpt-6-luna")


def test_deepseek_usage_counts_cache_hit_tokens():
    result = _client("deepseek-flash", _recorded("deepseek_chat.json")).complete(
        MESSAGES, timeout=5.0, purpose="evaluation")
    assert result.usage.cached_input_tokens == 512    # prompt_cache_hit_tokens


def test_the_request_names_the_model_its_token_parameter_and_its_effort():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _recorded("openai_chat.json")(request)

    _client("openai-gpt-6-luna", handler).complete(MESSAGES, timeout=5.0, purpose="search")
    assert seen[0]["model"] == "gpt-6-luna"
    assert seen[0]["max_completion_tokens"] == 1024 and "max_tokens" not in seen[0]
    assert seen[0]["reasoning_effort"] == "none"
    assert seen[0]["messages"] == MESSAGES


def test_the_key_goes_only_in_the_authorization_header():
    seen = []

    def handler(request):
        seen.append(request)
        return _recorded("openai_chat.json")(request)

    _client("openai-gpt-6-luna", handler).complete(MESSAGES, timeout=5.0, purpose="search")
    assert seen[0].headers["authorization"] == "Bearer sk-test-must-never-appear"
    assert str(seen[0].url) == "https://api.openai.com/v1/chat/completions"
    assert b"sk-test" not in seen[0].content


def test_a_successful_call_is_settled_at_its_actual_cost():
    ledger = MemoryLedger(daily_cap_usd=Decimal("1.00"))
    result = _client("openai-gpt-6-luna", _recorded("openai_chat.json"), ledger=ledger).complete(
        MESSAGES, timeout=5.0, purpose="evaluation", run_id="run-a")
    assert not result.cached and result.cost_usd > 0
    assert ledger.run_spend("run-a") == pytest.approx(result.cost_usd)


def test_a_refused_reservation_makes_no_request():
    calls = []
    client = _client("openai-gpt-6-luna", lambda r: calls.append(r), cap="0.000001")
    with pytest.raises(LlmError, match="llm_spend_cap"):
        client.complete(MESSAGES, timeout=5.0, purpose="search")
    assert calls == []


def test_rate_limits_retry_within_the_deadline_then_succeed():
    recorded = json.loads((FIXTURES / "openai_chat.json").read_text())
    responses = iter([httpx.Response(429, headers={"retry-after": "0"}),
                      httpx.Response(503),
                      httpx.Response(200, json=recorded)])
    result = _client("openai-gpt-6-luna", lambda r: next(responses)).complete(
        MESSAGES, timeout=5.0, purpose="search")
    assert result.text


def test_retries_stop_after_three_attempts_and_charge_the_estimate():
    calls = []
    ledger = MemoryLedger(daily_cap_usd=Decimal("1.00"))

    def handler(request):
        calls.append(request)
        return httpx.Response(503, headers={"retry-after": "0"})

    with pytest.raises(LlmError) as raised:
        _client("openai-gpt-6-luna", handler, ledger=ledger).complete(
            MESSAGES, timeout=5.0, purpose="search", run_id="run-b")
    assert raised.value.code == "llm_server_error" and raised.value.retryable
    assert len(calls) == 3
    # A 5xx leaves each attempt's bill unknown, so every attempt keeps its whole estimate.
    assert ledger.run_spend("run-b") == pytest.approx(3 * _estimate("openai-gpt-6-luna"), abs=1e-6)


def test_a_call_refused_outright_is_charged_nothing():
    # Measured 2026-10-02: a 400 from gpt-6-luna was charged its 4,096-token estimate.
    # Every attempt refused with a 4xx means nothing was generated, so nothing is billed.
    ledger = MemoryLedger(daily_cap_usd=Decimal("1.00"))
    rate_limited = lambda r: httpx.Response(429, headers={"retry-after": "0"})  # noqa: E731
    refused = lambda r: httpx.Response(400, json={"error": {"message": "unsupported"}})  # noqa: E731
    for run, handler in (("run-429", rate_limited), ("run-400", refused)):
        with pytest.raises(LlmError) as raised:
            _client("openai-gpt-6-luna-low", handler, ledger=ledger).complete(
                MESSAGES, timeout=5.0, purpose="search", run_id=run)
        assert raised.value.unbilled
        assert ledger.run_spend(run) == 0.0


def test_a_refusal_after_a_server_error_keeps_only_that_attempts_estimate():
    responses = iter([httpx.Response(503, headers={"retry-after": "0"}),
                      httpx.Response(429, headers={"retry-after": "0"}),
                      httpx.Response(429, headers={"retry-after": "0"})])
    ledger = MemoryLedger(daily_cap_usd=Decimal("1.00"))
    with pytest.raises(LlmError, match="llm_rate_limited"):
        _client("openai-gpt-6-luna", lambda r: next(responses), ledger=ledger).complete(
            MESSAGES, timeout=5.0, purpose="search", run_id="run-d")
    assert ledger.run_spend("run-d") == pytest.approx(_estimate("openai-gpt-6-luna"), abs=1e-6)


def test_a_success_after_a_timed_out_attempt_is_charged_for_both():
    # Review finding, 2026-10-02: a timed-out attempt may have been generated and billed,
    # so it keeps its own estimate beside the successful attempt's actual cost.
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return _recorded("openai_chat.json")(request)

    ledger = MemoryLedger(daily_cap_usd=Decimal("1.00"))
    result = _client("openai-gpt-6-luna", handler, ledger=ledger).complete(
        MESSAGES, timeout=5.0, purpose="evaluation", run_id="run-e")
    usage = json.loads((FIXTURES / "openai_chat.json").read_text())["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    actual = ((usage["prompt_tokens"] - cached) * 0.10 + cached * 0.01
              + usage["completion_tokens"] * 0.50) / 1e6
    expected = _estimate("openai-gpt-6-luna") + actual
    assert ledger.run_spend("run-e") == pytest.approx(expected, abs=1e-6)
    assert result.cost_usd == pytest.approx(actual, abs=1e-6)


def test_every_retry_needs_its_own_reservation_under_the_cap():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, headers={"retry-after": "0"})

    one_attempt = Decimal(str(round(_estimate("openai-gpt-6-luna") * 1.5, 6)))
    with pytest.raises(LlmError, match="llm_spend_cap"):
        _client("openai-gpt-6-luna", handler, ledger=MemoryLedger(daily_cap_usd=one_attempt)
                ).complete(MESSAGES, timeout=5.0, purpose="search")
    assert len(calls) == 1


def test_an_unreadable_cache_entry_is_a_miss(tmp_path):
    cache = ResponseCache(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        return _recorded("openai_chat.json")(request)

    client = _client("openai-gpt-6-luna", handler, cache=cache)
    client.complete(MESSAGES, timeout=5.0, purpose="evaluation")
    (entry,) = tmp_path.rglob("*.json")
    entry.write_text("{not json")
    again = client.complete(MESSAGES, timeout=5.0, purpose="evaluation")
    assert len(calls) == 2 and not again.cached


def test_a_failed_cache_write_never_loses_a_paid_answer(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("a file where the cache directory should be")
    result = _client("openai-gpt-6-luna", _recorded("openai_chat.json"),
                     cache=ResponseCache(blocked)).complete(
        MESSAGES, timeout=5.0, purpose="evaluation")
    assert result.text == "ok" and not result.cached


def test_a_transport_timeout_is_typed():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LlmError, match="llm_timeout"):
        _client("openai-gpt-6-luna", handler).complete(MESSAGES, timeout=5.0, purpose="search")


def test_a_rejected_key_is_typed_not_retried_and_never_echoed(caplog):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            401, json={"error": {"message": "Incorrect API key sk-test-must-never-appear"}}
        )

    with pytest.raises(LlmError) as raised:
        _client("deepseek-flash", handler).complete(MESSAGES, timeout=5.0, purpose="search")
    assert raised.value.code == "llm_auth_failed" and len(calls) == 1
    assert "sk-test" not in str(raised.value) and "sk-test" not in caplog.text
    assert "sk-test" not in repr(raised.value)


def test_a_retired_model_is_typed_not_substituted():
    handler = lambda r: httpx.Response(404, json={"error": {"code": "model_not_found"}})  # noqa: E731
    with pytest.raises(LlmError, match="llm_model_unavailable"):
        _client("openai-gpt-6-luna", handler).complete(MESSAGES, timeout=5.0, purpose="search")


def test_another_client_error_is_a_bad_request_and_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "Unsupported value: 'rank these'"}})

    with pytest.raises(LlmError) as raised:
        _client("openai-gpt-6-luna", handler).complete(MESSAGES, timeout=5.0, purpose="search")
    assert raised.value.code == "llm_bad_request" and len(calls) == 1
    assert "rank these" not in str(raised.value)


def test_an_answer_cut_off_at_the_output_cap_is_invalid_not_partial():
    body = json.loads((FIXTURES / "openai_chat.json").read_text())
    body["choices"][0]["finish_reason"] = "length"
    with pytest.raises(LlmError, match="llm_response_invalid"):
        _client("openai-gpt-6-luna", lambda r: httpx.Response(200, json=body)).complete(
            MESSAGES, timeout=5.0, purpose="search")


def test_a_cut_off_answer_is_still_charged_its_reported_usage():
    body = json.loads((FIXTURES / "openai_chat.json").read_text())
    body["choices"][0]["finish_reason"] = "length"
    ledger = MemoryLedger(daily_cap_usd=Decimal("1.00"))
    with pytest.raises(LlmError):
        _client("openai-gpt-6-luna", lambda r: httpx.Response(200, json=body),
                ledger=ledger).complete(MESSAGES, timeout=5.0, purpose="search", run_id="run-c")
    usage = body["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    expected = ((usage["prompt_tokens"] - cached) * 0.10 + cached * 0.01
                + usage["completion_tokens"] * 0.50) / 1e6
    assert ledger.run_spend("run-c") == pytest.approx(expected, abs=1e-6)


def test_a_body_missing_its_content_is_invalid():
    body = json.loads((FIXTURES / "openai_chat.json").read_text())
    del body["choices"][0]["message"]["content"]
    with pytest.raises(LlmError, match="llm_response_invalid"):
        _client("openai-gpt-6-luna", lambda r: httpx.Response(200, json=body)).complete(
            MESSAGES, timeout=5.0, purpose="search")


def test_a_cached_response_is_free_and_stores_no_prompt_text(tmp_path):
    cache = ResponseCache(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        return _recorded("openai_chat.json")(request)

    client = _client("openai-gpt-6-luna", handler, cache=cache)
    first = client.complete(MESSAGES, timeout=5.0, purpose="evaluation")
    second = client.complete(MESSAGES, timeout=5.0, purpose="evaluation")
    assert len(calls) == 1 and second.cached and second.cost_usd == 0.0
    assert second.text == first.text
    assert all("rank these" not in path.read_text() for path in tmp_path.rglob("*.json"))


def test_two_efforts_of_one_model_never_share_a_cached_answer(tmp_path):
    cache = ResponseCache(tmp_path)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["reasoning_effort"])
        return _recorded("openai_chat.json")(request)

    for key in ("openai-gpt-6-luna", "openai-gpt-6-luna-low"):
        _client(key, handler, cache=cache).complete(MESSAGES, timeout=5.0, purpose="evaluation")
    assert calls == ["none", "low"]


def test_build_client_refuses_a_missing_key_or_cap(monkeypatch, tmp_path):
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_DAILY_SPEND_CAP_USD"):
        monkeypatch.delenv(name, raising=False)
    ledger = MemoryLedger(daily_cap_usd=Decimal("1.00"))
    no_key = Settings(_env_file=None, data_dir=str(tmp_path), llm_daily_spend_cap_usd="3")
    with pytest.raises(LlmError, match="llm_unconfigured"):
        build_client("openai-gpt-6-luna", no_key, ledger)
    no_cap = Settings(_env_file=None, data_dir=str(tmp_path), openai_api_key="sk-test-x")
    with pytest.raises(LlmError, match="llm_unconfigured"):
        build_client("openai-gpt-6-luna", no_cap, ledger)
    with pytest.raises(LlmError, match="llm_unconfigured"):
        build_client("no-such-model", no_cap, ledger)
    configured = Settings(_env_file=None, data_dir=str(tmp_path), openai_api_key="sk-test-x",
                          llm_daily_spend_cap_usd="3")
    assert build_client("openai-gpt-6-luna", configured, ledger).model.key == "openai-gpt-6-luna"
