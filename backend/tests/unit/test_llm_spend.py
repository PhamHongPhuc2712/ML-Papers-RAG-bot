"""Prices, cost, the model config and the in-process spend ledger; no services."""

from __future__ import annotations

from decimal import Decimal

import pytest

from copilot.config import Settings
from copilot.models.spend import (
    MemoryLedger,
    ModelPrice,
    SpendLedger,
    Usage,
    cost_usd,
    estimate_usd,
    load_llm_config,
)

PRICE = ModelPrice(input=2.50, cached_input=1.25, output=10.00)
LLM_ENV = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_DAILY_SPEND_CAP_USD")


def test_cost_charges_cached_input_at_its_own_rate():
    usage = Usage(input_tokens=40_000, cached_input_tokens=10_000, output_tokens=600)
    assert cost_usd(PRICE, usage) == pytest.approx(
        (30_000 * 2.50 + 10_000 * 1.25 + 600 * 10.00) / 1_000_000
    )


def test_the_estimate_is_an_upper_bound_on_any_answer_within_max_tokens():
    # 4 characters per token is typical English; the estimate assumes 3, so it over-reserves.
    estimate = estimate_usd(PRICE, prompt_chars=120_000, max_output_tokens=1_024)
    actual = cost_usd(PRICE, Usage(input_tokens=30_000, cached_input_tokens=0, output_tokens=1_024))
    assert estimate >= actual


def test_a_model_without_a_price_cannot_be_configured(tmp_path):
    path = tmp_path / "llm.yaml"
    path.write_text(
        "schema_version: 1\n"
        "providers: {openai: {base_url: 'https://api.openai.com/v1', api_key_env: OPENAI_API_KEY,"
        " max_tokens_param: max_completion_tokens, usage: openai}}\n"
        "models: {m: {provider: openai, model: gpt-6-luna, pinned: false,"
        " max_output_tokens: 1024}}\n"
    )
    with pytest.raises(ValueError, match="llm_config_invalid:models.m.price"):
        load_llm_config(path)


@pytest.mark.parametrize(
    ("models", "field"),
    [
        ("{m: {provider: other, model: x, pinned: false, max_output_tokens: 8,"
         " price: {input: 1, cached_input: 1, output: 1}}}", "models.m.provider"),
        ("{m: {provider: openai, model: x, pinned: false, max_output_tokens: 0,"
         " price: {input: 1, cached_input: 1, output: 1}}}", "models.m.max_output_tokens"),
        ("{m: {provider: openai, model: x, pinned: false, max_output_tokens: 8,"
         " price: {input: -1, cached_input: 1, output: 1}}}", "models.m.price"),
    ],
)
def test_an_unknown_provider_an_empty_budget_or_a_negative_price_is_refused(
    tmp_path, models, field
):
    path = tmp_path / "llm.yaml"
    path.write_text(
        "schema_version: 1\n"
        "providers: {openai: {base_url: 'https://api.openai.com/v1', api_key_env: OPENAI_API_KEY,"
        " max_tokens_param: max_completion_tokens, usage: openai}}\n"
        f"models: {models}\n"
    )
    with pytest.raises(ValueError, match=f"llm_config_invalid:{field}"):
        load_llm_config(path)


def test_the_repository_config_prices_every_model():
    config = load_llm_config("configs/llm.yaml")
    assert {"openai-gpt-6-luna", "openai-gpt-6-luna-low", "deepseek-flash"} <= set(config.models)
    assert config.models["openai-gpt-6-luna"].request["reasoning_effort"] == "none"
    assert config.models["openai-gpt-6-luna-low"].request["reasoning_effort"] == "low"
    assert config.models["deepseek-flash"].pinned is False
    assert config.models["deepseek-flash"].request == {}


def test_the_memory_ledger_refuses_past_the_cap_and_charges_failures_their_estimate():
    model = load_llm_config("configs/llm.yaml").models["deepseek-flash"]
    ledger = MemoryLedger(daily_cap_usd=Decimal("0.25"))
    first = ledger.reserve(model=model, estimated_usd=0.10, purpose="evaluation",
                           request_id=None, run_id="r")
    second = ledger.reserve(model=model, estimated_usd=0.10, purpose="evaluation",
                            request_id=None, run_id="r")
    assert first and second
    assert ledger.reserve(model=model, estimated_usd=0.10, purpose="evaluation",
                          request_id=None, run_id="r") is None
    # A failed call with unknown usage keeps its whole estimate: it may have been billed.
    assert ledger.settle(second, status="failed", usage=None, served_model=None,
                         latency_ms=5, error_code="timeout") == pytest.approx(0.10)
    assert ledger.run_spend("r") == pytest.approx(0.20)


def test_a_settled_call_frees_the_rest_of_its_estimate():
    model = load_llm_config("configs/llm.yaml").models["openai-gpt-6-luna"]
    ledger = MemoryLedger(daily_cap_usd=Decimal("0.10"))
    call = ledger.reserve(model=model, estimated_usd=0.08, purpose="search",
                          request_id=None, run_id=None)
    cost = ledger.settle(call, status="succeeded",
                         usage=Usage(input_tokens=35_000, cached_input_tokens=0, output_tokens=600),
                         served_model="gpt-6-luna", latency_ms=900, error_code=None)
    assert cost == pytest.approx((35_000 * 0.10 + 600 * 0.50) / 1e6)
    assert ledger.reserve(model=model, estimated_usd=0.09, purpose="search",
                          request_id=None, run_id=None) is not None


def test_a_spend_ledger_without_a_cap_cannot_be_built():
    with pytest.raises(ValueError, match="llm_spend_cap_unset"):
        SpendLedger(None, daily_cap_usd=None)


def test_blank_llm_settings_mean_unset(monkeypatch, tmp_path):
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, data_dir=str(tmp_path), openai_api_key="",
                        deepseek_api_key="  ", llm_daily_spend_cap_usd="")
    assert settings.openai_api_key is None
    assert settings.deepseek_api_key is None
    assert settings.llm_daily_spend_cap_usd is None


def test_the_cap_and_keys_are_read_and_the_keys_stay_masked(monkeypatch, tmp_path):
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-must-never-appear")
    monkeypatch.setenv("LLM_DAILY_SPEND_CAP_USD", "3")
    settings = Settings(_env_file=None, data_dir=str(tmp_path))
    assert settings.llm_daily_spend_cap_usd == Decimal("3")
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test-must-never-appear"
    assert "sk-test" not in repr(settings) and "sk-test" not in str(settings.model_dump())


def test_a_cap_of_zero_or_less_is_refused(monkeypatch, tmp_path):
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="llm_daily_spend_cap_usd"):
        Settings(_env_file=None, data_dir=str(tmp_path), llm_daily_spend_cap_usd="0")
