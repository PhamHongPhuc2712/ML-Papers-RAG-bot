"""The listwise prompt and the strict permutation parser; no network is touched."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from copilot.models.llm import ChatClient, LlmError
from copilot.models.spend import MemoryLedger, load_llm_config
from copilot.search.llm_rerank import (
    LlmListwiseReranker,
    build_messages,
    load_prompt,
    parse_permutation,
)
from copilot.search.rerank import RerankError

PROMPT = load_prompt("prompts/rerank/listwise-v1.yaml")
CONFIG = load_llm_config("configs/llm.yaml")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"


def test_a_clean_answer_is_the_permutation_it_names():
    assert parse_permutation("[3] > [1] > [2]", 3) == ([2, 0, 1], 3)


def test_duplicates_and_out_of_range_ids_are_dropped_and_the_unranked_appended():
    order, ranked = parse_permutation("[2] > [2] > [9] > [0] > [4]", 5)
    assert order == [1, 3, 0, 2, 4] and ranked == 2


def test_bare_numbers_are_read_when_the_model_drops_the_brackets():
    assert parse_permutation("2 > 1", 2) == ([1, 0], 2)


def test_an_answer_with_no_identifier_is_unparseable():
    with pytest.raises(RerankError, match="llm_rerank_unparseable"):
        parse_permutation("I cannot rank these papers.", 4)


def test_a_number_in_prose_is_not_a_ranking():
    with pytest.raises(RerankError, match="llm_rerank_unparseable"):
        parse_permutation("I cannot rank these 4 papers.", 4)


def test_every_candidate_is_shown_once_truncated_to_the_word_budget():
    texts = ["alpha " * 400, "beta gamma"]
    user = build_messages(PROMPT, "graph models", texts, words=300)[1]["content"]
    # The prompt's own example answer also contains "[1] ", so match the candidate lines.
    assert user.count("[1] alpha") == 1 and user.count("[2] beta gamma") == 1
    assert user.count("alpha") == 300 and "Rank all 2 candidates" in user


def test_braces_in_the_query_or_an_abstract_are_text_not_template():
    user = build_messages(PROMPT, "{candidates} as a query", ["uses {query} literally"], 300)[1]
    assert "{candidates} as a query" in user["content"]
    assert "uses {query} literally" in user["content"]


def test_the_identity_changes_with_the_prompt_text(tmp_path):
    edited = tmp_path / "p.yaml"
    edited.write_text(
        Path("prompts/rerank/listwise-v1.yaml").read_text().replace("Rank all", "Order all")
    )
    assert load_prompt(edited).sha256 != PROMPT.sha256


def _client(answer=None, status=200):
    model = CONFIG.models["openai-gpt-6-luna"]
    body = json.loads((FIXTURES / "openai_chat.json").read_text())
    if answer is not None:
        body["choices"][0]["message"]["content"] = answer
    return ChatClient(model, CONFIG.providers["openai"], SecretStr("sk-test-x"),
                      MemoryLedger(daily_cap_usd=Decimal("1.00")),
                      transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body)),
                      sleep=lambda _seconds: None)


def test_the_reranker_orders_the_candidates_and_reports_what_it_cost():
    reranker = LlmListwiseReranker(_client("[2] > [3]"), PROMPT)
    result = reranker.order("graph models", ["a", "b", "c"], timeout=5.0, purpose="search")
    assert result.order == [1, 2, 0] and result.ranked_by_model == 2
    assert result.served_model == "gpt-6-luna" and not result.cached and result.cost_usd > 0
    assert reranker.identity == (
        f"openai-gpt-6-luna=gpt-6-luna#listwise-v1:{PROMPT.sha256[:12]}#w300"
    )


def test_the_reranker_lets_failures_through_for_the_service_to_handle():
    with pytest.raises(RerankError, match="llm_rerank_unparseable"):
        LlmListwiseReranker(_client("No ranking."), PROMPT).order(
            "q", ["a", "b"], timeout=5.0, purpose="search")
    with pytest.raises(LlmError, match="llm_server_error"):
        LlmListwiseReranker(_client(status=500), PROMPT).order(
            "q", ["a", "b"], timeout=5.0, purpose="search")


def test_the_api_builds_the_llm_reranker_only_when_fully_configured(monkeypatch, tmp_path):
    from copilot.config import Settings
    from copilot.search.api import default_listwise

    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_DAILY_SPEND_CAP_USD",
                 "LLM_RERANK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    engine = object()  # never touched: building makes no call and no query

    def settings(**values):
        return Settings(_env_file=None, data_dir=str(tmp_path), **values)

    assert default_listwise(settings(), engine) is None
    unconfigured = (
        {"llm_rerank_model": "openai-gpt-6-luna"},
        {"llm_rerank_model": "openai-gpt-6-luna", "llm_daily_spend_cap_usd": "3"},
        {"llm_rerank_model": "openai-gpt-6-luna", "openai_api_key": "sk-test-x"},
        {"llm_rerank_model": "no-such-model", "openai_api_key": "sk-test-x",
         "llm_daily_spend_cap_usd": "3"},
    )
    for values in unconfigured:
        assert default_listwise(settings(**values), engine) is None
    built = default_listwise(settings(llm_rerank_model="openai-gpt-6-luna",
                                      openai_api_key="sk-test-x",
                                      llm_daily_spend_cap_usd="3"), engine)
    assert built is not None and built.identity.startswith("openai-gpt-6-luna=gpt-6-luna#")
