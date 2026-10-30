"""ChatOpenAI behind our ledger: reserved before it leaves, settled from usage, cached free."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from copilot.models.lc_chat import (
    GROUNDED_ANSWER_SCHEMA,
    LedgerChatModel,
    assert_no_remote_tracing,
)
from copilot.models.llm import LlmError, ResponseCache
from copilot.models.spend import MemoryLedger, Usage, cost_usd, load_llm_config

CONFIG = load_llm_config("configs/llm.yaml")
MODEL = CONFIG.models["openai-gpt-6-luna"]
PROVIDER = CONFIG.providers["openai"]
KEY = SecretStr("sk-test-must-never-appear")
FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "openai_chat.json").read_text()
)
USAGE = Usage(
    input_tokens=FIXTURE["usage"]["prompt_tokens"],
    cached_input_tokens=FIXTURE["usage"]["prompt_tokens_details"]["cached_tokens"],
    output_tokens=FIXTURE["usage"]["completion_tokens"],
)
MESSAGES = [{"role": "system", "content": "Answer as JSON."}, {"role": "user", "content": "Q"}]
GOOD = json.dumps({"claims": [], "insufficient_evidence": True})


def _reply(content: str = GOOD) -> httpx.Response:
    body = json.loads(json.dumps(FIXTURE))
    body["choices"][0]["message"]["content"] = content
    return httpx.Response(200, json=body)


class Recorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return self.responses.pop(0)


class CountingLedger(MemoryLedger):
    def __init__(self, cap: str = "1.00") -> None:
        super().__init__(daily_cap_usd=Decimal(cap))
        self.reservations = 0

    def reserve(self, **kwargs):
        self.reservations += 1
        return super().reserve(**kwargs)


def _model(recorder: Recorder, ledger: MemoryLedger, cache: ResponseCache | None = None):
    return LedgerChatModel(
        MODEL,
        PROVIDER,
        KEY,
        ledger,
        timeout=30.0,
        cache=cache,
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
        sleep=lambda _seconds: None,
    )


def test_a_structured_call_sends_the_schema_and_settles_its_usage():
    recorder = Recorder([_reply()])
    ledger = CountingLedger()
    result = _model(recorder, ledger).complete(
        MESSAGES, purpose="chat", schema=GROUNDED_ANSWER_SCHEMA, run_id="r1"
    )
    body = recorder.bodies[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["name"] == "grounded_answer"
    assert body["max_completion_tokens"] == MODEL.max_output_tokens
    assert (body["reasoning_effort"], body["temperature"], body["seed"]) == ("none", 0, 42)
    assert body["messages"] == [
        {"role": "system", "content": "Answer as JSON."},
        {"role": "user", "content": "Q"},
    ]
    assert result.text == GOOD and not result.cached and result.served_model == FIXTURE["model"]
    assert result.cost_usd == pytest.approx(cost_usd(MODEL.price, USAGE), abs=1e-6)
    assert ledger.reservations == 1
    assert ledger.run_spend("r1") == pytest.approx(result.cost_usd, abs=1e-6)


def test_a_refused_reservation_never_reaches_the_network():
    recorder = Recorder([])
    with pytest.raises(LlmError) as caught:
        _model(recorder, CountingLedger(cap="0.000001")).complete(MESSAGES, purpose="chat")
    assert caught.value.code == "llm_spend_cap" and recorder.bodies == []


def test_a_cache_hit_is_free_and_never_reserved(tmp_path):
    recorder = Recorder([_reply()])
    ledger = CountingLedger()
    model = _model(recorder, ledger, cache=ResponseCache(tmp_path))
    first = model.complete(MESSAGES, purpose="chat")
    second = model.complete(MESSAGES, purpose="chat")
    assert second.text == first.text and second.cached and second.cost_usd == 0.0
    assert len(recorder.bodies) == 1 and ledger.reservations == 1


def test_a_server_error_is_retried_and_each_attempt_is_its_own_reservation():
    recorder = Recorder([httpx.Response(500, json={"error": {"message": "x"}}), _reply()])
    ledger = CountingLedger()
    result = _model(recorder, ledger).complete(MESSAGES, purpose="chat")
    assert result.text == GOOD and len(recorder.bodies) == 2 and ledger.reservations == 2


def test_a_rate_limit_is_unbilled_and_retried():
    limited = httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "1"})
    recorder = Recorder([limited, _reply()])
    ledger = CountingLedger()
    result = _model(recorder, ledger).complete(MESSAGES, purpose="chat", run_id="r2")
    assert ledger.run_spend("r2") == pytest.approx(result.cost_usd, abs=1e-6)


def test_a_bad_request_is_typed_unbilled_and_not_retried():
    recorder = Recorder([httpx.Response(400, json={"error": {"message": "bad"}})])
    ledger = CountingLedger()
    with pytest.raises(LlmError) as caught:
        _model(recorder, ledger).complete(MESSAGES, purpose="chat", run_id="r3")
    assert caught.value.code == "llm_bad_request" and len(recorder.bodies) == 1
    assert ledger.run_spend("r3") == 0.0


@pytest.mark.parametrize("name", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"])
def test_remote_tracing_is_refused(name):
    with pytest.raises(ValueError, match="remote_tracing_enabled"):
        assert_no_remote_tracing({name: "true"})
    assert_no_remote_tracing({name: "false"})


TOOL_CALL = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "openai_tool_call.json").read_text()
)
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search the corpus.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def test_a_tool_call_sends_the_tools_and_settles_its_usage():
    recorder = Recorder([httpx.Response(200, json=TOOL_CALL)])
    ledger = CountingLedger()
    message, cost = _model(recorder, ledger).invoke_tools(
        [SystemMessage("s"), HumanMessage("q")], [SEARCH_TOOL], purpose="evaluation", run_id="t1"
    )
    body = recorder.bodies[0]
    assert body["tools"][0]["function"]["name"] == "search_papers"
    assert body["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
    ]
    assert message.tool_calls[0]["name"] == "search_papers"
    assert message.tool_calls[0]["args"] == {"query": "dropout augmentation sentence embeddings"}
    assert cost > 0 and ledger.reservations == 1
    assert ledger.run_spend("t1") == pytest.approx(cost, abs=1e-6)


def test_a_refused_tool_call_reservation_never_reaches_the_network():
    recorder = Recorder([])
    with pytest.raises(LlmError) as caught:
        _model(recorder, CountingLedger(cap="0.000001")).invoke_tools(
            [HumanMessage("q")], [SEARCH_TOOL], purpose="evaluation"
        )
    assert caught.value.code == "llm_spend_cap" and recorder.bodies == []


def test_a_tool_call_is_retried_after_a_server_error():
    recorder = Recorder(
        [httpx.Response(500, json={"error": {"message": "x"}}), httpx.Response(200, json=TOOL_CALL)]
    )
    ledger = CountingLedger()
    message, _ = _model(recorder, ledger).invoke_tools(
        [HumanMessage("q")], [SEARCH_TOOL], purpose="evaluation"
    )
    assert message.tool_calls and len(recorder.bodies) == 2 and ledger.reservations == 2


def test_a_provider_spelling_max_tokens_differently_is_refused():
    deepseek = CONFIG.models["deepseek-flash"]
    with pytest.raises(ValueError, match="llm_provider_unsupported"):
        LedgerChatModel(deepseek, CONFIG.providers["deepseek"], KEY, CountingLedger(), timeout=30.0)
