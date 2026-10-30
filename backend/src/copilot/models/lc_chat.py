"""LangChain's ChatOpenAI behind our spend ledger and response cache (spec §12).

Each request is reserved at its worst case by a callback that runs before the request
leaves — a refusal raises there, so nothing is sent — and settled from the usage the
provider reports. LangChain's own retries and cache are off: one invoke is one request is
one ledger row. Our response cache sits in front of the model, so a hit never reaches
LangChain and is never reserved. Structured output is requested as a JSON-schema dict,
which keeps the raw text for our own validation; a Pydantic schema would be parsed inside
the OpenAI SDK, which raises on malformed text and loses it.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from uuid import UUID

import httpx
import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..config import Settings
from .llm import MAX_ATTEMPTS, ChatResult, LlmError, ResponseCache, _api_key, _retry_after
from .spend import LlmModel, ProviderSpec, SpendGate, Usage, estimate_usd, load_llm_config

TRACING_VARIABLES = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
_TRUE = {"1", "true", "yes", "on"}

GROUNDED_ANSWER_SCHEMA: dict[str, Any] = {
    "title": "grounded_answer",
    "description": "Claims about the question, each citing the evidence ids it rests on.",
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["claims", "insufficient_evidence"],
    "additionalProperties": False,
}


def assert_no_remote_tracing(environ: Mapping[str, str] = os.environ) -> None:
    """LangChain sends every run to LangSmith when one of these is set; chat text must not."""

    for name in TRACING_VARIABLES:
        if environ.get(name, "").strip().lower() in _TRUE:
            raise ValueError(f"remote_tracing_enabled:{name}")


def _lc_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    kinds: dict[str, Callable[[str], BaseMessage]] = {
        "system": lambda content: SystemMessage(content=content),
        "user": lambda content: HumanMessage(content=content),
        "assistant": lambda content: AIMessage(content=content),
    }
    return [kinds[item["role"]](item["content"]) for item in messages]


def _usage(metadata: Mapping[str, Any] | None) -> Usage | None:
    """None when the provider reported nothing: the call then settles at its estimate."""

    if not metadata:
        return None
    details = metadata.get("input_token_details") or {}
    return Usage(
        input_tokens=int(metadata.get("input_tokens", 0)),
        cached_input_tokens=int(details.get("cache_read", 0) or 0),
        output_tokens=int(metadata.get("output_tokens", 0)),
    )


def _sdk_response(error: openai.APIStatusError) -> Any:
    """The SDK's response: openai 3 uses httpx2's, whose headers read exactly as httpx's do."""

    return error.response


def _llm_error(error: BaseException) -> LlmError:
    """The same codes ChatClient uses, so callers handle one vocabulary."""

    if isinstance(error, LlmError):
        return error
    if isinstance(error, openai.APITimeoutError):
        return LlmError("llm_timeout", retryable=True)
    if isinstance(error, openai.APIConnectionError):
        return LlmError("llm_server_error", retryable=True)
    if isinstance(error, openai.APIStatusError):
        status = error.status_code
        if status in (401, 403):
            return LlmError("llm_auth_failed", status=status, unbilled=True)
        if status == 404:
            return LlmError("llm_model_unavailable", status=status, unbilled=True)
        if status == 429:
            return LlmError(
                "llm_rate_limited",
                retryable=True,
                status=status,
                unbilled=True,
                retry_after=_retry_after(_sdk_response(error)),
            )
        if status >= 500:
            return LlmError(
                "llm_server_error",
                retryable=True,
                status=status,
                retry_after=_retry_after(_sdk_response(error)),
            )
        return LlmError("llm_bad_request", status=status, unbilled=True)
    return LlmError("llm_client_error")


class LedgerCallback(BaseCallbackHandler):
    """Reserve before the request leaves; settle when it ends, however it ends."""

    # A refused reservation must stop the request, not be logged and ignored.
    raise_error = True

    def __init__(
        self,
        ledger: SpendGate,
        model: LlmModel,
        *,
        estimate: float,
        purpose: str,
        request_id: UUID | None,
        run_id: str | None,
        clock: Callable[[], float],
    ) -> None:
        self._ledger = ledger
        self._model = model
        self._estimate = estimate
        self._purpose = purpose
        self._request_id = request_id
        self._run_id = run_id
        self._clock = clock
        self._started = 0.0
        self.call_id: UUID | None = None
        self.cost_usd = 0.0
        self.usage: Usage | None = None
        self.served_model = model.model

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        call = self._ledger.reserve(
            model=self._model,
            estimated_usd=self._estimate,
            purpose=self._purpose,
            request_id=self._request_id,
            run_id=self._run_id,
        )
        if call is None:
            raise LlmError("llm_spend_cap")
        self.call_id = call
        self._started = self._clock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        generation = response.generations[0][0]
        if isinstance(generation, ChatGeneration):
            self.usage = _usage(getattr(generation.message, "usage_metadata", None))
        name = (response.llm_output or {}).get("model_name")
        if isinstance(name, str):
            self.served_model = name
        self.cost_usd = self._settle("succeeded", None, self.usage)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        if self.call_id is None:
            return
        failure = _llm_error(error)
        self.cost_usd = self._settle(
            "failed", failure.code, Usage(0, 0, 0) if failure.unbilled else None
        )

    def _settle(self, status: str, error_code: str | None, usage: Usage | None) -> float:
        assert self.call_id is not None
        return self._ledger.settle(
            self.call_id,
            status=status,
            usage=usage,
            served_model=self.served_model,
            latency_ms=int((self._clock() - self._started) * 1000),
            error_code=error_code,
        )


class LedgerChatModel:
    """One configured model through ChatOpenAI, every request under the spend ledger."""

    def __init__(
        self,
        model: LlmModel,
        provider: ProviderSpec,
        api_key: SecretStr,
        ledger: SpendGate,
        *,
        timeout: float,
        cache: ResponseCache | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if provider.name != model.provider:
            raise ValueError("llm_provider_mismatch")
        # ChatOpenAI always spells the output cap max_completion_tokens.
        if provider.max_tokens_param != "max_completion_tokens":
            raise ValueError(f"llm_provider_unsupported:{provider.name}")
        assert_no_remote_tracing()
        self.model = model
        self.provider = provider
        self._ledger = ledger
        self._cache = cache
        self._timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._extra = {**provider.request_extra, **model.request}
        self._chat = ChatOpenAI(
            model=model.model,
            base_url=provider.base_url,
            api_key=api_key,
            # The field's alias, which is also the name sent on the wire.
            max_completion_tokens=model.max_output_tokens,
            timeout=timeout,
            max_retries=0,
            use_responses_api=False,
            extra_body=dict(self._extra),
            http_client=http_client,
        )

    def _wire_body(
        self, messages: list[dict[str, str]], schema: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """What identifies a request: the cache key's input, and nothing secret."""

        return {
            **self._extra,
            "model": self.model.model,
            "messages": messages,
            "max_completion_tokens": self.model.max_output_tokens,
            "response_format": None if schema is None else dict(schema),
        }

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        purpose: str,
        schema: Mapping[str, Any] | None = None,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> ChatResult:
        started = self._clock()
        key = ResponseCache.key(self.provider.base_url, self._wire_body(messages, schema))
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                elapsed = int((self._clock() - started) * 1000)
                return ChatResult(hit[0], hit[1], hit[2], 0.0, True, elapsed)
        runnable: Any = self._chat
        if schema is not None:
            runnable = self._chat.with_structured_output(
                dict(schema), method="json_schema", strict=True, include_raw=True
            )
        estimate = estimate_usd(
            self.model.price,
            len(json.dumps(messages)) + len(json.dumps(schema or {})),
            self.model.max_output_tokens,
        )
        output, gate, spent = self._invoke(
            runnable,
            _lc_messages(messages),
            estimate=estimate,
            purpose=purpose,
            request_id=request_id,
            run_id=run_id,
            started=started,
        )
        message = output["raw"] if schema is not None else output
        content = message.content if isinstance(message.content, str) else ""
        usage = gate.usage or Usage(0, 0, 0)
        if self._cache is not None:
            self._cache.put(key, content, usage, gate.served_model)
        latency = int((self._clock() - started) * 1000)
        return ChatResult(content, usage, gate.served_model, spent, False, latency)

    def invoke_tools(
        self,
        messages: list[BaseMessage],
        tools: Sequence[Mapping[str, Any]],
        *,
        purpose: str,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> tuple[AIMessage, float]:
        """One tool-calling request under the ledger: the model's message and what it cost.

        Never cached: a cache entry holds text, and a tool step's answer is a message with
        calls, so a rerun of the agent spends again.
        """

        started = self._clock()
        wire = json.dumps([message.model_dump(mode="json") for message in messages])
        estimate = estimate_usd(
            self.model.price,
            len(wire) + len(json.dumps(list(tools))),
            self.model.max_output_tokens,
        )
        output, _, spent = self._invoke(
            self._chat.bind_tools([dict(tool) for tool in tools]),
            messages,
            estimate=estimate,
            purpose=purpose,
            request_id=request_id,
            run_id=run_id,
            started=started,
        )
        if not isinstance(output, AIMessage):
            raise LlmError("llm_client_error")
        return output, spent

    def _invoke(
        self,
        runnable: Any,
        messages: list[BaseMessage],
        *,
        estimate: float,
        purpose: str,
        request_id: UUID | None,
        run_id: str | None,
        started: float,
    ) -> tuple[Any, LedgerCallback, float]:
        """Each attempt reserved and settled on its own; retried while the deadline allows."""

        spent = 0.0
        attempt = 0
        while True:
            attempt += 1
            gate = LedgerCallback(
                self._ledger,
                self.model,
                estimate=estimate,
                purpose=purpose,
                request_id=request_id,
                run_id=run_id,
                clock=self._clock,
            )
            try:
                output = runnable.invoke(messages, config={"callbacks": [gate]})
            except Exception as error:  # noqa: BLE001 - every failure becomes a typed LlmError
                spent += gate.cost_usd
                failure = _llm_error(error)
                remaining = self._timeout - (self._clock() - started)
                if not failure.retryable or attempt >= MAX_ATTEMPTS or remaining <= 0:
                    raise failure from error
                wait = 2.0**attempt if failure.retry_after is None else failure.retry_after
                self._sleep(min(wait, remaining))
                continue
            return output, gate, spent + gate.cost_usd


def build_ledger_chat_model(
    model_key: str,
    settings: Settings,
    ledger: SpendGate,
    *,
    timeout: float,
    cache: ResponseCache | None = None,
    llm_config: str = "configs/llm.yaml",
) -> LedgerChatModel:
    """A model from configuration, or ``llm_unconfigured`` when its key or the cap is missing."""

    config = load_llm_config(llm_config)
    model = config.models.get(model_key)
    if model is None:
        raise LlmError("llm_unconfigured")
    provider = config.providers[model.provider]
    key = _api_key(settings, provider)
    if key is None or settings.llm_daily_spend_cap_usd is None:
        raise LlmError("llm_unconfigured")
    return LedgerChatModel(model, provider, key, ledger, timeout=timeout, cache=cache)
