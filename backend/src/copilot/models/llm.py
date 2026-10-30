"""One client for OpenAI-compatible Chat Completions: OpenAI and DeepSeek.

Every request goes through the spend ledger (``models/spend.py``). Each attempt
reserves its own worst case before it is sent, and a refusal raises
``llm_spend_cap`` with no request made. A retry is a new request the provider
may bill, so it is a new reservation too. Each attempt then settles on every
path:
* at the cost its usage implies, when the provider reports usage;
* at zero, when the provider refused it outright with a 4xx;
* at its whole estimate, when a timeout, a 5xx or a dropped connection leaves
  the bill unknown.
Rate limits, server errors and transport timeouts are retried while the caller's
deadline allows. A response cut off at the output cap is invalid, never a
partial answer. httpx applies the deadline to each phase (connect, each read),
not to the whole exchange, so a response that trickles in can run past it. The
search service stops waiting at its own budget either way.

Errors carry a code and, at most, an HTTP status. A provider's error body can
echo the key or the prompt, so no body, header or message ever reaches an
exception, a log or the cache. Responses are cached by the digest of the whole
request, so a rerun of an evaluation is free and identical; the cache keeps the
answer, its usage and the served model, never the messages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from pydantic import SecretStr

from ..config import Settings
from .spend import (
    LlmModel,
    ProviderSpec,
    SpendGate,
    Usage,
    estimate_usd,
    load_llm_config,
)

MAX_ATTEMPTS = 3
ZERO_USAGE = Usage(input_tokens=0, cached_input_tokens=0, output_tokens=0)

logger = logging.getLogger(__name__)


class LlmError(RuntimeError):
    """A hosted-LLM call failed. The message is the code and, at most, an HTTP status.

    ``unbilled`` is True when the attempt that raised was refused outright with a
    4xx, so the provider generated nothing for it. ``retry_after`` is the
    provider's requested wait, when it gave one.
    """

    def __init__(
        self,
        code: str,
        retryable: bool = False,
        status: int | None = None,
        *,
        unbilled: bool = False,
        retry_after: float | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.status = status
        self.unbilled = unbilled
        self.retry_after = retry_after
        super().__init__(code if status is None else f"{code}:http_{status}")


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: Usage
    served_model: str
    cost_usd: float
    cached: bool
    latency_ms: int


class ResponseCache:
    """Answers keyed by the sha256 of the whole request; never the messages themselves."""

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    @staticmethod
    def key(base_url: str, body: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            {"base_url": base_url, "body": body}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> tuple[str, Usage, str] | None:
        """The cached answer, or None. An unreadable entry is a miss, never an error."""

        path = self._path(key)
        try:
            if not path.exists():
                return None
            entry = json.loads(path.read_text(encoding="utf-8"))
            usage = entry["usage"]
            return (
                str(entry["text"]),
                Usage(
                    input_tokens=int(usage["input_tokens"]),
                    cached_input_tokens=int(usage["cached_input_tokens"]),
                    output_tokens=int(usage["output_tokens"]),
                ),
                str(entry["served_model"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("unreadable LLM cache entry %s treated as a miss", key[:12])
            return None

    def put(self, key: str, text: str, usage: Usage, served_model: str) -> None:
        """Store an answer. A paid answer is never lost to a failed write: it is logged."""

        path = self._path(key)
        entry = {
            "text": text,
            "usage": {
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "output_tokens": usage.output_tokens,
            },
            "served_model": served_model,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            partial.write_text(json.dumps(entry, sort_keys=True), encoding="utf-8")
            partial.replace(path)
        except OSError as error:
            logger.warning("LLM cache entry %s not written (%s)", key[:12], type(error).__name__)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _count(section: Mapping[str, Any], name: str) -> int:
    value = section.get(name, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LlmError("llm_response_invalid")
    return int(value)


def _usage(payload: Mapping[str, Any], style: str) -> Usage:
    usage = payload.get("usage")
    if not isinstance(usage, dict) or "prompt_tokens" not in usage:
        raise LlmError("llm_response_invalid")
    if "completion_tokens" not in usage:
        raise LlmError("llm_response_invalid")
    if style == "openai":
        details = usage.get("prompt_tokens_details") or {}
        cached = _count(details, "cached_tokens") if isinstance(details, dict) else 0
    else:
        cached = _count(usage, "prompt_cache_hit_tokens")
    # OpenAI's completion_tokens already includes reasoning tokens, billed as output.
    return Usage(
        input_tokens=_count(usage, "prompt_tokens"),
        cached_input_tokens=cached,
        output_tokens=_count(usage, "completion_tokens"),
    )


def _answer(payload: Mapping[str, Any]) -> tuple[str, str]:
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        served = payload["model"]
    except (KeyError, IndexError, TypeError) as error:
        raise LlmError("llm_response_invalid") from error
    if not isinstance(content, str) or not isinstance(served, str):
        raise LlmError("llm_response_invalid")
    # An answer cut off at the output cap is never parsed as a partial ranking.
    if choice.get("finish_reason") == "length":
        raise LlmError("llm_response_invalid")
    return content, served


class ChatClient:
    """Chat Completions for one configured model, behind the spend ledger."""

    def __init__(
        self,
        model: LlmModel,
        provider: ProviderSpec,
        api_key: SecretStr,
        ledger: SpendGate,
        *,
        transport: httpx.BaseTransport | None = None,
        cache: ResponseCache | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if provider.name != model.provider:
            raise ValueError("llm_provider_mismatch")
        self.model = model
        self.provider = provider
        self._key = api_key
        self._ledger = ledger
        self._cache = cache
        self._sleep = sleep
        self._http = httpx.Client(transport=transport)
        self._url = f"{provider.base_url}/chat/completions"

    def close(self) -> None:
        self._http.close()

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            **self.provider.request_extra,
            **self.model.request,
            "model": self.model.model,
            "messages": messages,
            self.provider.max_tokens_param: self.model.max_output_tokens,
        }
        return body

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float,
        purpose: str,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> ChatResult:
        started = time.monotonic()
        deadline = started + timeout
        body = self._body(messages)
        key = ResponseCache.key(self.provider.base_url, body)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                elapsed = int((time.monotonic() - started) * 1000)
                return ChatResult(hit[0], hit[1], hit[2], 0.0, True, elapsed)

        estimate = estimate_usd(
            self.model.price, len(json.dumps(messages)), self.model.max_output_tokens
        )
        spent = 0.0
        attempt = 0
        while True:
            attempt += 1
            if deadline - time.monotonic() <= 0:
                raise LlmError("llm_timeout", retryable=True)
            try:
                text, usage, served, cost = self._attempt(
                    body, estimate, deadline, purpose, request_id, run_id
                )
            except LlmError as error:
                if not error.retryable or attempt >= MAX_ATTEMPTS:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                wait = 2.0**attempt if error.retry_after is None else error.retry_after
                self._sleep(min(wait, remaining))
                continue
            spent += cost
            break
        if self._cache is not None:
            self._cache.put(key, text, usage, served)
        latency = int((time.monotonic() - started) * 1000)
        return ChatResult(text, usage, served, spent, False, latency)

    def _attempt(
        self,
        body: Mapping[str, Any],
        estimate: float,
        deadline: float,
        purpose: str,
        request_id: UUID | None,
        run_id: str | None,
    ) -> tuple[str, Usage, str, float]:
        """One request under its own reservation, settled whatever happens to it."""

        call = self._ledger.reserve(
            model=self.model,
            estimated_usd=estimate,
            purpose=purpose,
            request_id=request_id,
            run_id=run_id,
        )
        if call is None:
            raise LlmError("llm_spend_cap")
        started = time.monotonic()
        # Usage is recorded as soon as it is known: a cut-off answer was still billed.
        usage: Usage | None = None
        served: str | None = None

        def settle(status: str, error_code: str | None) -> float:
            return self._ledger.settle(
                call,
                status=status,
                usage=usage,
                served_model=served,
                latency_ms=int((time.monotonic() - started) * 1000),
                error_code=error_code,
            )

        try:
            payload = self._post(body, deadline=deadline)
            usage = _usage(payload, self.provider.usage)
            model_name = payload.get("model")
            served = model_name if isinstance(model_name, str) else None
            text, served_model = _answer(payload)
        except LlmError as error:
            if usage is None and error.unbilled:
                # Refused outright: nothing was generated, so nothing is charged.
                usage = ZERO_USAGE
            settle("failed", error.code)
            raise
        except Exception:
            # Unexpected: the reservation keeps its whole estimate.
            settle("failed", "llm_client_error")
            raise
        return text, usage, served_model, settle("succeeded", None)

    def _post(self, body: Mapping[str, Any], *, deadline: float) -> dict[str, Any]:
        """One HTTP request. Its failure says whether a retry may help and whether it cost."""

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LlmError("llm_timeout", retryable=True, unbilled=True)
        headers = {"Authorization": f"Bearer {self._key.get_secret_value()}"}
        try:
            response = self._http.post(self._url, json=body, headers=headers, timeout=remaining)
        except httpx.TimeoutException as error:
            raise LlmError("llm_timeout", retryable=True) from error
        except httpx.TransportError as error:
            raise LlmError("llm_server_error", retryable=True) from error
        code = response.status_code
        if code == 200:
            try:
                payload = response.json()
            except ValueError as error:
                raise LlmError("llm_response_invalid") from error
            if not isinstance(payload, dict):
                raise LlmError("llm_response_invalid")
            return payload
        if code in (401, 403):
            raise LlmError("llm_auth_failed", status=code, unbilled=True)
        if code == 404 or _error_code(response) == "model_not_found":
            raise LlmError("llm_model_unavailable", status=code, unbilled=True)
        if code == 429:
            raise LlmError(
                "llm_rate_limited",
                retryable=True,
                status=code,
                unbilled=True,
                retry_after=_retry_after(response),
            )
        if code >= 500:
            raise LlmError(
                "llm_server_error", retryable=True, status=code, retry_after=_retry_after(response)
            )
        raise LlmError("llm_bad_request", status=code, unbilled=True)


def _api_key(settings: Settings, provider: ProviderSpec) -> SecretStr | None:
    value = getattr(settings, provider.api_key_env.lower(), None)
    if not isinstance(value, SecretStr) or not value.get_secret_value().strip():
        return None
    return value


def build_client(
    model_key: str,
    settings: Settings,
    ledger: SpendGate,
    *,
    cache: ResponseCache | None = None,
    llm_config: str | Path = "configs/llm.yaml",
) -> ChatClient:
    """A client for one configured model, or ``llm_unconfigured`` when anything is missing."""

    config = load_llm_config(llm_config)
    model = config.models.get(model_key)
    if model is None:
        raise LlmError("llm_unconfigured")
    provider = config.providers[model.provider]
    key = _api_key(settings, provider)
    if key is None or settings.llm_daily_spend_cap_usd is None:
        raise LlmError("llm_unconfigured")
    return ChatClient(model, provider, key, ledger, cache=cache)
