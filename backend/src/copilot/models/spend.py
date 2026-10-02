"""Hosted LLM prices, the cost of each call, and the ledger that holds the daily cap.

The hosted LLM API is the one paid resource (spec §2), so every call is logged
with its cost and no call starts once the operator's daily cap would be passed
(spec §12). A call first *reserves* its worst-case cost — every prompt
character counted at three per token, no cache discount, the whole output
budget — and only then reaches the network. When it ends it *settles*: the
reservation is replaced by the cost the provider's usage implies, or, when the
usage is unknown, kept whole, since a timed-out call may still have been billed.

``configs/llm.yaml`` names the providers and prices the models. A model without
a price cannot be configured, so a priced call can never run unpriced. The
ledger stores model names, token counts, costs and timings — never a prompt, a
query or a response.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import UUID, uuid4

import yaml
from sqlalchemy import Engine, text

PURPOSES = ("search", "evaluation")
SETTLED = ("succeeded", "failed")
USAGE_STYLES = ("openai", "deepseek")
TOKEN_PARAMETERS = ("max_tokens", "max_completion_tokens")
# Body fields the client owns. A model's own request fields may not replace them.
RESERVED_FIELDS = frozenset({"model", "messages", "stream", *TOKEN_PARAMETERS})
# "LLMSPEND": one cap for every process that shares this database.
_SPEND_LOCK = 0x4C4C4D5350454E44

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens."""

    input: float
    cached_input: float
    output: float


@dataclass(frozen=True)
class Usage:
    """Tokens a call consumed, as the provider reported them.

    ``cached_input_tokens`` is part of ``input_tokens``. ``output_tokens``
    includes any reasoning tokens, which providers bill as output.
    """

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ProviderSpec:
    """An OpenAI-compatible endpoint and how its requests and usage are spelled."""

    name: str
    base_url: str
    api_key_env: str
    max_tokens_param: str
    usage: str
    request_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmModel:
    """One callable model: provider, wire name, price and output budget.

    ``request`` holds this model's own body fields, such as
    ``reasoning_effort``; they are sent over the provider's ``request_extra``.
    """

    key: str
    provider: str
    model: str
    pinned: bool
    price: ModelPrice
    max_output_tokens: int
    request: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmConfig:
    providers: dict[str, ProviderSpec]
    models: dict[str, LlmModel]


def cost_usd(price: ModelPrice, usage: Usage) -> float:
    """What a call cost: uncached input, cached input and output, each at its own rate."""

    uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
    return (
        uncached * price.input
        + usage.cached_input_tokens * price.cached_input
        + usage.output_tokens * price.output
    ) / 1_000_000


def estimate_usd(price: ModelPrice, prompt_chars: int, max_output_tokens: int) -> float:
    """The most a call can cost: 3 characters per token (English runs nearer 4), no cache
    discount, and the whole output budget. Reserved before the call, so the cap holds."""

    return (math.ceil(prompt_chars / 3) * price.input + max_output_tokens * price.output) / 1e6


def _invalid(where: str) -> ValueError:
    return ValueError(f"llm_config_invalid:{where}")


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _invalid(where)
    return {str(key): item for key, item in value.items()}


def _text(section: Mapping[str, Any], name: str, where: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{where}.{name}")
    return value.strip()


def _provider(name: str, raw: object) -> ProviderSpec:
    where = f"providers.{name}"
    section = _mapping(raw, where)
    base_url = _text(section, "base_url", where).rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise _invalid(f"{where}.base_url")
    token_parameter = _text(section, "max_tokens_param", where)
    if token_parameter not in TOKEN_PARAMETERS:
        raise _invalid(f"{where}.max_tokens_param")
    usage = _text(section, "usage", where)
    if usage not in USAGE_STYLES:
        raise _invalid(f"{where}.usage")
    extra = _mapping(section.get("request_extra"), f"{where}.request_extra")
    if RESERVED_FIELDS & set(extra):
        raise _invalid(f"{where}.request_extra")
    return ProviderSpec(
        name=name,
        base_url=base_url,
        api_key_env=_text(section, "api_key_env", where),
        max_tokens_param=token_parameter,
        usage=usage,
        request_extra=extra,
    )


def _price(raw: object, where: str) -> ModelPrice:
    section = _mapping(raw, where)
    values: dict[str, float] = {}
    for name in ("input", "cached_input", "output"):
        value = section.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _invalid(where)
        if not math.isfinite(value) or value < 0:
            raise _invalid(where)
        values[name] = float(value)
    return ModelPrice(**values)


def _model(key: str, raw: object, providers: Mapping[str, ProviderSpec]) -> LlmModel:
    where = f"models.{key}"
    section = _mapping(raw, where)
    provider = _text(section, "provider", where)
    if provider not in providers:
        raise _invalid(f"{where}.provider")
    pinned = section.get("pinned")
    if not isinstance(pinned, bool):
        raise _invalid(f"{where}.pinned")
    budget = section.get("max_output_tokens")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise _invalid(f"{where}.max_output_tokens")
    if "price" not in section:
        raise _invalid(f"{where}.price")
    request = _mapping(section.get("request"), f"{where}.request")
    if RESERVED_FIELDS & set(request):
        raise _invalid(f"{where}.request")
    return LlmModel(
        key=key,
        provider=provider,
        model=_text(section, "model", where),
        pinned=pinned,
        price=_price(section["price"], f"{where}.price"),
        max_output_tokens=budget,
        request=request,
    )


def load_llm_config(path: str | Path) -> LlmConfig:
    """Read ``configs/llm.yaml``, refusing anything a priced call could not rely on."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise _invalid("schema_version")
    providers = {
        str(name): _provider(str(name), section)
        for name, section in _mapping(raw.get("providers"), "providers").items()
    }
    models = {
        str(key): _model(str(key), section, providers)
        for key, section in _mapping(raw.get("models"), "models").items()
    }
    if not models:
        raise _invalid("models")
    return LlmConfig(providers=providers, models=models)


class SpendGate(Protocol):
    """Reserve before a call, settle after it. ``reserve`` returns None past the cap."""

    def reserve(
        self,
        *,
        model: LlmModel,
        estimated_usd: float,
        purpose: str,
        request_id: UUID | None,
        run_id: str | None,
    ) -> UUID | None: ...

    def settle(
        self,
        call_id: UUID,
        *,
        status: str,
        usage: Usage | None,
        served_model: str | None,
        latency_ms: int,
        error_code: str | None,
    ) -> float: ...


def _usd(value: float) -> Decimal:
    return Decimal(str(round(value, 6)))


def _checked_estimate(estimated_usd: float, purpose: str) -> Decimal:
    if purpose not in PURPOSES:
        raise ValueError(f"llm_purpose_invalid:{purpose}")
    if not math.isfinite(estimated_usd) or estimated_usd < 0:
        raise ValueError("llm_estimate_invalid")
    return _usd(estimated_usd)


def _settled_cost(model: LlmModel, estimate: Decimal, status: str, usage: Usage | None) -> Decimal:
    if status not in SETTLED:
        raise ValueError(f"llm_status_invalid:{status}")
    return estimate if usage is None else _usd(cost_usd(model.price, usage))


def _utc(clock: Clock) -> datetime:
    now = clock()
    if now.tzinfo is None:
        raise ValueError("llm_clock_must_be_aware")
    return now.astimezone(UTC)


def _day_start(moment: datetime) -> datetime:
    return datetime(moment.year, moment.month, moment.day, tzinfo=UTC)


def _cap(daily_cap_usd: Decimal | None) -> Decimal:
    if daily_cap_usd is None:
        raise ValueError("llm_spend_cap_unset")
    cap = Decimal(daily_cap_usd)
    if not cap.is_finite() or cap <= 0:
        raise ValueError("llm_spend_cap_unset")
    return cap


@dataclass
class _MemoryCall:
    day: date
    model: LlmModel
    estimate: Decimal
    run_id: str | None
    cost: Decimal | None = None


class MemoryLedger:
    """The ledger's rules in process memory, for unit tests and the offline smoke set."""

    def __init__(self, daily_cap_usd: Decimal | None, clock: Clock | None = None) -> None:
        self._cap = _cap(daily_cap_usd)
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        self._calls: dict[UUID, _MemoryCall] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        *,
        model: LlmModel,
        estimated_usd: float,
        purpose: str,
        request_id: UUID | None,
        run_id: str | None,
    ) -> UUID | None:
        estimate = _checked_estimate(estimated_usd, purpose)
        day = _utc(self._clock).date()
        with self._lock:
            spent = sum(
                (call.estimate if call.cost is None else call.cost)
                for call in self._calls.values()
                if call.day == day
            )
            if spent + estimate > self._cap:
                return None
            call_id = uuid4()
            self._calls[call_id] = _MemoryCall(day, model, estimate, run_id)
        return call_id

    def settle(
        self,
        call_id: UUID,
        *,
        status: str,
        usage: Usage | None,
        served_model: str | None,
        latency_ms: int,
        error_code: str | None,
    ) -> float:
        with self._lock:
            call = self._calls.get(call_id)
            if call is None or call.cost is not None:
                raise ValueError("llm_call_not_reserved")
            call.cost = _settled_cost(call.model, call.estimate, status, usage)
            return float(call.cost)

    def run_spend(self, run_id: str) -> float:
        with self._lock:
            return float(
                sum(
                    (call.estimate if call.cost is None else call.cost)
                    for call in self._calls.values()
                    if call.run_id == run_id
                )
            )


class SpendLedger:
    """The PostgreSQL ledger. Every API worker and evaluation run shares one cap.

    A reservation is one transaction under a transaction-scoped advisory lock,
    so concurrent callers serialize on the cap. A ``reserved`` row counts at its
    estimate until it is settled; a crashed process's reservation stays charged
    for that day, which errs on the safe side.
    """

    def __init__(
        self,
        engine: Engine | None,
        daily_cap_usd: Decimal | None,
        clock: Clock | None = None,
    ) -> None:
        self._cap = _cap(daily_cap_usd)
        if engine is None:
            raise ValueError("llm_ledger_requires_an_engine")
        self._engine = engine
        self._clock: Clock = clock or (lambda: datetime.now(UTC))
        # Settling needs the reserved model's price; calls settle in the process
        # that reserved them.
        self._pending: dict[UUID, tuple[LlmModel, Decimal]] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        *,
        model: LlmModel,
        estimated_usd: float,
        purpose: str,
        request_id: UUID | None,
        run_id: str | None,
    ) -> UUID | None:
        estimate = _checked_estimate(estimated_usd, purpose)
        now = _utc(self._clock)
        start = _day_start(now)
        call_id = uuid4()
        with self._engine.begin() as connection:
            connection.execute(text("select pg_advisory_xact_lock(:key)"), {"key": _SPEND_LOCK})
            spent = connection.execute(
                text(
                    "select coalesce(sum(coalesce(cost_usd, estimated_usd)), 0) from llm_calls"
                    " where created_at >= :start and created_at < :end"
                ),
                {"start": start, "end": start + timedelta(days=1)},
            ).scalar_one()
            if Decimal(spent) + estimate > self._cap:
                return None
            connection.execute(
                text(
                    "insert into llm_calls (id, request_id, run_id, purpose, model_key, provider,"
                    " model, status, estimated_usd, created_at) values (:id, :request, :run,"
                    " :purpose, :key, :provider, :model, 'reserved', :estimate, :now)"
                ),
                {
                    "id": call_id,
                    "request": request_id,
                    "run": run_id,
                    "purpose": purpose,
                    "key": model.key,
                    "provider": model.provider,
                    "model": model.model,
                    "estimate": estimate,
                    "now": now,
                },
            )
        with self._lock:
            self._pending[call_id] = (model, estimate)
        return call_id

    def settle(
        self,
        call_id: UUID,
        *,
        status: str,
        usage: Usage | None,
        served_model: str | None,
        latency_ms: int,
        error_code: str | None,
    ) -> float:
        with self._lock:
            pending = self._pending.pop(call_id, None)
        if pending is None:
            raise ValueError("llm_call_not_reserved")
        model, estimate = pending
        cost = _settled_cost(model, estimate, status, usage)
        with self._engine.begin() as connection:
            updated = connection.execute(
                text(
                    "update llm_calls set status = :status, cost_usd = :cost,"
                    " input_tokens = :input, cached_input_tokens = :cached,"
                    " output_tokens = :output, served_model = :served, latency_ms = :latency,"
                    " error_code = :error, settled_at = :now"
                    " where id = :id and status = 'reserved'"
                ),
                {
                    "id": call_id,
                    "status": status,
                    "cost": cost,
                    "input": None if usage is None else usage.input_tokens,
                    "cached": None if usage is None else usage.cached_input_tokens,
                    "output": None if usage is None else usage.output_tokens,
                    "served": None if served_model is None else served_model[:128],
                    "latency": latency_ms,
                    "error": None if error_code is None else error_code[:64],
                    "now": _utc(self._clock),
                },
            ).rowcount
        if updated != 1:
            raise ValueError("llm_call_not_reserved")
        return float(cost)

    def run_spend(self, run_id: str) -> float:
        with self._engine.connect() as connection:
            spent = connection.execute(
                text(
                    "select coalesce(sum(coalesce(cost_usd, estimated_usd)), 0)"
                    " from llm_calls where run_id = :run"
                ),
                {"run": run_id},
            ).scalar_one()
        return float(spent)


def spend_report(
    engine: Engine, *, day: date | None = None, run_id: str | None = None
) -> dict[str, Any]:
    """Calls, tokens and cost by model for one UTC day or one evaluation run.

    Unsettled reservations count at their estimate, as the cap counts them.
    """

    params: dict[str, Any]
    if run_id is not None:
        where, params = "run_id = :run", {"run": run_id}
        scope: dict[str, Any] = {"run_id": run_id}
    else:
        start = _day_start(datetime.combine(day or datetime.now(UTC).date(), datetime.min.time()))
        where = "created_at >= :start and created_at < :end"
        params = {"start": start, "end": start + timedelta(days=1)}
        scope = {"day": start.date().isoformat()}
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "select model_key, count(*) as calls,"
                " count(*) filter (where status = 'failed') as failed,"
                " count(*) filter (where status = 'reserved') as unsettled,"
                " coalesce(sum(input_tokens), 0) as input_tokens,"
                " coalesce(sum(cached_input_tokens), 0) as cached_input_tokens,"
                " coalesce(sum(output_tokens), 0) as output_tokens,"
                " coalesce(sum(coalesce(cost_usd, estimated_usd)), 0) as cost_usd"
                f" from llm_calls where {where} group by model_key order by model_key"
            ),
            params,
        ).mappings()
        models = {
            str(row["model_key"]): {
                "calls": int(row["calls"]),
                "failed": int(row["failed"]),
                "unsettled": int(row["unsettled"]),
                "input_tokens": int(row["input_tokens"]),
                "cached_input_tokens": int(row["cached_input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "cost_usd": float(row["cost_usd"]),
            }
            for row in rows
        }
    return scope | {
        "models": models,
        "total_usd": round(sum(entry["cost_usd"] for entry in models.values()), 6),
    }
