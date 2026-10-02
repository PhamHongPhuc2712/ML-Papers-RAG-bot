# LLM Listwise Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in search mode in which a hosted LLM — OpenAI or DeepSeek — reorders
the cross-encoder's reranked head listwise, as LitSearch's best published system does. It is
guarded by per-call cost logging and a daily spend cap, and promoted only by a pre-registered
rule on E3 validation.

**Architecture:** A thin `httpx` client speaks the OpenAI-compatible Chat Completions API to
both providers. Every call goes through a PostgreSQL spend ledger: reserve the worst-case cost
atomically under a daily cap, call, then settle the actual cost. A listwise reranker builds one
prompt over the head, and parses the answer into a strict permutation; it can reorder, never
add or drop a candidate. The search service runs it as one more stage with its own deadline and
lane. Any failure falls back to the cross-encoder order with a typed warning, the same pattern as
every existing stage.

**Tech Stack:** Python 3.12, `httpx` (already pinned, 0.28.1), PostgreSQL + Alembic, FastAPI,
pytest with `httpx.MockTransport`. No new runtime dependency.

**Spec:** [Technical specification](../specs/2026-09-05-ml-research-copilot-design.md):
- §1: search must work without an LLM;
- §6: contracts;
- §7: retrieval and ranking;
- §10: HTTP API;
- §11: evaluation;
- §12: provider spend cap, cost logging and latency budgets.

Also read the [retrieval plan](2026-09-05-02-retrieval-ranking.md) P2.6, and P5.2 in the
[production beta plan](2026-09-05-05-production-beta.md).

**Status:** Draft for review, 2026-10-01. No task started. Requested by the developer, who funds
OpenAI and DeepSeek API usage.

## Global Constraints

- Python 3.12; UTC timestamps; locked dependencies; pinned model revisions before benchmark runs.
- The hosted LLM API is the one paid resource, funded personally by the developer. Per-request
  cost logging and the operator-configured daily spend cap are required, not optional (P5.2).
- Priced model calls are disabled when cost configuration is missing (spec §12).
- Search degrades explicitly: a failed stage leaves the previous stage's order, with a warning;
  never an empty page that looks like "no results" (spec §7).
- Raw private queries are excluded from default traces; stored artifacts keep hashes, not query
  text (spec §12; `search/cache.py`).
- LitSearch query text has no license: it stays under `DATA_DIR`, never in git or a report.
- All persistent local data lives under `DATA_DIR`.
- Never pass `.env` to pytest; integration tests use `.env.test` and the `test` services.
- Tune on development, choose on validation; the test split runs once, with `--locked-test`,
  with the developer's go-ahead.
- `mypy` always runs with `--config-file backend/pyproject.toml`.

## Measured facts this plan is built on (checked 2026-10-01)

| Fact | Value | Source |
|---|---|---|
| LitSearch's best system | GPT-4o reranking over GritLM-7B: 79.2 average specific R@5, 75.3 average broad R@20. GritLM-7B alone: 74.8 / 70.8 | arXiv 2407.18940v2, Table 3 |
| LitSearch's reranking setup | RankGPT-style single listwise prompt; top **100** candidates; first **300 words** of title+abstract each; output `[i] > [j]`; unranked candidates appended in original order. The code's default model is `gpt-4-1106-preview`; the paper reports GPT-4o | `princeton-nlp/LitSearch` `eval/reranking/rerank.py` |
| DeepSeek API | OpenAI-compatible; base URL `https://api.deepseek.com`; models `deepseek-flash` (served by DeepSeek-V4.1-Flash) and `deepseek-v4-pro`. **Aliases only, no dated snapshots** | api-docs.deepseek.com |
| DeepSeek prices, per 1M tokens, peak | flash: input $0.30, cache-hit input $0.006, output $1.20. v4-pro: $1.32 / $0.044 / $3.96. Off-peak is half; peak hours are 01:00–04:00 and 06:00–10:00 UTC, Monday–Friday | api-docs.deepseek.com/quick_start/pricing |
| OpenAI prices, per 1M tokens, read 2026-10-02 | `gpt-6-luna`: input $0.10, cached input $0.01, output $0.50. `gpt-5.6-luna`: $0.20 / $0.02 / $1.20. `gpt-6-sol`: $2.00 / $0.20 / $10.00. `gpt-4o`: $2.50 / $1.25 / $10.00. Batch API 50% off | developers.openai.com/api/docs/pricing |
| `gpt-6-luna` | A reasoning model. `reasoning_effort` takes `none`, `low`, `medium` (the default), `high`, `xhigh` or `max`; reasoning tokens are billed as output and count against the output cap. Chat Completions and structured outputs supported; 1,050,000-token context, 128,000 max output. **Alias only, no dated snapshots** | developers.openai.com/api/docs/models/gpt-6-luna |

Prices are list prices read on 2026-10-01. Every figure in this plan derived from them is an
**estimate**, replaced by the ledger's recorded cost after the first real call.

## Cost estimates

The assumptions:
- the LLM sees the whole reranked head, 100 candidates if P2.6 adopts depth 100 (the paper's
  setting), otherwise 50;
- 100 candidates of at most 300 words come to about 35k input tokens;
- the answer is about 600 output tokens.

Halve every figure for a 50-candidate head.

| Model key | Per search | Pilot (20 queries) | Dev + val + in-domain (681 queries) |
|---|---|---|---|
| `openai-gpt-6-luna` (effort `none`) | ≈ $0.004 | ≈ $0.08 | ≈ $2.6 |
| `openai-gpt-6-luna-low` (effort `low`, at most 4,096 output tokens) | ≤ $0.006 | ≤ $0.11 | ≤ $3.8 |
| `deepseek-flash` (peak rate; fallback only) | ≈ $0.011 | ≈ $0.22 | ≈ $7.6 |

**Model decided 2026-10-02: `gpt-6-luna`.** GPT-4o (≈ $64 for the 681 queries) and
`deepseek-v4-pro` are dropped for cost. LitSearch's GPT-4o rows are therefore compared
indirectly, beside Table 3 through `eval gaps`, not reproduced. The whole evaluation comes
to about $6: a $0.41 pilot, both luna variants on development and validation, and the
chosen one in-domain. The worst case, `deepseek-flash` replacing a failed luna variant, is
under $11.

Reruns of an evaluation are free: responses are cached by request digest (L2).

## Decisions (settled 2026-10-02, except 4)

1. **Opt-in or default: opt-in.** A new mode, `hybrid_rerank_llm`. The default stays
   `hybrid_rerank` unless validation and the latency budget justify a change and you decide it.
   This keeps spec §1 true: search still works without an LLM.
2. **Daily spend cap** (`LLM_DAILY_SPEND_CAP_USD`): **$3 a day**, lowered from the
   recommended $10. Without a value, LLM calls are disabled. The pilot and each luna variant's
   validation and in-domain runs fit inside one day. A development run may not: both luna
   variants over 359 queries can reach $3.30. If the cap stops it, L5 reruns it the next UTC
   day, and cached answers cost nothing.
3. **Deep-search latency budget: p95 ≤ 20 s** for `hybrid_rerank_llm`. The 3 s budget stays
   for every other mode.
4. **Data sent to providers. Open until L5.** Queries and paper titles and abstracts leave
   this machine, and during evaluation that includes LitSearch's query text. Record each
   provider's API data-use and retention terms in the decision row before L5. DeepSeek's terms
   and data location differ from OpenAI's.
5. **Before or after the locked test: before.** The locked test scores once per release, so it
   must score the configuration that ships.

Each becomes a row in `docs/superpowers/progress.md` → "Decisions recorded during implementation".

## What this plan changes elsewhere

| Document | Change | Where |
|---|---|---|
| Spec §6 | `SearchRequest.mode` gains `hybrid_rerank_llm`; new `ListwiseResult` contract | L4 |
| Spec §7 | New stage after the cross-encoder: listwise LLM over the reranked head; fallback to its order | L4 |
| Spec §10 | `POST /v1/search` accepts the new mode | L4 |
| Spec §12 | Providers named (OpenAI, DeepSeek); deep-search latency budget; spend ledger live before P5.2 | L1, L4 |
| P5.2 | Cost logging and the daily cap are pulled forward, with a demonstrated need. P5.2 keeps tracing, rate limits and quotas, and builds on `models/spend.py` | L1 |
| Benchmarks plan E6/E7 | The judge was planned as `claude-opus-5` via the `anthropic` SDK, but the funded providers are now OpenAI and DeepSeek. Revise E6's model and SDK before it starts. Not part of this plan | — |

## Sequencing with P2.6 and GritLM

1. Finish P2.6's ceiling and depth-100 runs. They fix the head size the LLM sees.
2. P2.6 step 4 includes GritLM-7B as a first-stage ablation. It is independent of this plan and
   can run while L1–L4 are built, since L1–L4 need no GPU.
3. L1–L4: code and tests. The only spend is L2's two live calls, under $0.05.
4. L5: pilot, then development, commit the rule, then validation, then in-domain.
5. The locked test, once, on the final configuration, with your go-ahead.

## File structure

```text
backend/migrations/versions/0005_llm_calls.py     ledger table (L1)
backend/src/copilot/models/spend.py               prices, cost, ledger: reserve/settle under a daily cap (L1)
backend/src/copilot/models/llm.py                 provider config, OpenAI-compatible chat client, response cache (L2)
backend/src/copilot/search/llm_rerank.py          listwise prompt, permutation parsing, the reranker (L3)
backend/src/copilot/search/service.py             new mode's stage, lane, deadline, fallbacks (L4, modify)
backend/src/copilot/search/api.py                 builds the LLM reranker when configured (L4, modify)
backend/src/copilot/contracts.py                  mode literal, ListwiseResult (L3/L4, modify)
backend/src/copilot/config.py                     keys, cap, chosen model (L1/L4, modify)
backend/src/copilot/evaluation/retrieval.py       `llm` variant field, spend preflight, cost in manifest (L5, modify)
backend/src/copilot/cli.py                        --max-spend-usd; `llm spend` report (L1/L5, modify)
configs/llm.yaml                                  providers, models, prices with verification date (L1)
configs/search.yaml                               llm deadlines (L4, modify)
prompts/rerank/listwise-v1.yaml                   the pinned prompt (L3)
configs/experiments/e3-llm-pilot.yaml             20-query pilot over the candidate models (L5)
configs/experiments/e3-llm-rerank.yaml            pre-registered rule (L5)
reports/m2-llm-rerank.md                          evidence (L5)
```

## Review Focus

1. **The model answers with prose, duplicate ids, ids out of range, a partial list, or bare
   numbers.** The order is still a permutation of exactly the input candidates, with unranked
   ones appended in incoming order. If nothing parses, search falls back to the cross-encoder
   order with `llm_rerank_unparseable`. *Tested in L3.*
2. **Concurrent searches arrive just below the cap.** The cap is never exceeded. A refused call
   makes **no** HTTP request, and search serves the cross-encoder order with `llm_spend_cap`.
   *Tested in L1 and L4.*
3. **The provider returns 429 or 5xx, hangs, or rejects the key.** Retries stay inside the
   deadline and honour `Retry-After`. Then search falls back; never a 500. A bad key is
   `llm_auth_failed`, not retried. *Tested in L2 and L4.*
4. **An API key or private text leaks.** No key appears in an exception message, log record,
   manifest or trace. No query or abstract text is stored in the ledger or in cached responses.
   *Tested in L1 and L2.*
5. **A model goes away or changes underneath.** A retired or unknown model is a typed
   `llm_model_unavailable`, never silently substituted. `gpt-6-luna` and the DeepSeek models
   are aliases with no dated snapshots, so every call records the model that actually served
   it. An evaluation
   whose served model changed between its pilot and its validation run says so. *Tested in L2
   and L5.*
6. **A reasoning model spends its output cap thinking.** `gpt-6-luna` defaults to `medium`
   effort, and its reasoning tokens count against `max_completion_tokens` and are billed as
   output. Every luna model sets `reasoning_effort` explicitly. An answer cut off at the cap
   (`finish_reason: length`) is `llm_response_invalid`, never parsed as a partial ranking.
   *Tested in L2.*

---

### Task L1: Spend ledger, prices and the daily cap

**Files:**
- Create: `backend/migrations/versions/0005_llm_calls.py`, `backend/src/copilot/models/spend.py`,
  `configs/llm.yaml`
- Modify: `backend/src/copilot/config.py`, `backend/src/copilot/db/models.py` (an `LlmCall`
  model, for the migration-parity test), `backend/src/copilot/cli.py` (`llm spend`), `.env.example`
- Test: `backend/tests/unit/test_llm_spend.py`, `backend/tests/integration/test_llm_ledger.py`

**Interfaces:**
- Produces:
  - `ModelPrice(input: float, cached_input: float, output: float)`, in USD per 1M tokens;
  - `Usage(input_tokens: int, cached_input_tokens: int, output_tokens: int)`;
  - `cost_usd(price, usage) -> float`;
  - `estimate_usd(price, prompt_chars: int, max_output_tokens: int) -> float`;
  - `LlmModel(key: str, provider: str, model: str, pinned: bool, price: ModelPrice, max_output_tokens: int, request: Mapping[str, Any])`, where `request` holds that model's
    own body fields, such as `reasoning_effort`, merged over the provider's `request_extra`;
    it defaults to `{}`;
  - `ProviderSpec(name: str, base_url: str, api_key_env: str, max_tokens_param: str, usage: str, request_extra: Mapping[str, Any])`,
    where `usage` is `"openai"` or `"deepseek"` and `request_extra` defaults to `{}`;
  - `load_llm_config(path) -> LlmConfig`, with `.providers: dict[str, ProviderSpec]` and `.models: dict[str, LlmModel]`;
  - `SpendGate` (Protocol): `reserve(*, model: LlmModel, estimated_usd: float, purpose: str, request_id: UUID | None, run_id: str | None) -> UUID | None` and `settle(call_id: UUID, *, status: str, usage: Usage | None, served_model: str | None, latency_ms: int, error_code: str | None) -> float`;
  - `SpendLedger(engine, daily_cap_usd: Decimal, clock)`, the PostgreSQL implementation, plus
    `run_spend(run_id: str) -> float`;
  - `MemoryLedger(daily_cap_usd)`, an in-process implementation for unit tests and the smoke
    set, with the same `run_spend`.

- [ ] **Step 1: Write the failing unit tests** in `backend/tests/unit/test_llm_spend.py`:

```python
from decimal import Decimal

import pytest

from copilot.models.spend import (
    MemoryLedger,
    ModelPrice,
    Usage,
    cost_usd,
    estimate_usd,
    load_llm_config,
)

PRICE = ModelPrice(input=2.50, cached_input=1.25, output=10.00)


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


def test_the_repository_config_prices_every_model():
    config = load_llm_config("configs/llm.yaml")
    assert {"openai-gpt-6-luna", "openai-gpt-6-luna-low", "deepseek-flash"} <= set(config.models)
    assert config.models["openai-gpt-6-luna"].request["reasoning_effort"] == "none"
    assert config.models["openai-gpt-6-luna-low"].request["reasoning_effort"] == "low"
    assert config.models["deepseek-flash"].pinned is False


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
```

- [ ] **Step 2: Run them and confirm they fail on the missing module**

Run: `uv run --project backend pytest backend/tests/unit/test_llm_spend.py -q`
Expected: `ModuleNotFoundError: No module named 'copilot.models.spend'`.

- [ ] **Step 3: Write `configs/llm.yaml`.** Use the prices in the facts table, with the reading
  date:

```yaml
# Hosted LLM providers and the models this project may call (spec §12). Prices are USD per 1M
# tokens, read from the providers' pricing pages: DeepSeek 2026-10-01, OpenAI 2026-10-02.
# A model without a price cannot be called: priced calls are disabled when cost
# configuration is missing.
# Keys are read from the environment variables named here, never from this file.
schema_version: 1
providers:
  openai:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    # Newer OpenAI models reject max_tokens on Chat Completions.
    max_tokens_param: max_completion_tokens
    usage: openai
  deepseek:
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
    max_tokens_param: max_tokens
    usage: deepseek
models:
  openai-gpt-6-luna:
    # The chosen model (2026-10-02). A reasoning model whose default effort is medium;
    # `none` keeps hidden reasoning tokens, billed as output, out of every call. An alias
    # with no dated snapshot: the served model is recorded on every call. L2 step 3 adds
    # `temperature: 0` and `seed: 42` here if the live call shows the model accepts them.
    provider: openai
    model: gpt-6-luna
    pinned: false
    max_output_tokens: 1024
    request: {reasoning_effort: none}
    price: {input: 0.10, cached_input: 0.01, output: 0.50}
  openai-gpt-6-luna-low:
    # The same model with a little reasoning, piloted against `none`. Reasoning tokens
    # count against the output cap, hence the larger one.
    provider: openai
    model: gpt-6-luna
    pinned: false
    max_output_tokens: 4096
    request: {reasoning_effort: low}
    price: {input: 0.10, cached_input: 0.01, output: 0.50}
  deepseek-flash:
    # Fallback: used only in place of a luna variant that fails the pilot, and only when
    # DEEPSEEK_API_KEY is set. DeepSeek publishes aliases only. Peak rates: an upper bound
    # off-peak.
    provider: deepseek
    model: deepseek-flash
    pinned: false
    max_output_tokens: 1024
    price: {input: 0.30, cached_input: 0.006, output: 1.20}
```

- [ ] **Step 4: Implement the pure part of `backend/src/copilot/models/spend.py`**: the dataclasses,
  `cost_usd`, `estimate_usd`, `load_llm_config` (refusing any model without
  `price.input/cached_input/output`, an unknown provider, or `max_output_tokens < 1`) and
  `MemoryLedger`.

```python
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
```

- [ ] **Step 5: Run the unit tests; expect PASS.**

- [ ] **Step 6: Write the failing integration tests** in
  `backend/tests/integration/test_llm_ledger.py`, against the `migrated_database` fixture:

```python
import pytest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

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


def test_the_ledger_stores_no_prompt_or_query_text(migrated_database):
    with migrated_database.connect() as connection:
        columns = set(connection.execute(text(
            "select column_name from information_schema.columns where table_name = 'llm_calls'"
        )).scalars())
    assert not columns & {"prompt", "query", "messages", "response", "text"}


def test_a_new_utc_day_starts_with_the_full_cap(migrated_database, ledger):
    assert ledger.reserve(model=MODEL, estimated_usd=1.00, purpose="search",
                          request_id=None, run_id=None)
    tomorrow = SpendLedger(migrated_database, daily_cap_usd=Decimal("1.00"),
                           clock=lambda: NOON.replace(day=3))
    assert tomorrow.reserve(model=MODEL, estimated_usd=1.00, purpose="search",
                            request_id=None, run_id=None)
```

- [ ] **Step 7: Run them and confirm they fail** (no `llm_calls` table). Use
  `uv run --env-file .env.test --project backend pytest backend/tests/integration/test_llm_ledger.py -q`.

- [ ] **Step 8: Write migration `0005_llm_calls`** with `down_revision = "0004_search_orderings"`.
  Table `llm_calls`:
  - `id` uuid PK; `request_id` uuid null; `run_id` varchar(255) null;
  - `purpose` varchar(16), check in `('search','evaluation')`;
  - `model_key` varchar(64); `provider` varchar(32); `model` varchar(128); `served_model` varchar(128) null;
  - `status` varchar(16), check in `('reserved','succeeded','failed')`;
  - `estimated_usd` numeric(12,6) not null; `cost_usd` numeric(12,6) null;
  - `input_tokens`, `cached_input_tokens`, `output_tokens`, `latency_ms`: int null;
  - `error_code` varchar(64) null; `created_at` timestamptz not null; `settled_at` timestamptz null.

  Indexes on `(created_at)` and `(run_id)`. **No text columns for prompts, queries or responses.**
  Mirror it as `LlmCall` in `db/models.py`.

- [ ] **Step 9: Implement `SpendLedger`.** Reservation is one transaction under a
  transaction-scoped advisory lock, so concurrent API workers serialize on the cap. A
  `reserved` row counts at its estimate until it is settled. A crashed process's reservation
  stays charged for that day, which errs on the safe side.

```python
_SPEND_LOCK = 0x4C4C4D5350454E44  # "LLMSPEND"; one cap for every process on this database


def reserve(self, *, model, estimated_usd, purpose, request_id, run_id) -> UUID | None:
    now = self._clock()
    start = datetime(now.year, now.month, now.day, tzinfo=UTC)
    estimate = Decimal(str(round(estimated_usd, 6)))
    with self._engine.begin() as connection:
        connection.execute(text("select pg_advisory_xact_lock(:key)"), {"key": _SPEND_LOCK})
        spent = connection.execute(
            text("select coalesce(sum(coalesce(cost_usd, estimated_usd)), 0) from llm_calls"
                 " where created_at >= :start and created_at < :end"),
            {"start": start, "end": start + timedelta(days=1)},
        ).scalar_one()
        if spent + estimate > self._cap:
            return None
        call_id = uuid4()
        connection.execute(
            text("insert into llm_calls (id, request_id, run_id, purpose, model_key, provider,"
                 " model, status, estimated_usd, created_at) values (:id, :request, :run,"
                 " :purpose, :key, :provider, :model, 'reserved', :estimate, :now)"),
            {"id": call_id, "request": request_id, "run": run_id, "purpose": purpose,
             "key": model.key, "provider": model.provider, "model": model.model,
             "estimate": estimate, "now": now},
        )
    return call_id
```

  `settle` writes the status, tokens, served model, latency and error code. It sets `cost_usd`
  to `cost_usd(price, usage)` when usage is known, and otherwise to the estimate.
  `run_spend(run_id) -> float` sums `coalesce(cost_usd, estimated_usd)` for one run.

- [ ] **Step 10: Settings and CLI.** In `config.py`:
  - `openai_api_key: SecretStr | None` (alias `OPENAI_API_KEY`);
  - `deepseek_api_key: SecretStr | None` (alias `DEEPSEEK_API_KEY`);
  - `llm_daily_spend_cap_usd: Decimal | None` (alias `LLM_DAILY_SPEND_CAP_USD`, `gt=0`).

  Add all three to `.env.example`, empty. Add the CLI command
  `llm spend [--day YYYY-MM-DD] [--run RUN_ID]`, printing calls, tokens and cost by model.

- [ ] **Step 11: Run the suites.**
  - L1's tests: `test_llm_spend.py` and `test_llm_ledger.py`;
  - `test_migrations.py`, since migrations must replay up, down and up;
  - the full suite;
  - Ruff;
  - `mypy --config-file backend/pyproject.toml backend/src`.

  Migrate `copilot_v2` with `alembic upgrade head`.

- [ ] **Step 12: Record the decision row.** It says the spend ledger and cap were pulled
  forward from P5.2, why, and what P5.2 still owns. Then commit:
  `git commit -m "feat: log LLM spend per call and enforce a daily cap"`

**Acceptance cases:**
- concurrent reservations stop exactly at the cap;
- cached input is charged at its own rate;
- a failed call keeps its estimate;
- a new UTC day resets the cap;
- a model with no price is refused at load;
- there is no text column for prompts or queries;
- with no cap configured, `SpendLedger` cannot be built (`ValueError("llm_spend_cap_unset")`).

---

### Task L2: OpenAI-compatible chat client for OpenAI and DeepSeek

**Files:**
- Create: `backend/src/copilot/models/llm.py`
- Test: `backend/tests/unit/test_llm_client.py`, plus recorded responses in
  `backend/tests/fixtures/llm/{openai_chat.json,deepseek_chat.json}`

**Interfaces:**
- Consumes: `LlmModel`, `ProviderSpec`, `SpendGate`, `Usage`, `estimate_usd` (L1).
- Produces:
  - `ChatResult(text: str, usage: Usage, served_model: str, cost_usd: float, cached: bool, latency_ms: int)`;
  - `LlmError(code: str, retryable: bool)`, with codes `llm_spend_cap`, `llm_timeout`,
    `llm_rate_limited`, `llm_server_error`, `llm_auth_failed`, `llm_model_unavailable`,
    `llm_bad_request`, `llm_response_invalid`;
  - `ChatClient(model, provider, api_key: SecretStr, ledger: SpendGate, *, transport: httpx.BaseTransport | None = None, cache: ResponseCache | None = None, sleep: Callable[[float], None] = time.sleep)`;
    `sleep` is injected so the retry tests never wait;
  - `ChatClient.complete(messages: list[dict[str, str]], *, timeout: float, purpose: str, request_id: UUID | None = None, run_id: str | None = None) -> ChatResult`;
  - `ResponseCache(directory: Path)`, keyed by the sha256 of the whole request body: model,
    messages, token cap and every `request` field. Two variants of one model, such as the two
    `reasoning_effort` settings, therefore never share an entry. It stores the response text,
    usage and served model, never the messages.
  - `build_client(model_key: str, settings: Settings, ledger: SpendGate, *, cache: ResponseCache | None = None, llm_config: str | Path = "configs/llm.yaml") -> ChatClient`.
    It raises `LlmError("llm_unconfigured")` when the key, the cap or the price is missing.

- [ ] **Step 1: Write the failing tests** with `httpx.MockTransport`; no network is touched:

```python
import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from copilot.models.llm import ChatClient, LlmError, ResponseCache
from copilot.models.spend import MemoryLedger, load_llm_config

CONFIG = load_llm_config("configs/llm.yaml")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"
KEY = SecretStr("sk-test-must-never-appear")
MESSAGES = [{"role": "user", "content": "rank these"}]


def _client(model_key, handler, *, cap="1.00", cache=None):
    model = CONFIG.models[model_key]
    return ChatClient(model, CONFIG.providers[model.provider], KEY,
                      MemoryLedger(daily_cap_usd=Decimal(cap)),
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


def test_a_refused_reservation_makes_no_request():
    calls = []
    client = _client("openai-gpt-6-luna", lambda r: calls.append(r), cap="0.000001")
    with pytest.raises(LlmError, match="llm_spend_cap"):
        client.complete(MESSAGES, timeout=5.0, purpose="search")
    assert calls == []


def test_rate_limits_retry_within_the_deadline_then_succeed():
    responses = iter([httpx.Response(429, headers={"retry-after": "0"}),
                      httpx.Response(503),
                      httpx.Response(200, json=json.loads((FIXTURES / "openai_chat.json").read_text()))])
    result = _client("openai-gpt-6-luna", lambda r: next(responses)).complete(
        MESSAGES, timeout=5.0, purpose="search")
    assert result.text


def test_a_rejected_key_is_typed_not_retried_and_never_echoed(caplog):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "Incorrect API key sk-test-must-never-appear"}})

    with pytest.raises(LlmError) as raised:
        _client("deepseek-flash", handler).complete(MESSAGES, timeout=5.0, purpose="search")
    assert raised.value.code == "llm_auth_failed" and len(calls) == 1
    assert "sk-test" not in str(raised.value) and "sk-test" not in caplog.text


def test_a_retired_model_is_typed_not_substituted():
    handler = lambda r: httpx.Response(404, json={"error": {"code": "model_not_found"}})
    with pytest.raises(LlmError, match="llm_model_unavailable"):
        _client("openai-gpt-6-luna", handler).complete(MESSAGES, timeout=5.0, purpose="search")


def test_an_answer_cut_off_at_the_output_cap_is_invalid_not_partial():
    body = json.loads((FIXTURES / "openai_chat.json").read_text())
    body["choices"][0]["finish_reason"] = "length"
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
```

- [ ] **Step 2: Run them and confirm they fail** on the missing module:
  `uv run --project backend pytest backend/tests/unit/test_llm_client.py -q`

- [ ] **Step 3: Capture the fixtures from real calls.** These are the plan's only spend
  before L5, under $0.05 in total. Call `gpt-6-luna` with the message "Reply with the word
  ok.", `reasoning_effort: none` and the `.env` key; call DeepSeek the same way only if
  `DEEPSEEK_API_KEY` is set. Without a DeepSeek key, write `deepseek_chat.json` by hand from
  the response example in DeepSeek's API reference, and say so in the fixture README. Save
  each JSON body with the `id` field replaced by `"fixture"`. Send one more luna call each
  with `temperature: 0` and with `seed: 42`; add to `openai-gpt-6-luna`'s and
  `openai-gpt-6-luna-low`'s `request` whichever the API accepts, and note any it rejects in
  `configs/llm.yaml`. Then:
  - check DeepSeek's usage field names against the body and edit `cached_input_tokens` above
    to match the captured values;
  - set the fixture's cached counts to 1024 and 512 by hand so the tests above pin the parsing;
  - note in the fixture README which fields were edited.

  Read DeepSeek's thinking-mode page and set whatever request field selects **non-thinking**
  mode for `deepseek-flash`. Record its value as `request_extra` under
  the provider in `configs/llm.yaml`.

- [ ] **Step 4: Implement `ChatClient.complete`**:
  1. Look up the cache. On a hit, return `cached=True` with cost 0 and no reservation.
  2. Reserve `estimate_usd(price, len(json.dumps(messages)), model.max_output_tokens)`. A
     refusal raises `llm_spend_cap` before any request is built.
  3. POST `{base_url}/chat/completions` with `Authorization: Bearer <key>`,
     `{"model", "messages", <max_tokens_param>: max_output_tokens, **provider.request_extra,
     **model.request}`. `temperature` and `seed` are sent only where a model's `request`
     carries them (step 3).
  4. Retry `429`, `5xx` and transport timeouts while time remains: at most 3 attempts, sleeping
     `min(Retry-After or 2**attempt, remaining)`. `401`/`403` are `llm_auth_failed`. A `404` or
     an error code `model_not_found` is `llm_model_unavailable`. Any other `4xx` is
     `llm_bad_request`.
  5. Parse `choices[0].message.content`, `usage` (per provider style) and `model`. A missing
     field is `llm_response_invalid`, and so is `finish_reason == "length"`: an answer cut
     off at the cap is never parsed as a partial ranking. OpenAI's `completion_tokens`
     already includes reasoning tokens, so the cost needs no separate field.
  6. Settle the reservation on every path, success or failure, in a `finally`. Store the
     response in the cache on success.

  Error messages carry only the code and HTTP status. Never include a response body (it can
  echo the key), the headers, or the messages.

- [ ] **Step 5: Run the tests; expect PASS. Then run Ruff and mypy, and commit:**
  `git commit -m "feat: add OpenAI and DeepSeek chat clients behind the spend ledger"`

**Acceptance cases:** both usage styles parse; the pinned model and the right token parameter
are sent; a cap refusal makes zero requests; 429/503 retry inside the deadline; 401 is not
retried and the key is never echoed; a retired model is typed; a cache hit is free and stores
no prompt text.

---

### Task L3: Listwise reranker with strict permutations

**Files:**
- Create: `prompts/rerank/listwise-v1.yaml`, `backend/src/copilot/search/llm_rerank.py`
- Modify: `backend/src/copilot/contracts.py`, adding `ListwiseResult`
- Test: `backend/tests/unit/test_llm_rerank.py`

**Interfaces:**
- Consumes: `ChatClient.complete` and `LlmError` (L2).
- Produces:
  - in `contracts.py`: `ListwiseResult(order: list[int], ranked_by_model: int, cost_usd: float, served_model: str, cached: bool)`;
  - `load_prompt(path) -> ListwisePrompt`, with `.name`, `.sha256`, `.system`, `.user`;
  - `build_messages(prompt, query: str, texts: Sequence[str], words: int) -> list[dict[str, str]]`;
  - `parse_permutation(answer: str, count: int) -> tuple[list[int], int]`;
  - `LlmListwiseReranker(client: ChatClient, prompt: ListwisePrompt, *, words: int = 300)`,
    with `.identity: str`;
  - `LlmListwiseReranker.order(query, texts, *, timeout, purpose, request_id=None, run_id=None) -> ListwiseResult`.

- [ ] **Step 1: Write `prompts/rerank/listwise-v1.yaml`**:

```yaml
# Listwise reranking prompt, version 1. Its sha256 is part of the reranker's identity: any
# edit is a new ranker, compared as one. The format follows LitSearch's RankGPT adaptation
# (bracketed identifiers, "[i] > [j]" answers) in our own wording.
name: listwise-v1
system: >-
  You rank scientific papers by how well each one answers a literature-search request,
  judging only from the title and abstract shown.
user: |-
  Literature-search request: {query}

  Candidate papers, each with an identifier in square brackets:

  {candidates}

  Rank all {count} candidates from most to least relevant to the request. Answer with the
  identifiers only, in the form [3] > [1] > [2], and nothing else.
```

- [ ] **Step 2: Write the failing tests**:

```python
import pytest

from copilot.search.llm_rerank import build_messages, load_prompt, parse_permutation
from copilot.search.rerank import RerankError

PROMPT = load_prompt("prompts/rerank/listwise-v1.yaml")


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
        open("prompts/rerank/listwise-v1.yaml").read().replace("Rank all", "Order all")
    )
    assert load_prompt(edited).sha256 != PROMPT.sha256
```

- [ ] **Step 3: Run them and confirm they fail** on the missing module.

- [ ] **Step 4: Implement.** Fill the template in a single pass, so text inserted for one
  placeholder is never treated as another:

```python
_PLACEHOLDER = re.compile(r"\{(query|candidates|count)\}")
_BRACKETED = re.compile(r"\[(\d+)\]")
_BARE = re.compile(r"\b(\d+)\b")


def build_messages(prompt, query, texts, words):
    block = "\n\n".join(
        f"[{number}] {' '.join(text.split()[:words])}" for number, text in enumerate(texts, 1)
    )
    values = {"query": query, "candidates": block, "count": str(len(texts))}
    user = _PLACEHOLDER.sub(lambda match: values[match.group(1)], prompt.user)
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]


def parse_permutation(answer, count):
    """Indices into the candidates, best first, and how many the model itself placed.

    Unranked candidates follow in their incoming order, as LitSearch's reranker does, so the
    result is always a permutation of exactly the candidates given.
    """

    numbers = _BRACKETED.findall(answer) or _BARE.findall(answer)
    placed = list(dict.fromkeys(i for i in (int(n) - 1 for n in numbers) if 0 <= i < count))
    if not placed:
        raise RerankError("llm_rerank_unparseable")
    chosen = set(placed)
    return placed + [i for i in range(count) if i not in chosen], len(placed)
```

  `LlmListwiseReranker.identity` is `f"{model.key}={model.model}#{prompt.name}:{prompt.sha256[:12]}#w{words}"`.
  `order()` calls `client.complete` and parses the answer. It returns a `ListwiseResult` and
  lets `LlmError` and `RerankError` propagate: the service decides the fallback.

- [ ] **Step 5: Run the tests; expect PASS. Then run Ruff and mypy, and commit:**
  `git commit -m "feat: add listwise LLM reranking with validated permutations"`

**Acceptance cases:** clean answers; duplicates; out-of-range ids; unranked candidates appended
in order; bare numbers; an unparseable answer; truncation to 300 words; braces in a query or
abstract; a prompt edit changes the identity.

---

### Task L4: Serve `hybrid_rerank_llm` with explicit fallbacks

**Files:**
- Modify:
  - `backend/src/copilot/contracts.py` (the mode literal);
  - `backend/src/copilot/search/service.py`;
  - `backend/src/copilot/search/api.py`;
  - `backend/src/copilot/config.py` (`llm_rerank_model`, alias `LLM_RERANK_MODEL`);
  - `configs/search.yaml`;
  - `frontend/openapi.json`;
  - spec §6, §7, §10 and §12.
- Test: `backend/tests/integration/test_search_service.py`, `backend/tests/integration/test_search_api.py`

**Interfaces:**
- Consumes: `LlmListwiseReranker.order -> ListwiseResult` (L3), `LlmError` (L2), `build_client` (L2).
- Produces:
  - the mode `hybrid_rerank_llm`, with the same branches as `hybrid_rerank`;
  - `SearchConfig.llm_rerank_seconds` (default 20.0) and `SearchConfig.llm_total_seconds`
    (default 25.0);
  - `SearchService(..., listwise: ListwiseReranker | None = None)`;
  - the warnings `llm_rerank_timeout`, `llm_rerank_unavailable`, `llm_spend_cap`,
    `llm_rerank_unparseable` and `llm_rerank_failed`;
  - `scores[paper_id]["llm_rank"]`, the one-based position the LLM gave.

- [ ] **Step 1: Write the failing service tests.** Add two helpers to `test_search_service.py`.

  `FakeListwise(reverse: bool = False, fail: str | None = None)` has an `identity` of
  `"fake/listwise@v1"`. Its `order(query, texts, *, timeout, purpose, request_id=None, run_id=None)`
  returns a `ListwiseResult` whose order is the reversed indices. Depending on `fail`, it
  instead raises:
  - `"spend_cap"`: `LlmError("llm_spend_cap")`;
  - `"unparseable"`: `RerankError("llm_rerank_unparseable")`;
  - `"server_error"`: `LlmError("llm_server_error")`.

  `service_factory(listwise)` is a fixture. It builds a `SearchService` over the module's
  `search_corpus` with `FixtureEmbedding`, `FixtureReranker` and the deterministic runner that
  `test_timeouts_fall_back_explicitly` injects. When `listwise.fail == "timeout"`, that runner
  reports the `llm` stage as `Outcome(error=TIMEOUT)` without calling it, the same way that
  test times out the reranker. No real sleep.

```python
def test_the_llm_reorders_the_reranked_head_without_changing_its_members(service_factory):
    service = service_factory(listwise=FakeListwise(reverse=True))
    request = SearchRequest(query="graph contrastive learning", mode="hybrid_rerank_llm",
                            filters=PaperFilters(), limit=50)
    plain = service.rank(request.model_copy(update={"mode": "hybrid_rerank"}))
    deep = service.rank(request)
    assert [pid for pid, _ in deep.ordering.items] == [pid for pid, _ in plain.ordering.items][::-1]
    assert not deep.ordering.warnings
    assert all("llm_rank" in deep.ordering.scores[pid] for pid, _ in deep.ordering.items)


@pytest.mark.parametrize("failure, warning", [
    ("spend_cap", "llm_spend_cap"),
    ("unparseable", "llm_rerank_unparseable"),
    ("timeout", "llm_rerank_timeout"),
    ("server_error", "llm_rerank_failed"),
])
def test_an_llm_failure_serves_the_cross_encoder_order_and_says_why(service_factory, failure, warning):
    service = service_factory(listwise=FakeListwise(fail=failure))
    request = SearchRequest(query="graph contrastive learning", mode="hybrid_rerank_llm",
                            filters=PaperFilters(), limit=50)
    plain = service.rank(request.model_copy(update={"mode": "hybrid_rerank"}))
    deep = service.rank(request)
    assert deep.ordering.items == plain.ordering.items
    assert warning in deep.ordering.warnings


def test_without_a_configured_llm_the_mode_degrades_rather_than_failing(service_factory):
    deep = service_factory(listwise=None).rank(
        SearchRequest(query="graph", mode="hybrid_rerank_llm", filters=PaperFilters()))
    assert "llm_rerank_unavailable" in deep.ordering.warnings and deep.ordering.items
```

  In `test_search_api.py`:
  - `POST /v1/search` with `mode: hybrid_rerank_llm` and no LLM configured returns 200,
    `degraded: true`, `warnings` containing `llm_rerank_unavailable`;
  - the degraded ordering is not reused, per the existing cache rule;
  - the published schema test passes after re-exporting `frontend/openapi.json`.

- [ ] **Step 2: Run them and confirm they fail** on the unknown mode.

- [ ] **Step 3: Implement.**
  - **Mode and lane.** Add the mode to `_MODE_BRANCHES`, and the lane `"llm": 8` to
    `ThreadedStageRunner.LANES`. The LLM is remote, so concurrent searches must not queue
    behind one another.
  - **Where the stage runs.** In `rank()`, run the cross-encoder exactly as for
    `hybrid_rerank`, then `_llm_rerank`. Its budget is
    `min(llm_rerank_seconds, llm_total_seconds - elapsed)`. Earlier stages keep the 3 s total.
  - **On success.** Reorder the head by `result.order`, and set each item's score to
    `len(head) - position` (higher is better, an ordinal signal). Record `llm_rank`, plus a
    trace stage with seconds, `ranked_by_model`, `cost_usd`, `served_model` and `cached`.
  - **On failure.** Keep the head as the cross-encoder left it, with a warning by cause:
    - a stage `TIMEOUT` gives `llm_rerank_timeout`;
    - `LlmError("llm_spend_cap")` gives `llm_spend_cap`;
    - `RerankError("llm_rerank_unparseable")` gives `llm_rerank_unparseable`;
    - any other `LlmError` gives `llm_rerank_failed`;
    - no reranker configured gives `llm_rerank_unavailable`.
  - **Ranking identity.** `identity("hybrid_rerank_llm")` adds `listwise.identity`.
  - **Building it.** `default_search_service` builds the reranker when `llm_rerank_model` is
    set, its key and the cap are present, and the price is configured. Otherwise `listwise` is
    None and the mode serves `llm_rerank_unavailable`; startup never fails over it.
    `warm()` makes **no** LLM call.
  - **Configuration.** Add `llm_rerank: 20.0` and `llm_total: 25.0` under `deadlines_seconds`
    in `configs/search.yaml`.

- [ ] **Step 4: Run the tests; expect PASS. Re-export the schema:**
  `uv run --project backend python -m copilot.cli api export-schema --out frontend/openapi.json`

- [ ] **Step 5: Amend the spec** in the same commit:
  - §6: the mode literal;
  - §7: the LLM stage and its fallback;
  - §10: the new mode;
  - §12: OpenAI and DeepSeek as the providers, the deep-search p95 budget from decision 3, the
    spend ledger live from L1.

  Record decisions 1 and 3 in the tracker.

- [ ] **Step 6: Run the full suite, Ruff and mypy, then commit:**
  `git commit -m "feat: serve an opt-in LLM-reranked search mode with explicit fallbacks"`

**Acceptance cases:** the LLM only reorders; each failure keeps the cross-encoder order with its
own warning; an unconfigured LLM degrades rather than failing startup; `hybrid_rerank` is
byte-for-byte unchanged (its existing tests pass untouched); warm-up spends nothing; the schema
is re-exported.

---

### Task L5: Evaluate on LitSearch under a pre-registered rule

**Files:**
- Modify:
  - `backend/src/copilot/evaluation/retrieval.py`: the `llm` variant field, a spend preflight
    and run budget, LLM facts and real cost in the manifest, and LLM-free warm-up;
  - `backend/src/copilot/cli.py`: `--max-spend-usd`;
  - `backend/src/copilot/evaluation/report.py`: a cost column;
  - `backend/src/copilot/evaluation/regression.py`: `llm_changed` in `compare_runs`.
- Create: `configs/experiments/e3-llm-pilot.yaml`, `configs/experiments/e3-llm-rerank.yaml`,
  `reports/m2-llm-rerank.md`, `reports/retrieval/e3-llm-*/`
- Test: `backend/tests/unit/test_regression.py` (config validation),
  `backend/tests/integration/test_eval_runner.py` (a run with a fixture listwise model)

**Interfaces:**
- Consumes: the L1–L4 producers; P2.6's `Variant.search` overrides and `require_undegraded`.
- Produces:
  - `Variant.llm: str | None`, a `configs/llm.yaml` model key, required exactly when
    `mode == "hybrid_rerank_llm"`;
  - `ModelFactory.listwise(model_key: str, run_id: str) -> LlmListwiseReranker`;
  - `run_retrieval(..., max_spend_usd: float | None)`;
  - manifest fields:
    - `variants.<name>.llm = {model_key, model, pinned, served_models, prompt, prompt_sha256, words, calls, cache_hits, input_tokens, output_tokens, cost_usd}`;
    - `cost.metered_usd` is the run's real spend from the ledger;
  - `compare_runs(...)["llm_changed"]: list[str]` in `evaluation/regression.py`.

- [ ] **Step 1: Write the failing tests.**
  - **Config validation (unit):**
    - `llm` on a non-LLM mode is refused;
    - an LLM mode without `llm` is refused;
    - an unknown model key is refused.
  - **Integration, with a `FixtureModels.listwise` that reverses the head:**
    - a run records `variants.<name>.llm` with `served_models` and `prompt_sha256`;
    - `cost.metered_usd` equals the ledger's `run_spend` for the run;
    - a second run over the same cache reruns with `cache_hits == calls`, metered 0, and
      identical rankings;
    - `max_spend_usd` below the preflight estimate refuses with
      `ExperimentError("run_budget_exceeded")` before any query runs;
    - warm-up makes no LLM call.
  - **Comparison (unit, `test_regression.py`):** `compare_runs` reports `llm_changed`, listing
    shared variants whose `llm.served_models` or `llm.prompt_sha256` differ between the two runs.
    It sits beside P2.6's `search_changed`, so a silently moved DeepSeek alias shows up when
    pilot, development and validation runs are compared.

- [ ] **Step 2: Run them and confirm they fail; implement; run them and confirm they pass.**
  - **Preflight.** The estimate is the number of queries times `estimate_usd(price, words × 7
    characters × head size, max_output_tokens)`. A run with any LLM variant requires
    `--max-spend-usd`.
  - **Mid-run guard.** After each query, the run stops with `run_budget_exceeded` once
    `ledger.run_spend(run_id)` passes the budget, writing nothing partial.
  - **Warm-up.** An LLM variant's warm-up queries run in `hybrid_rerank` mode.
  - **Response cache.** Kept under `${DATA_DIR}/cache/llm/`.
  - **Then:** commit
    `git commit -m "feat: run LLM reranking variants with metered cost and a spend budget"`.

- [ ] **Step 3: Pilot, on development only.** Write `e3-llm-pilot.yaml`:
  - it runs over P2.6's final configuration;
  - one `hybrid_rerank_llm` variant per model key: `openai-gpt-6-luna` and
    `openai-gpt-6-luna-low`, plus `deepseek-flash` only if `DEEPSEEK_API_KEY` is set;
  - it has no `decision` block.

  Run it with `--limit-queries 20 --max-spend-usd 1` on development. For each model, record:
  - the mean `ranked_by_model / head size`;
  - unparseable and failed calls;
  - LLM-stage p95;
  - cost per query;
  - the served model.

  Keep a model for the full run only if it had zero unparseable answers, mean coverage ≥ 0.9,
  and LLM-stage p95 within the deep-search budget. Keep at most two, cheapest first: both
  `gpt-6-luna` variants if both survive, and `deepseek-flash` only in place of a failed one.
  Record the `low` variant's mean output tokens, which set its real cost per search. Write the
  choice and its numbers
  into `reports/m2-llm-rerank.md` **before** the development run.

- [ ] **Step 4: Development run** of the chosen variants plus `hybrid_rerank` on all 359
  development queries:
  `eval retrieval --config configs/experiments/e3-llm-rerank.yaml --split development --out reports/retrieval/e3-llm-rerank --max-spend-usd <estimate × 1.5>`.
  Then run `eval gaps` on it. Under the $3 daily cap, the two luna variants together can pass
  one day's cap here. A capped call degrades its query (`llm_spend_cap`). If that happens,
  rerun on the next UTC day: answers already received come back from the cache for free, so
  only the remainder is billed. Only a run with no degraded query is reported.

- [ ] **Step 5: Pre-register and commit the rule before validation.**
  `e3-llm-rerank.yaml`'s `decision`:
  - `primary: ndcg@10`;
  - `order: [hybrid_rerank, <cheaper LLM variant>, <costlier LLM variant>]`;
  - `latency_p95_seconds:` the deep-search budget from decision 3;
  - `require_undegraded: true`.

  A failed LLM call degrades the query, so a validation run with any failure is rerun. Its
  cached successes cost nothing, and only the failures are retried. Commit with
  `git commit -m "docs: pre-register the LLM reranking rule before its validation run"`.

- [ ] **Step 6: Validation run (120 queries), then the in-domain check.** The in-domain check
  repeats the chosen variant on P2.5's slice of our corpus (150 development and 52 validation
  queries), as secondary evidence.

- [ ] **Step 7: Write `reports/m2-llm-rerank.md`**:
  - commands and the GPU state;
  - every model's pilot table;
  - the decision with its interval;
  - actual cost by model, from `llm spend --run`;
  - served models.

  Report the LLM's gain separately for author-written and inline-citation queries. LitSearch's
  inline-citation questions were generated with an LLM, and the papers may sit in the models'
  training data, so a gain confined to those sets is weaker evidence. Compare beside the
  paper's Table 3 through `eval gaps`.

  Update the tracker with the task status, the decision, decision 2 (the cap actually used) and
  decision 4 (the provider data terms). Commit with
  `git commit -m "feat: evaluate LLM reranking on LitSearch under a pre-registered rule"`.

**Acceptance cases:**
- every rule is committed before its validation run;
- every manifest carries real metered cost, served models and the prompt digest;
- a cached rerun is free and identical;
- the budget preflight refuses an over-budget run;
- no LitSearch query text is committed;
- the locked test split is still unrun when L5 closes.

## Exit checkpoint

Done when L1–L5 are committed with evidence, the decision is recorded, and the spec and plans
reflect what shipped. Next: the locked test split, once, with the final configuration and
your go-ahead.
