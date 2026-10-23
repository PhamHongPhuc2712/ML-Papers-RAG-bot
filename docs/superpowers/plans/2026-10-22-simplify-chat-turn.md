# Replace the linear LangGraph chat turn: implementation plan

**Status:** Implemented, 2026-10-22. Verification is recorded in [the implementation report](../../../reports/langgraph-simplification.md).

## Goal and scope

Remove the direct `langgraph` dependency and represent the existing one-pass chat turn as ordinary, typed Python. Keep the product contract: an optional follow-up rewrite, exactly one paper search, evidence retrieval inside the selected papers, one grounded generation attempt with its existing bounded citation-repair behavior, and a persisted answer sent over SSE only after validation. Keep `ChatOpenAI` structured generation and its spend ledger. Do not add an agent loop, another search, a new model call, or a new benchmark decision.

This is a maintainability change. It should not claim an answer-quality or latency gain without a separate measurement.

## Current coupling

- `chat/graph.py` builds the fixed `condense -> retrieve -> generate` `StateGraph`; there are no branches, graph checkpoints, or graph-owned history. It also constructs the production search, evidence, condenser, generator, and turn identity.
- `chat/service.py` consumes graph update/value frames to emit `chat`, `status`, `references`, `evidence`, `final`, and `error` SSE frames. `ChatStore` owns idempotency, ownership, daily limits, persistence, interruption, and replay.
- `chat/retriever.py` converts `Evidence` to LangChain `Document` and back to satisfy `BaseRetriever`. The chat and follow-up evaluation are its only consumers; the search service and evidence retriever already return domain objects.
- `evaluation/chat_eval.py` constructs two graphs, with condensing on/off, and invokes them for the standalone, raw, and condensed conditions. The saved experiment format and historical results need no migration.
- `app.py` and `chat/api.py` inject a graph into the API. Unit and integration tests construct fixture graphs.
- `models/lc_chat.py` uses `langchain-core` and `langchain-openai` for structured `ChatOpenAI` and ledger callbacks. These dependencies stay. Only `langgraph` and its unneeded transitive packages should leave the lockfile.

## Proposed design

Add `chat/pipeline.py` with a `TurnPipeline` holding the existing `Condenser`, a domain-level `CorpusRetriever`, the existing generator, `condense_enabled`, `history_turns`, and the unchanged identity dictionary. Use a typed `TurnState` for query, evidence, paper IDs, references, release ID, answer outcome, warnings, timings, and cost. Give the pipeline one execution path:

1. `iter_steps(message, history, request_id=None, run_id=None)` executes the three operations in order and yields a typed update **after** each step: `condense`, `retrieve`, `generate`. Its accumulated state is available on each update. Condensing disabled yields the raw message and `condense: 0.0`; a first turn retains the condenser's existing no-call behavior. Exceptions propagate unchanged.
2. `run(...)` consumes `iter_steps` and returns the same final `TurnState`. The evaluation harness uses this method; the SSE service consumes `iter_steps`. There must be one implementation of the step logic, not a second evaluation-only version.
3. `build_default_turn_pipeline(...)` moves the production wiring from `build_default_chat_graph` without changing config reads, model construction, search mode, evidence settings, mock behavior, ledger purpose, cache, or the stored identity keys/values.

Keep `ServicePaperSearch` and `PaperHits`. Make `CorpusRetriever` a small typed adapter whose `search_evidence(query)` returns `(list[Evidence], PaperHits)` directly. Remove its `BaseRetriever` inheritance and `Document` round trip; no production or evaluation path calls it through LangChain's retriever interface. One integration test does, and should be updated to check the domain interface. The name can be kept to limit churn, but its docstring should describe that interface.

`ChatService` translates typed step updates into the existing SSE sequence. In particular, emit References and evidence as soon as retrieval completes, before generation; store the validated result before `final`; replay must still emit only `chat` and `final`. Keep `try`/`except`/`finally` ownership of failed and disconnected turns in the service. Preserve warning order (`condense`, retrieval, generation), per-step timing keys, total cost (including condenser and repair calls), query, release ID, evidence snapshot, References, and `identity` on the stored message. No database or HTTP schema migration is needed.

## Implementation tasks

### 1. Freeze the behavior at the boundary

- Extend `test_chat_graph.py` (to be renamed for the pipeline) and `test_chat_stream.py` with focused assertions for one search, step order, first-turn no-call condensing, disabled condensing, fallback warning/cost, spend-cap propagation before search, empty-evidence abstention without generation, generation validation warning, reference ordering, timings and total cost.
- Preserve the existing integration assertions for HTTP status, exact SSE event order and payloads, final-before-history visibility, replay without rerunning, failed retry, client disconnect, ownership, and the daily limit. Add a follow-up SSE case so `condensing -> searching(query) -> references -> evidence -> writing -> final` is locked down.
- Use fixture models and the isolated test services. This refactor does not require paid LLM calls or a frozen benchmark rerun.

### 2. Replace graph execution

- Implement the typed pipeline and direct evidence adapter. Move production construction from `chat/graph.py` to `chat/pipeline.py` and delete `chat/graph.py` once all imports have moved.
- Update `chat/service.py`, `chat/api.py`, and `app.py` to inject `TurnPipeline` (`chat_pipeline` in internal test overrides). Keep startup failure handling and `chat_unavailable` behavior.
- Update `evaluation/chat_eval.py` to construct two pipelines and call `run` for the existing three conditions. Keep the result columns, identities, budget checks, and saved-run format identical. Rename internal variables that call pipelines “graphs”; do not rewrite historical report text as though those runs used the new implementation.
- Update the affected unit/integration tests to use the domain objects instead of `Document`. Remove the round-trip test and replace it with an evidence-preservation assertion at the retriever boundary.

### 3. Remove dependency and align current documentation

- Run `uv remove --project backend langgraph` and review `backend/uv.lock` for `langgraph`, checkpoint, prebuilt, and SDK packages no longer needed. Retain `langchain-core` and `langchain-openai` for `ChatOpenAI`.
- Replace current-architecture references to a LangGraph turn or LangChain `BaseRetriever` in `README.md`, `CLAUDE.md`, `docs/case-study.md`, and the active technical spec. Amend the spec's 2026-10-11 design decision with a dated note; preserve the historical implementation plan and measured reports as history.
- Confirm `rg -n 'langgraph|BaseRetriever|from_document|to_document|chat_graph' backend/src backend/tests backend/pyproject.toml` finds no live dependency. Historical plans and reports may still mention them.

## Verification and exit criteria

1. Run targeted chat pipeline, retriever, stream, and follow-up-evaluation tests, then the full offline suite, Ruff, and strict mypy. Run the full integration suite with `.env.test` and the isolated Postgres/Qdrant services; never use `.env` for pytest.
2. Run the deterministic QA and chat evaluation smoke checks, if available without a hosted model, to confirm the harness output shape and costs. Compare a fixed fixture turn before/after for query, evidence IDs, answer claims, warnings, timings keys, cost, References, and identity. Numeric timings need not match.
3. Run frontend typecheck and the chat/browser tests against the scripted backend to confirm the unchanged SSE contract. No screenshot pass is needed if the UI is untouched.
4. Inspect the dependency diff and `git diff --check`. Update `docs/superpowers/progress.md` and a short report with the actual commands and outcomes only after implementation. The task is complete when no production or evaluation path imports LangGraph, all required checks pass, and the user-visible turn contract is unchanged.

**Rollback:** Revert the implementation commit. There is no stored graph checkpoint or schema change to migrate back; existing chats remain readable because their stored answer and identity fields are preserved.
