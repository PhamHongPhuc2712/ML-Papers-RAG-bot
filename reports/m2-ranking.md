# M2 ranking evidence — fusion, bounded reranking, deadlines and fallbacks

Date: 2026-09-30
Task: P2.3, reproducible fusion and bounded cross-encoder reranking
Status: **done** — every plan acceptance case passes against the test services; the real
service answered 150 of 150 in-domain LitSearch development queries undegraded in
`hybrid_rerank` mode, p95 631 ms against a 3 s budget, on the active release.
Host: WSL2 Linux, 12 cores / 23 GB RAM, RTX 3080 Laptop 16 GB. `postgres:17.11-bookworm`,
`qdrant/qdrant:v1.19.1`, qdrant-client 1.19.0, torch 2.14.0+cu130, transformers 5.17.0,
tokenizers 0.23.2. Release `m2-20260924T095724Z` (85,729 paper points).
Code: branch `phuc` from `4b90a35`, plus this task's commit.

## What exists

| Piece | Where |
|---|---|
| `rrf(rankings, k=60)` — each list counts an ID once, ties broken by ID | `search/fusion.py` |
| Score validation, alignment-preserving reorder, the pinned cross-encoder adapter, a deterministic fixture reranker | `search/rerank.py` |
| `SearchService.search` / `search_with_trace`, the stage runner, metadata hydration, `SearchTrace` | `search/service.py` |
| Ranking parameters: 100 per branch, k = 60, depth 50, deadlines 1 / 1.5 / 3 s, one retry | `configs/search.yaml` |
| The reranker pin: repo, commit, every file's sha256, pair budget, batch, precision | `configs/models.yaml` |
| Pinned-file verification and fetching for any model, not only the embedder | `models/embeddings.py` |
| `search fetch-model` fetches both models; `search compare-precision --model reranker`; `search pilot` | `cli.py` |

## The contract as built

A request reads the active release **once** and passes that release id to every stage,
so the two branches can never answer from different collection pairs (test
`test_the_release_is_read_once_per_request`). Both branches receive the same query,
filters, limit (100) and release.

| Mode | Branches | Fusion | Reranker | Returns |
|---|---|---|---|---|
| `bm25` | lexical | none — the branch's order | never called | up to `limit` |
| `dense` | dense | none — the branch's order | never called | up to `limit` |
| `hybrid` | both, concurrently | RRF, k = 60 | never called | up to `limit` of the fused list |
| `hybrid_rerank` | both, concurrently | RRF, k = 60 | first 50 fused | up to `limit` **of the reranked 50** |

`hybrid_rerank` never appends the unreranked tail after the reranked head: that would
place reranker logits and RRF scores in one ordering, and they measure different things.
Every item carries the stage scores it earned (`bm25`, `dense`, `rrf`, `rerank`) as
ranking signals; the trace keeps each stage's full candidate list, seconds and the model
identities for offline analysis. Nothing presents a score as a probability.

### Degradation, all explicit

| What happens | Response | Warning |
|---|---|---|
| One branch errors or overruns 1 s | the other branch's own order, not fused | `lexical_unavailable` / `dense_unavailable` / `*_timeout` |
| Both branches fail | typed `SearchUnavailable("candidates_unavailable", retryable=True)` — never an empty page | — |
| No active release | `SearchUnavailable("corpus_not_ready", retryable=True)` | — |
| Reranker overruns 1.5 s, or the 3 s total has no room left | the RRF order | `rerank_timeout` |
| Reranker returns the wrong count or a non-finite score | the RRF order | `rerank_invalid_scores` |
| Reranker raises | retried once if the total deadline still has room, then the RRF order | `rerank_unavailable` |
| A candidate indexed but missing from PostgreSQL | dropped, never served with invented metadata | `metadata_missing` |

Any warning sets `degraded: true`. A timeout is never retried: it has already spent its
budget.

## Reranker pin and the precision comparison

`BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, fetched by
`search fetch-model` into `${DATA_DIR}/models/rerankers/` and verified on every load:
`config.json` `13dcd6c3…fe41a`, `tokenizer.json` `69564b69…79c15` (not the embedder's
file), `model.safetensors` `d9e3e081…b5286`. The load report lists nothing missing and
nothing unexpected. Pairs are truncated with `only_second` at 1,024 tokens, so the query
always survives whole; the pair budget is part of the identity
(`…#pair1024`), because P2.5 compares 512.

Spec §7's CPU-float32-first rule, applied to the reranker:

```text
search compare-precision --model reranker --release m2-20260924T095724Z --sample 10
  -> 10 sampled paper titles (seed 42) x their BM25 top 50 = 500 pairs
```

| Measure | Value |
|---|---|
| Top-10 overlap, GPU fp16 vs CPU fp32 — mean / min | 1.0 / 1.0 |
| Pairwise order agreement — mean / min | 0.99959 / 0.99837 |
| Largest absolute logit gap | 0.023 |
| Throughput, CPU fp32 / GPU fp16 | 0.8 / 23.8 pairs per second |

Every query's top 10 is the same set under both precisions, and 0.04% of pairs swap
order — pairs whose logits sit within the 0.023 gap. At 0.8 pairs/s, one CPU rerank of
50 candidates would take about a minute. **Decision: GPU float16**, CPU float32 kept as
the reference. A four-query spot check gave the same median for batches of 16 and 32
(546 ms), so 16, the smaller activation footprint, is configured.

## Pilot: stage timing on the real service

```text
search pilot --split development --traces m2-ranking-pilot-dev
  -> LitSearch development split, in-domain: 150 queries, mode hybrid_rerank, limit 20
  -> 2026-09-30 05:37:56 – 05:39:36 UTC
```

| Stage | p50 | p95 | max |
|---|---|---|---|
| lexical (BM25, top 100) | 6.9 ms | 9.2 ms | 19.7 ms |
| dense (encode + top 100) | 33.8 ms | 45.9 ms | 63.8 ms |
| hydrate (metadata for the 50 reranked) | 9.0 ms | 14.0 ms | 465.6 ms |
| rerank (50 pairs, fp16, batch 16) | 499.5 ms | 566.8 ms | 669.9 ms |
| **request** | **554.4 ms** | **631.4 ms** | **1,053.8 ms** |

150 of 150 requests undegraded, 0 failures, 0 warnings. The branches run concurrently,
so a request costs roughly the slower branch plus hydration plus reranking. Cold start —
the first request of the process, with first-call CUDA work and the BM25 statistics
load — is reported apart: 1,400 ms. The single 466 ms hydrate is one request's
PostgreSQL read; the p95 is 14 ms.

The GPU was sampled every second through the run: memory 595–4,094 MiB (the 595 MiB
floor is the host's idle baseline; the rest is the two models and their activations),
utilization median 96% — the reranker's own work. No other workload held the device.
Per-query traces went to `${DATA_DIR}/runs/m2-ranking-pilot-dev.jsonl`, outside the
repository, because they carry LitSearch query text, which has no declared license
(see [labels](m2-labels.md)). This report quotes none of it.

**No quality claim is made here.** The reranker's effect on recall and nDCG is P2.5's
ablation on the frozen test split with bootstrap intervals; this pilot measures only
time and degradation.

## Trace examples

Handwritten queries, not LitSearch's — `search pilot --query …` on the final code,
2026-09-30, GPU idle apart from this process (peak 3,432 MiB). Three requests, none
degraded: request p50 506 ms (lexical 7.0, dense 27.4, hydrate 14.2, rerank 449.6 ms),
cold start 1,237 ms. Top five of each; `rrf` is the fused score, `rerank` the
cross-encoder logit, both ranking signals only.

**"contrastive learning for sentence embeddings"** — stages: lexical 7.0 ms, dense
27.4 ms, hydrate 14.2 ms, rerank 428.9 ms

| # | Paper | bm25 | dense | rrf | rerank |
|---|---|---|---|---|---|
| 1 | Contrastive Learning of Sentence Embeddings from Scratch (EMNLP 2023) | 21.16 | 0.721 | 0.0318 | 6.906 |
| 2 | WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings (ACL 2023) | 16.31 | 0.713 | 0.0271 | 6.781 |
| 3 | miCSE: Mutual Information Contrastive Learning for Low-shot Sentence Embeddings (ACL 2023) | 20.91 | 0.727 | 0.0320 | 6.488 |
| 4 | SKICSE: Sentence Knowable Information Prompted by LLMs Improves Contrastive Sentence Embeddings (NAACL 2024) | 19.04 | 0.680 | 0.0277 | 5.797 |
| 5 | KDMCSE: Knowledge Distillation Multimodal Sentence Embeddings with Adaptive Angular margin Contrastive Learning (NAACL 2024) | 20.31 | 0.673 | 0.0271 | 5.746 |

**"graph neural networks for molecular property prediction"** — stages: lexical 7.5 ms,
dense 30.0 ms, hydrate 15.7 ms, rerank 449.6 ms

| # | Paper | bm25 | dense | rrf | rerank |
|---|---|---|---|---|---|
| 1 | MolHFCNet: Enhancing Molecular Graph Representations with Hierarchical Feature Combining and Hybrid Pretraining (IJCAI 2025) | 22.52 | 0.676 | 0.0285 | 6.012 |
| 2 | Learning Topology-Specific Experts for Molecular Property Prediction (AAAI 2023) | 24.83 | 0.647 | 0.0301 | 5.598 |
| 3 | Robust Heterogeneous Graph Classification for Molecular Property Prediction with Information Bottleneck (AAAI 2025) | 25.98 | 0.640 | 0.0294 | 5.305 |
| 4 | Fragment-based Pretraining and Finetuning on Molecular Graphs (NeurIPS 2023) | 23.53 | 0.678 | 0.0304 | 5.082 |
| 5 | Bi-level Contrastive Learning for Knowledge-Enhanced Molecule Representations (AAAI 2025) | 27.20 | 0.664 | 0.0311 | 4.930 |

**"reducing hallucination in retrieval augmented generation"** — stages: lexical 5.9 ms,
dense 25.5 ms, hydrate 13.6 ms, rerank 482.5 ms

| # | Paper | bm25 | dense | rrf | rerank |
|---|---|---|---|---|---|
| 1 | ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability (ICLR 2025) | 21.56 | 0.685 | 0.0317 | 5.590 |
| 2 | Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning (NeurIPS 2025) | 14.35 | 0.614 | 0.0131 | 5.234 |
| 3 | Removal of Hallucination on Hallucination: Debate-Augmented RAG (ACL 2025) | 23.90 | 0.699 | 0.0328 | 5.094 |
| 4 | Bridging External and Parametric Knowledge: Mitigating Hallucination of LLMs with Shared-Private Semantic Synergy in Dual-Stream Knowledge (EMNLP 2025) | 18.43 | — | 0.0132 | 4.906 |
| 5 | Micro-Macro Retrieval: Reducing Long-Form Hallucination in Large Language Models (ICLR 2026) | 22.12 | 0.638 | 0.0275 | 4.867 |

What the traces show. The reranker reorders only inside the fused head: in the third
query, ranks 2 and 4 sat far down the RRF list (0.0131 and 0.0132 against 0.0328 at the
top) and were lifted from within the 50, not added. Rank 4 has no `dense` score — the
dense branch did not return it in its top 100 — so RRF credited it from BM25 alone, and
the item says so instead of reporting a zero. Fused scores follow `1/(60+rank)`: the
third query's RRF leader, "Removal of Hallucination on Hallucination" at 0.0328, is first
in both lists (2/61 ≈ 0.0328, the largest two-list score), and the reranker placed it
third.

**The same query, degraded.** Run earlier the same day while the Windows-side workload
held the GPU (see below), the third query's rerank overran 1.5 s. The response came back
in RRF order with `degraded: true` and `warnings: ["rerank_timeout"]`, each item
carrying `bm25`, `dense` and `rrf` but no `rerank` — rank 1 was "Removal of
Hallucination on Hallucination" (`rrf` 0.0328), the fused leader. In the runs before
that, the dense branch overran as well, and the page was BM25's own order under
`["dense_timeout", "rerank_timeout"]`, carrying only `bm25` scores: no fusion is
performed over one list.

## An incident the pilot found: a shared GPU and abandoned stages

The first pilot runs degraded **every** request — `dense_timeout` and `rerank_timeout`
on all of them, BM25 alone answering in about 2.5 s. The fallback behaved as specified;
the cause needed finding.

**The device was not ours.** `nvidia-smi` showed 14.7 GB of the 16 GB in use and 53–100%
utilization with none of our processes running, held by a process WSL cannot see
(`[Not Found]`, also invisible from other containers) — a Windows-side workload. With
about 1.6 GB free for 2.3 GB of weights, the driver paged GPU memory, and alternating
between the embedder and the reranker thrashed it. Timed outside the service, a query
encode took 16–28 ms steady but 0.3–3.3 s just after the other model ran, and Qdrant's
first dense searches after the container restart took 0.4–1.8 s while its pages loaded.

**The runner made it worse.** Python cannot cancel a running thread, so a stage past its
deadline keeps running. Instrumenting every call inside the service showed the
consequence: six requests in a row left six reranks of 50 pairs running, none finished
eight seconds after the last request, and each new query encode queued behind them and
took 3–11 s. All stages shared one pool of eight threads, so enough overruns would have
held every thread and starved BM25 too — the runner's docstring claimed the opposite.

**Fix, in this task.** `ThreadedStageRunner` now gives each stage kind its own lane —
four threads for lexical, one for dense, one for rerank — so overrunning model calls
hold only their own lane, and a model lane's single worker makes abandoned calls wait
their turn instead of piling onto the device. A stage still **queued** when its deadline
passes is cancelled and never runs, since its answer could no longer be used. Two
regression tests fail on the old runner and pass on the new one, deterministically —
the overrunning stage waits on an event, not a sleep:

- `test_abandoned_model_work_cannot_starve_the_lexical_branch` — twelve overrunning
  reranks, then BM25 still answers;
- `test_a_stage_still_queued_at_its_deadline_never_runs`.

When the external workload stopped, the same instrumented run completed six of six
requests undegraded (0.6–1.0 s each, reranks 0.48–0.64 s). The 150-query timing above
was taken with the GPU sampled and uncontended.

What remains true, and belongs to operations rather than to this task: this host's GPU
is shared with whatever the Windows side runs, and when that workload holds most of the
memory, `hybrid_rerank` degrades to BM25 or RRF on every request — explicitly, within
the deadline, but degraded. A trace with `rerank_timeout` in `warnings` is the signal.

## Deterministic fallback tests

`backend/tests/integration/test_search_service.py` builds a real 8-paper release
(`search-a`, both collections, fixture embedding) in the test services and runs the
service against it. Deadlines are injected through a `ScriptedRunner` that reports named
stages as timed out without waiting, and a clock that is read, never slept on. 20 tests:

| Acceptance case | Test |
|---|---|
| Happy path with metadata and every stage score | `test_hybrid_rerank_returns_ranked_papers_with_metadata_and_every_stage_score` (real threaded runner) |
| Identical branch filters and release | `test_both_branches_receive_the_same_query_filters_and_release` |
| BM25/dense bypass fusion and reranker | `test_single_branch_modes_bypass_fusion_and_the_reranker` |
| Hybrid bypasses the reranker, equals `rrf` | `test_hybrid_fuses_by_rrf_and_never_calls_the_reranker` |
| Empty candidates | `test_no_candidates_is_an_empty_page_not_an_error` |
| Reranker reordering keeps the set | `test_the_reranker_reorders_the_fused_set_without_changing_it` |
| Mismatched count / non-finite scores | `test_unusable_reranker_scores_leave_the_fusion_order_and_say_so` (one score short; NaN) |
| One branch failure | `test_one_failed_branch_serves_the_other_with_a_warning` |
| Both branches failing | `test_both_branches_failing_is_a_retryable_error_not_an_empty_page` |
| Timeout fallback explicitly degraded | `test_timeouts_fall_back_explicitly` |
| Retry only model errors, within the deadline | `test_a_model_error_is_retried_once_within_the_deadline`, `test_an_exhausted_total_deadline_skips_the_reranker` |
| One release per request | `test_the_release_is_read_once_per_request`, `test_no_active_release_is_a_retryable_error` |
| Typed error codes reach the trace | `test_the_reranker_error_type_is_what_the_trace_records` |
| Missing metadata | `test_a_candidate_missing_from_the_database_is_dropped_and_reported` |
| Hydration time is traced | `test_hydrate_time_includes_the_metadata_the_reranker_reads` |
| Overruns cannot starve other stages | the two runner tests above |

`backend/tests/unit/test_fusion.py`, 12 tests: the plan's regression test verbatim; ties
broken by ID; a repeated ID counted once per list; empty input; one list keeps its
order; `k < 1` refused; `rerank_order` keeps ID alignment and breaks ties by incoming
order; miscounted and non-finite scores refused; the fixture reranker deterministic; the
reranker pin refuses anything but a 40-character commit.

## Deviations from the plan, and why

- **Where the reranker is pinned.** The plan said "pinned `BAAI/bge-reranker-v2-m3`"
  without saying where; it now sits in `configs/models.yaml` beside the embedder, and the
  plan was amended to say so before implementation (`4b90a35`).
- **`models/embeddings.py` generalized.** File verification and fetching took an
  embedding spec; they now take any `PinnedModel`, so the reranker reuses the same
  sha256-checked, offline-only path instead of a copy of it.
- **`hybrid_rerank` returns at most 50.** The plan says rerank the first 50 and return up
  to 20 by default; with a larger `limit`, the unreranked tail is not appended (above).
- **The stage runner uses lanes.** Not in the plan; added after the pilot showed a
  shared pool could starve BM25 behind abandoned model calls.
- **`hydrate` is a traced stage.** The plan asks for metadata hydrated separately; the
  trace now times every metadata load, including the one that supplies the reranker's
  texts, which the first pilot had left out of every stage.

## Commands and results

```text
uv run --project backend ruff check backend                                     -> All checks passed
uv run --project backend mypy --config-file backend/pyproject.toml backend/src  -> no issues in 48 files
uv run --env-file .env.test --project backend pytest \
    backend/tests/unit/test_fusion.py backend/tests/integration/test_search_service.py -q
                                                                                -> 32 passed
uv run --env-file .env.test --project backend pytest backend/tests -q           -> 356 passed in 2m15s
uv run --project backend pytest backend/tests -m "not integration" -q           -> 240 passed
git diff --check                                                                -> clean
```
