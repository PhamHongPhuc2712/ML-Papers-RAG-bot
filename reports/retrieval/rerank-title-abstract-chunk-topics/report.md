# Retrieval experiment — rerank-title-abstract-chunk-topics

Rendered by `eval report` from `reports/retrieval/rerank-title-abstract-chunk-topics/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation |
|---|---|---|
| Run | `rerank-title-abstract-chunk-topics-development-20261020T064827Z` | `rerank-title-abstract-chunk-topics-validation-20261021T074704Z` |
| Code | `cb7bbbfa43` + uncommitted `977c77d7f0` | `2b90f370dc` + uncommitted `01dbd60564` |
| Corpus release | `m2-20260924T095724Z` | `m2-20260924T095724Z` |
| Dataset | `1a24aa151aa5`, all | `1a24aa151aa5`, all |
| Queries | 349 | 349 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 99% / max 100%; memory 6541–9805 MiB; 1 GPU processes seen; before the run 0% busy | util p50 98% / max 100%; memory 3230–6413 MiB; 1 GPU processes seen; before the run 57% busy — **contended** |
| Locked test | no | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 349 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 0.716 [0.670, 0.759] | 0.782 [0.739, 0.825] | 0.616 [0.571, 0.659] | 0.584 [0.537, 0.630] | 0.072 | 0.862 | 0 | 0 |
| `shipped` | 0.653 [0.599, 0.705] | 0.782 [0.739, 0.825] | 0.527 [0.482, 0.572] | 0.488 [0.442, 0.533] | 0.065 | 0.862 | 0 | 0 |
| `title_abstract_chunk` | 0.691 [0.642, 0.736] | 0.782 [0.739, 0.825] | 0.583 [0.537, 0.627] | 0.549 [0.501, 0.597] | 0.069 | 0.862 | 0 | 1 |
| `title_abstract_chunk_1536` | 0.702 [0.653, 0.748] | 0.782 [0.739, 0.825] | 0.583 [0.535, 0.630] | 0.546 [0.495, 0.593] | 0.070 | 0.862 | 0 | 1 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 544.1 | 592.8 | 640.6 | 41.5 | 54.3 | 538.3 | 16.028 |
| `shipped` | 1069.1 | 1226.6 | 1718.6 | 68.3 | 89.5 | 1121.6 | 19.533 |
| `title_abstract_chunk` | 1329.5 | 1441.7 | 2501.9 | 34.2 | 58.1 | 1369.3 | 24.056 |
| `title_abstract_chunk_1536` | 1561.9 | 1776.3 | 2687.8 | 48.2 | 63.7 | 1698.7 | 27.856 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | ndcg@10 | 0.527 | 0.616 | +0.089 [+0.061, +0.122] | 88 / 31 / 230 |
| `abstract_reranker` vs `shipped` | recall@50 | 0.782 | 0.782 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `abstract_reranker` vs `shipped` | mrr@10 | 0.488 | 0.584 | +0.096 [+0.065, +0.132] | 88 / 31 / 230 |
| `title_abstract_chunk_1536` vs `shipped` | ndcg@10 | 0.527 | 0.583 | +0.056 [+0.031, +0.083] | 67 / 39 / 243 |
| `title_abstract_chunk_1536` vs `shipped` | recall@50 | 0.782 | 0.782 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `title_abstract_chunk_1536` vs `shipped` | mrr@10 | 0.488 | 0.546 | +0.058 [+0.028, +0.089] | 67 / 39 / 243 |
| `title_abstract_chunk` vs `shipped` | ndcg@10 | 0.527 | 0.583 | +0.056 [+0.031, +0.082] | 69 / 36 / 244 |
| `title_abstract_chunk` vs `shipped` | recall@50 | 0.782 | 0.782 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `title_abstract_chunk` vs `shipped` | mrr@10 | 0.488 | 0.549 | +0.061 [+0.032, +0.091] | 69 / 36 / 244 |

## Validation — 349 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 0.765 [0.716, 0.805] | 0.811 [0.768, 0.848] | 0.663 [0.615, 0.706] | 0.630 [0.580, 0.676] | 0.077 | 0.894 | 0 | 0 |
| `shipped` | 0.708 [0.656, 0.754] | 0.811 [0.768, 0.848] | 0.567 [0.521, 0.614] | 0.522 [0.476, 0.572] | 0.071 | 0.894 | 0 | 0 |
| `title_abstract_chunk` | 0.734 [0.685, 0.779] | 0.811 [0.768, 0.848] | 0.582 [0.537, 0.625] | 0.534 [0.488, 0.577] | 0.073 | 0.894 | 0 | 1 |
| `title_abstract_chunk_1536` | 0.734 [0.685, 0.779] | 0.811 [0.768, 0.848] | 0.581 [0.535, 0.624] | 0.533 [0.485, 0.577] | 0.073 | 0.894 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 562.9 | 632.1 | 1331.1 | 50.8 | 81.7 | 555.3 | 15.437 |
| `shipped` | 1141.8 | 1351.2 | 1958.2 | 127.7 | 170.1 | 1160.6 | 20.055 |
| `title_abstract_chunk` | 1361.4 | 1445.4 | 3000.5 | 33.2 | 61.3 | 1381.5 | 22.218 |
| `title_abstract_chunk_1536` | 1578.2 | 1815.7 | 2166.5 | 32.8 | 62.6 | 1743.5 | 25.943 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | ndcg@10 | 0.567 | 0.663 | +0.096 [+0.066, +0.125] | 98 / 32 / 219 |
| `abstract_reranker` vs `shipped` | recall@50 | 0.811 | 0.811 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `abstract_reranker` vs `shipped` | mrr@10 | 0.522 | 0.630 | +0.107 [+0.071, +0.142] | 98 / 32 / 219 |
| `title_abstract_chunk_1536` vs `shipped` | ndcg@10 | 0.567 | 0.581 | +0.014 [-0.011, +0.037] | 60 / 62 / 227 |
| `title_abstract_chunk_1536` vs `shipped` | recall@50 | 0.811 | 0.811 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `title_abstract_chunk_1536` vs `shipped` | mrr@10 | 0.522 | 0.533 | +0.010 [-0.019, +0.036] | 60 / 62 / 227 |
| `title_abstract_chunk` vs `shipped` | ndcg@10 | 0.567 | 0.582 | +0.015 [-0.009, +0.040] | 61 / 62 / 226 |
| `title_abstract_chunk` vs `shipped` | recall@50 | 0.811 | 0.811 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `title_abstract_chunk` vs `shipped` | mrr@10 | 0.522 | 0.534 | +0.011 [-0.016, +0.038] | 61 / 62 / 226 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03. A candidate with any degraded query is never chosen (`require_undegraded`).

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Degraded | Promoted |
|---|---|---|---|---|---|---|

| Ablation | Change | Difference [interval] | Non-inferior | Applies to the choice | Adopted |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | the former search page's reranker, title + abstract at a 1.5 s rerank deadline | +0.096 [+0.066, +0.125] | yes | yes | **yes** |
| `title_abstract_chunk` vs `shipped` | the reranker reads title + abstract + the chunk that found the paper, 1,024-token pairs | +0.015 [-0.009, +0.040] | yes | yes | no |
| `title_abstract_chunk_1536` vs `shipped` | title + abstract + chunk with 1,536-token pairs, so less of the chunk is cut | +0.014 [-0.011, +0.037] | yes | yes | **yes** |

**Chosen on validation: `shipped` (mode `hybrid_rerank`); ablations adopted: `abstract_reranker`, `title_abstract_chunk_1536`.**
