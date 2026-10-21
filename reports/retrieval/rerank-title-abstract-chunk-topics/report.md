# Retrieval experiment — rerank-title-abstract-chunk-topics

Rendered by `eval report` from `reports/retrieval/rerank-title-abstract-chunk-topics/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development |
|---|---|
| Run | `rerank-title-abstract-chunk-topics-development-20261020T064827Z` |
| Code | `cb7bbbfa43` + uncommitted `977c77d7f0` |
| Corpus release | `m2-20260924T095724Z` |
| Dataset | `1a24aa151aa5`, all |
| Queries | 349 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 99% / max 100%; memory 6541–9805 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no |

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
