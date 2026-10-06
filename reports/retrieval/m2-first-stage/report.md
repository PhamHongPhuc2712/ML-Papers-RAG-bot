# Retrieval experiment — m2-first-stage

Rendered by `eval report` from `reports/retrieval/m2-first-stage/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development |
|---|---|
| Run | `m2-first-stage-development-20261006T062016Z` |
| Code | `eaf4bb450c` + uncommitted `2fd7490a0c` |
| Corpus release | `m2-20260924T095724Z` |
| Dataset | `dfcb643e87b5`, in_domain |
| Queries | 150 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 73% / max 99%; memory 3633–4862 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 150 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25_chunks` | 0.667 [0.593, 0.740] | 0.800 [0.733, 0.853] | 0.517 [0.449, 0.585] | 0.470 [0.401, 0.540] | 0.067 | 0.853 | 0 | 0 |
| `dense_chunks` | 0.573 [0.500, 0.653] | 0.720 [0.647, 0.793] | 0.432 [0.367, 0.501] | 0.387 [0.322, 0.457] | 0.058 | 0.793 | 2 | 0 |
| `hybrid_rerank` | 0.740 [0.667, 0.807] | 0.807 [0.740, 0.867] | 0.585 [0.523, 0.655] | 0.536 [0.473, 0.609] | 0.074 | 0.900 | 0 | 0 |
| `hybrid_rerank_both` | 0.767 [0.700, 0.827] | 0.880 [0.827, 0.927] | 0.597 [0.536, 0.661] | 0.543 [0.478, 0.612] | 0.077 | 0.927 | 0 | 0 |
| `hybrid_rerank_chunks` | 0.773 [0.707, 0.840] | 0.840 [0.780, 0.893] | 0.606 [0.543, 0.670] | 0.553 [0.486, 0.623] | 0.077 | 0.913 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25_chunks` | 53.9 | 180.1 | 363.3 | 179.9 | — | — | 15.981 |
| `dense_chunks` | 103.5 | 336.5 | 1033.8 | — | 287.7 | — | 16.286 |
| `hybrid_rerank` | 554.5 | 620.9 | 1187.2 | 10.4 | 47.4 | 561.3 | 11.344 |
| `hybrid_rerank_both` | 578.0 | 638.0 | 668.9 | 39.8 | 85.2 | 552.9 | 15.91 |
| `hybrid_rerank_chunks` | 565.3 | 639.5 | 1127.8 | 84.3 | 89.1 | 549.2 | 14.577 |

nDCG@10 by query set:

| Variant | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|
| `bm25_chunks` | 0.506 (3) | 0.551 (93) | 0.461 (54) |
| `dense_chunks` | 1.000 (3) | 0.474 (93) | 0.327 (54) |
| `hybrid_rerank` | 0.655 (3) | 0.657 (93) | 0.458 (54) |
| `hybrid_rerank_both` | 0.649 (3) | 0.662 (93) | 0.483 (54) |
| `hybrid_rerank_chunks` | 0.662 (3) | 0.673 (93) | 0.487 (54) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `bm25_chunks` | 0.332 (25) | 0.554 (125) |
| `dense_chunks` | 0.391 (25) | 0.440 (125) |
| `hybrid_rerank` | 0.465 (25) | 0.610 (125) |
| `hybrid_rerank_both` | 0.494 (25) | 0.618 (125) |
| `hybrid_rerank_chunks` | 0.484 (25) | 0.630 (125) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `hybrid_rerank_both` vs `hybrid_rerank` | ndcg@10 | 0.585 | 0.597 | +0.012 [-0.006, +0.032] | 12 / 9 / 129 |
| `hybrid_rerank_both` vs `hybrid_rerank` | recall@50 | 0.807 | 0.880 | +0.073 [+0.027, +0.127] | 13 / 2 / 135 |
| `hybrid_rerank_both` vs `hybrid_rerank` | mrr@10 | 0.536 | 0.543 | +0.007 [-0.010, +0.026] | 12 / 9 / 129 |
| `hybrid_rerank_chunks` vs `hybrid_rerank` | ndcg@10 | 0.585 | 0.606 | +0.020 [-0.006, +0.046] | 24 / 10 / 116 |
| `hybrid_rerank_chunks` vs `hybrid_rerank` | recall@50 | 0.807 | 0.840 | +0.033 [-0.020, +0.093] | 11 / 6 / 133 |
| `hybrid_rerank_chunks` vs `hybrid_rerank` | mrr@10 | 0.536 | 0.553 | +0.016 [-0.009, +0.042] | 24 / 10 / 116 |
