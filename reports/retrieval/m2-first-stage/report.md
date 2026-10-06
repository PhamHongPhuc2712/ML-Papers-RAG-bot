# Retrieval experiment — m2-first-stage

Rendered by `eval report` from `reports/retrieval/m2-first-stage/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation |
|---|---|---|
| Run | `m2-first-stage-development-20261006T062016Z` | `m2-first-stage-validation-20261006T062929Z` |
| Code | `eaf4bb450c` + uncommitted `2fd7490a0c` | `8654f22043` |
| Corpus release | `m2-20260924T095724Z` | `m2-20260924T095724Z` |
| Dataset | `dfcb643e87b5`, in_domain | `dfcb643e87b5`, in_domain |
| Queries | 150 | 52 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 73% / max 99%; memory 3633–4862 MiB; 1 GPU processes seen; before the run 0% busy | util p50 67% / max 99%; memory 3634–4604 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no | no |

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

## Validation — 52 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25_chunks` | 0.615 [0.481, 0.750] | 0.885 [0.788, 0.962] | 0.506 [0.389, 0.623] | 0.469 [0.351, 0.590] | 0.062 | 0.923 | 0 | 0 |
| `dense_chunks` | 0.538 [0.404, 0.673] | 0.788 [0.673, 0.885] | 0.440 [0.319, 0.559] | 0.409 [0.293, 0.530] | 0.054 | 0.827 | 0 | 0 |
| `hybrid_rerank` | 0.635 [0.500, 0.750] | 0.769 [0.654, 0.865] | 0.516 [0.396, 0.626] | 0.477 [0.356, 0.590] | 0.063 | 0.846 | 0 | 0 |
| `hybrid_rerank_both` | 0.615 [0.481, 0.731] | 0.846 [0.750, 0.942] | 0.514 [0.389, 0.627] | 0.481 [0.361, 0.593] | 0.062 | 0.942 | 0 | 0 |
| `hybrid_rerank_chunks` | 0.712 [0.596, 0.827] | 0.904 [0.808, 0.981] | 0.576 [0.450, 0.686] | 0.535 [0.409, 0.650] | 0.071 | 0.962 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25_chunks` | 63.8 | 101.3 | 110.5 | 101.1 | — | — | 13.587 |
| `dense_chunks` | 113.6 | 163.9 | 173.5 | — | 163.7 | — | 8.934 |
| `hybrid_rerank` | 557.7 | 698.8 | 1286.2 | 13.5 | 198.3 | 552.9 | 11.778 |
| `hybrid_rerank_both` | 601.4 | 684.0 | 1365.9 | 45.1 | 116.8 | 549.2 | 14.363 |
| `hybrid_rerank_chunks` | 558.1 | 596.6 | 1159.6 | 37.6 | 54.0 | 533.2 | 13.923 |

nDCG@10 by query set:

| Variant | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|
| `bm25_chunks` | 0.333 (3) | 0.452 (31) | 0.626 (18) |
| `dense_chunks` | 0.100 (3) | 0.447 (31) | 0.484 (18) |
| `hybrid_rerank` | 0.333 (3) | 0.530 (31) | 0.521 (18) |
| `hybrid_rerank_both` | 0.333 (3) | 0.530 (31) | 0.517 (18) |
| `hybrid_rerank_chunks` | 0.333 (3) | 0.572 (31) | 0.622 (18) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `bm25_chunks` | 0.254 (4) | 0.526 (48) |
| `dense_chunks` | 0.250 (4) | 0.455 (48) |
| `hybrid_rerank` | 0.125 (4) | 0.548 (48) |
| `hybrid_rerank_both` | 0.158 (4) | 0.544 (48) |
| `hybrid_rerank_chunks` | 0.233 (4) | 0.604 (48) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `hybrid_rerank_both` vs `hybrid_rerank_chunks` | ndcg@10 | 0.576 | 0.514 | -0.062 [-0.101, -0.026] | 0 / 11 / 41 |
| `hybrid_rerank_both` vs `hybrid_rerank_chunks` | recall@50 | 0.904 | 0.846 | -0.058 [-0.135, +0.000] | 0 / 3 / 49 |
| `hybrid_rerank_both` vs `hybrid_rerank_chunks` | mrr@10 | 0.535 | 0.481 | -0.055 [-0.100, -0.018] | 0 / 11 / 41 |
| `hybrid_rerank_chunks` vs `hybrid_rerank` | ndcg@10 | 0.516 | 0.576 | +0.060 [+0.025, +0.097] | 10 / 1 / 41 |
| `hybrid_rerank_chunks` vs `hybrid_rerank` | recall@50 | 0.769 | 0.904 | +0.135 [+0.058, +0.231] | 7 / 0 / 45 |
| `hybrid_rerank_chunks` vs `hybrid_rerank` | mrr@10 | 0.477 | 0.535 | +0.058 [+0.020, +0.102] | 10 / 1 / 41 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `recall@50`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03. A candidate with any degraded query is never chosen (`require_undegraded`). Guards: a step is promoted only if its `ndcg@10` interval rules out losing more than 0.03.

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Degraded | Guards | Promoted |
|---|---|---|---|---|---|---|---|
| `hybrid_rerank` → `hybrid_rerank_chunks` | +0.135 [+0.058, +0.231] | 0.60 | yes | yes | 0 | `ndcg@10` +0.025 ok | **yes** |
| `hybrid_rerank_chunks` → `hybrid_rerank_both` | -0.058 [-0.135, +0.000] | 0.68 | no | yes | 0 | `ndcg@10` -0.101 **fails** | no |

**Chosen on validation: `hybrid_rerank_chunks` (mode `hybrid_rerank`); ablations adopted: none.**
