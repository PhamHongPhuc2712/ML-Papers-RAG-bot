# Retrieval experiment — e3-litsearch

Rendered by `eval report` from `reports/retrieval/e3-final/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | validation |
|---|---|
| Run | `e3-litsearch-validation-20261006T064454Z` |
| Code | `8654f22043` + uncommitted `5fbf5cd333` |
| Corpus release | `litsearch-v1` |
| Dataset | `dfcb643e87b5`, all |
| Queries | 120 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 91% / max 100%; memory 3635–4653 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Validation — 120 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.499 [0.410, 0.588] | 0.649 [0.564, 0.725] | 0.360 [0.288, 0.431] | 0.319 [0.249, 0.390] | 0.051 | 0.690 | 0 | 0 |
| `dense` | 0.526 [0.435, 0.610] | 0.625 [0.537, 0.700] | 0.385 [0.306, 0.457] | 0.341 [0.266, 0.412] | 0.054 | 0.692 | 0 | 0 |
| `hybrid` | 0.536 [0.449, 0.621] | 0.696 [0.613, 0.771] | 0.415 [0.339, 0.492] | 0.377 [0.303, 0.455] | 0.055 | 0.787 | 0 | 0 |
| `hybrid_rerank` | 0.592 [0.504, 0.671] | 0.696 [0.613, 0.771] | 0.469 [0.390, 0.545] | 0.434 [0.354, 0.515] | 0.060 | 0.787 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25` | 4.4 | 7.3 | 11.8 | 7.0 | — | — | 2.661 |
| `dense` | 26.7 | 65.0 | 159.2 | — | 64.7 | — | 4.248 |
| `hybrid` | 25.9 | 34.2 | 40.1 | 6.3 | 33.8 | — | 0.493 |
| `hybrid_rerank` | 502.2 | 814.5 | 865.3 | 8.5 | 35.2 | 782.4 | 5.623 |

nDCG@10 by query set:

| Variant | inline_acl (n) | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|---|
| `bm25` | 0.292 (20) | 0.262 (51) | 0.421 (31) | 0.606 (18) |
| `dense` | 0.286 (20) | 0.358 (51) | 0.334 (31) | 0.664 (18) |
| `hybrid` | 0.318 (20) | 0.351 (51) | 0.417 (31) | 0.698 (18) |
| `hybrid_rerank` | 0.347 (20) | 0.436 (51) | 0.502 (31) | 0.642 (18) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `bm25` | 0.229 (23) | 0.391 (97) |
| `dense` | 0.288 (23) | 0.408 (97) |
| `hybrid` | 0.304 (23) | 0.441 (97) |
| `hybrid_rerank` | 0.296 (23) | 0.510 (97) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `dense` vs `bm25` | ndcg@10 | 0.360 | 0.385 | +0.026 [-0.047, +0.089] | 30 / 29 / 61 |
| `dense` vs `bm25` | recall@50 | 0.649 | 0.625 | -0.024 [-0.099, +0.046] | 10 / 14 / 96 |
| `dense` vs `bm25` | mrr@10 | 0.319 | 0.341 | +0.022 [-0.050, +0.089] | 30 / 29 / 61 |
| `hybrid_rerank` vs `hybrid` | ndcg@10 | 0.415 | 0.469 | +0.054 [+0.007, +0.101] | 29 / 16 / 75 |
| `hybrid_rerank` vs `hybrid` | recall@50 | 0.696 | 0.696 | +0.000 [+0.000, +0.000] | 0 / 0 / 120 |
| `hybrid_rerank` vs `hybrid` | mrr@10 | 0.377 | 0.434 | +0.056 [+0.006, +0.112] | 29 / 16 / 75 |
| `hybrid` vs `dense` | ndcg@10 | 0.385 | 0.415 | +0.029 [-0.012, +0.069] | 30 / 15 / 75 |
| `hybrid` vs `dense` | recall@50 | 0.625 | 0.696 | +0.071 [+0.021, +0.121] | 11 / 1 / 108 |
| `hybrid` vs `dense` | mrr@10 | 0.341 | 0.377 | +0.037 [-0.011, +0.079] | 31 / 14 / 75 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03.

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Promoted |
|---|---|---|---|---|---|
| `bm25` → `dense` | +0.026 [-0.047, +0.089] | 0.07 | no | yes | no |
| `bm25` → `hybrid` | +0.055 [+0.001, +0.104] | 0.03 | yes | yes | **yes** |
| `hybrid` → `hybrid_rerank` | +0.054 [+0.007, +0.101] | 0.81 | yes | yes | **yes** |

**Chosen on validation: `hybrid_rerank` (mode `hybrid_rerank`); ablations adopted: none.**
