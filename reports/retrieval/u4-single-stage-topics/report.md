# Retrieval experiment — u4-single-stage-topics

Rendered by `eval report` from `reports/retrieval/u4-single-stage-topics/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation |
|---|---|---|
| Run | `u4-single-stage-topics-development-20261016T040046Z` | `u4-single-stage-topics-validation-20261016T033546Z` |
| Code | `112a98c019` + uncommitted `77c56cac2d` | `112a98c019` + uncommitted `9ac3ba9d90` |
| Corpus release | `m2-20260924T095724Z` | `m2-20260924T095724Z` |
| Dataset | `1a24aa151aa5`, all | `1a24aa151aa5`, all |
| Queries | 349 | 349 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 98% / max 100%; memory 3249–4379 MiB; 0 GPU processes seen; before the run 0% busy | util p50 98% / max 100%; memory 3153–4415 MiB; 0 GPU processes seen; before the run 0% busy |
| Locked test | no | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 349 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 0.719 [0.673, 0.762] | 0.782 [0.739, 0.825] | 0.617 [0.572, 0.661] | 0.584 [0.538, 0.630] | 0.072 | 0.862 | 0 | 0 |
| `fusion_only` | 0.625 [0.570, 0.673] | 0.782 [0.739, 0.825] | 0.491 [0.441, 0.535] | 0.449 [0.399, 0.494] | 0.062 | 0.862 | 0 | 0 |
| `shipped` | 0.653 [0.599, 0.705] | 0.782 [0.739, 0.825] | 0.526 [0.481, 0.571] | 0.486 [0.440, 0.533] | 0.065 | 0.862 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 542.7 | 590.2 | 1257.5 | 28.6 | 53.6 | 536.7 | 12.999 |
| `fusion_only` | 39.8 | 51.2 | 803.8 | 25.7 | 50.7 | — | 7.582 |
| `shipped` | 1054.0 | 1206.7 | 1428.2 | 53.9 | 73.1 | 1109.8 | 20.814 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | ndcg@10 | 0.526 | 0.617 | +0.091 [+0.063, +0.124] | 90 / 30 / 229 |
| `abstract_reranker` vs `shipped` | recall@50 | 0.782 | 0.782 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `abstract_reranker` vs `shipped` | mrr@10 | 0.486 | 0.584 | +0.098 [+0.066, +0.134] | 90 / 30 / 229 |
| `fusion_only` vs `shipped` | ndcg@10 | 0.526 | 0.491 | -0.035 [-0.070, -0.004] | 61 / 77 / 211 |
| `fusion_only` vs `shipped` | recall@50 | 0.782 | 0.782 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `fusion_only` vs `shipped` | mrr@10 | 0.486 | 0.449 | -0.037 [-0.075, -0.003] | 61 / 77 / 211 |

## Validation — 349 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 0.765 [0.716, 0.805] | 0.811 [0.768, 0.848] | 0.663 [0.615, 0.706] | 0.630 [0.580, 0.676] | 0.077 | 0.894 | 0 | 0 |
| `fusion_only` | 0.656 [0.605, 0.705] | 0.808 [0.768, 0.848] | 0.512 [0.465, 0.557] | 0.466 [0.417, 0.513] | 0.066 | 0.894 | 0 | 0 |
| `shipped` | 0.705 [0.653, 0.751] | 0.808 [0.768, 0.848] | 0.563 [0.518, 0.610] | 0.518 [0.471, 0.567] | 0.070 | 0.894 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 555.1 | 602.8 | 864.5 | 31.2 | 56.8 | 545.9 | 14.922 |
| `fusion_only` | 50.2 | 60.4 | 376.5 | 29.6 | 59.8 | — | 9.753 |
| `shipped` | 1078.4 | 1273.8 | 2298.9 | 81.9 | 109.7 | 1132.8 | 24.153 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | ndcg@10 | 0.563 | 0.663 | +0.100 [+0.070, +0.130] | 99 / 30 / 220 |
| `abstract_reranker` vs `shipped` | recall@50 | 0.808 | 0.811 | +0.003 [+0.000, +0.009] | 1 / 0 / 348 |
| `abstract_reranker` vs `shipped` | mrr@10 | 0.518 | 0.630 | +0.111 [+0.074, +0.147] | 99 / 30 / 220 |
| `fusion_only` vs `shipped` | ndcg@10 | 0.563 | 0.512 | -0.052 [-0.087, -0.017] | 63 / 95 / 191 |
| `fusion_only` vs `shipped` | recall@50 | 0.808 | 0.808 | +0.000 [+0.000, +0.000] | 0 / 0 / 349 |
| `fusion_only` vs `shipped` | mrr@10 | 0.518 | 0.466 | -0.052 [-0.093, -0.014] | 63 / 95 / 191 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03. A candidate with any degraded query is never chosen (`require_undegraded`).

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Degraded | Promoted |
|---|---|---|---|---|---|---|

| Ablation | Change | Difference [interval] | Non-inferior | Applies to the choice | Adopted |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | the former search page's reranker, title + abstract at a 1.5 s rerank deadline | +0.100 [+0.070, +0.130] | yes | yes | **yes** |
| `fusion_only` vs `shipped` | no reranker, RRF fusion of the BM25 and dense chunk branches | -0.052 [-0.087, -0.017] | no | yes | no |

**Chosen on validation: `shipped` (mode `hybrid_rerank`); ablations adopted: `abstract_reranker`.**
