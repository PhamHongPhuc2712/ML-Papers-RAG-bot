# Retrieval experiment — u4-single-stage-sections

Rendered by `eval report` from `reports/retrieval/u4-single-stage-sections/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation |
|---|---|---|
| Run | `u4-single-stage-sections-development-20261016T034702Z` | `u4-single-stage-sections-validation-20261016T035336Z` |
| Code | `112a98c019` + uncommitted `d0c618923e` | `112a98c019` + uncommitted `6d3761e770` |
| Corpus release | `m2-20260924T095724Z` | `m2-20260924T095724Z` |
| Dataset | `57e034f2272a`, all | `57e034f2272a`, all |
| Queries | 190 | 190 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 98% / max 100%; memory 3153–4187 MiB; 0 GPU processes seen; before the run 0% busy | util p50 98% / max 100%; memory 3249–4241 MiB; 0 GPU processes seen; before the run 0% busy |
| Locked test | no | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 190 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 0.526 [0.458, 0.595] | 0.821 [0.768, 0.874] | 0.374 [0.317, 0.434] | 0.328 [0.270, 0.388] | 0.053 | 0.889 | 0 | 0 |
| `fusion_only` | 0.726 [0.668, 0.789] | 0.821 [0.768, 0.874] | 0.580 [0.524, 0.636] | 0.534 [0.475, 0.594] | 0.073 | 0.889 | 0 | 0 |
| `shipped` | 0.779 [0.721, 0.837] | 0.821 [0.768, 0.874] | 0.682 [0.619, 0.738] | 0.651 [0.585, 0.710] | 0.078 | 0.889 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 559.5 | 614.7 | 1191.2 | 34.2 | 51.8 | 560.5 | 12.94 |
| `fusion_only` | 39.6 | 49.2 | 52.9 | 30.8 | 48.8 | — | 7.788 |
| `shipped` | 1121.2 | 1303.9 | 1592.3 | 74.1 | 88.4 | 1195.2 | 24.251 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | ndcg@10 | 0.682 | 0.374 | -0.308 [-0.370, -0.243] | 16 / 95 / 79 |
| `abstract_reranker` vs `shipped` | recall@50 | 0.821 | 0.821 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `abstract_reranker` vs `shipped` | mrr@10 | 0.651 | 0.328 | -0.324 [-0.394, -0.252] | 16 / 95 / 79 |
| `fusion_only` vs `shipped` | ndcg@10 | 0.682 | 0.580 | -0.101 [-0.153, -0.050] | 26 / 56 / 108 |
| `fusion_only` vs `shipped` | recall@50 | 0.821 | 0.821 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `fusion_only` vs `shipped` | mrr@10 | 0.651 | 0.534 | -0.117 [-0.178, -0.058] | 26 / 56 / 108 |

## Validation — 190 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 0.526 [0.458, 0.595] | 0.842 [0.789, 0.895] | 0.350 [0.289, 0.404] | 0.295 [0.237, 0.351] | 0.053 | 0.911 | 0 | 0 |
| `fusion_only` | 0.747 [0.684, 0.805] | 0.842 [0.789, 0.895] | 0.605 [0.542, 0.664] | 0.560 [0.495, 0.622] | 0.075 | 0.911 | 0 | 0 |
| `shipped` | 0.774 [0.711, 0.826] | 0.842 [0.789, 0.895] | 0.698 [0.639, 0.756] | 0.673 [0.612, 0.734] | 0.077 | 0.911 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 550.7 | 600.6 | 716.4 | 37.6 | 52.5 | 549.5 | 13.231 |
| `fusion_only` | 39.9 | 49.1 | 57.2 | 29.9 | 48.6 | — | 8.243 |
| `shipped` | 1114.9 | 1283.5 | 1436.4 | 67.1 | 80.0 | 1182.0 | 22.274 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | ndcg@10 | 0.698 | 0.350 | -0.348 [-0.408, -0.292] | 10 / 106 / 74 |
| `abstract_reranker` vs `shipped` | recall@50 | 0.842 | 0.842 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `abstract_reranker` vs `shipped` | mrr@10 | 0.673 | 0.295 | -0.378 [-0.441, -0.319] | 10 / 106 / 74 |
| `fusion_only` vs `shipped` | ndcg@10 | 0.698 | 0.605 | -0.093 [-0.142, -0.044] | 22 / 54 / 114 |
| `fusion_only` vs `shipped` | recall@50 | 0.842 | 0.842 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `fusion_only` vs `shipped` | mrr@10 | 0.673 | 0.560 | -0.113 [-0.170, -0.058] | 22 / 54 / 114 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03. A candidate with any degraded query is never chosen (`require_undegraded`).

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Degraded | Promoted |
|---|---|---|---|---|---|---|

| Ablation | Change | Difference [interval] | Non-inferior | Applies to the choice | Adopted |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | the former search page's reranker, title + abstract at a 1.5 s rerank deadline | -0.348 [-0.408, -0.292] | no | yes | no |
| `fusion_only` vs `shipped` | no reranker, RRF fusion of the BM25 and dense chunk branches | -0.093 [-0.142, -0.044] | no | yes | no |

**Chosen on validation: `shipped` (mode `hybrid_rerank`); ablations adopted: none.**
