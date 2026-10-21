# Retrieval experiment — rerank-title-abstract-chunk-sections

Rendered by `eval report` from `reports/retrieval/rerank-title-abstract-chunk-sections/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation |
|---|---|---|
| Run | `rerank-title-abstract-chunk-sections-development-20261020T061357Z` | `rerank-title-abstract-chunk-sections-validation-20261020T063118Z` |
| Code | `cb7bbbfa43` + uncommitted `2c2bb689f7` | `cb7bbbfa43` + uncommitted `a2eb693af0` |
| Corpus release | `m2-20260924T095724Z` | `m2-20260924T095724Z` |
| Dataset | `57e034f2272a`, all | `57e034f2272a`, all |
| Queries | 190 | 190 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 98% / max 100%; memory 6501–9339 MiB; 1 GPU processes seen; before the run 0% busy | util p50 99% / max 100%; memory 6541–9361 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 190 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 0.526 [0.458, 0.595] | 0.821 [0.768, 0.874] | 0.374 [0.317, 0.434] | 0.328 [0.269, 0.387] | 0.053 | 0.889 | 0 | 0 |
| `shipped` | 0.779 [0.721, 0.837] | 0.821 [0.768, 0.874] | 0.682 [0.619, 0.738] | 0.651 [0.585, 0.710] | 0.078 | 0.889 | 0 | 0 |
| `title_abstract_chunk` | 0.779 [0.721, 0.832] | 0.821 [0.768, 0.874] | 0.705 [0.646, 0.762] | 0.683 [0.623, 0.743] | 0.078 | 0.889 | 0 | 0 |
| `title_abstract_chunk_1536` | 0.768 [0.711, 0.826] | 0.821 [0.768, 0.874] | 0.694 [0.633, 0.752] | 0.670 [0.609, 0.731] | 0.077 | 0.889 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 562.1 | 621.9 | 1432.0 | 37.0 | 56.8 | 562.5 | 16.449 |
| `shipped` | 1232.7 | 1545.1 | 2750.4 | 279.2 | 381.2 | 1212.8 | 34.045 |
| `title_abstract_chunk` | 1374.6 | 1479.7 | 1639.0 | 41.1 | 68.6 | 1404.8 | 22.635 |
| `title_abstract_chunk_1536` | 1654.9 | 1914.2 | 2098.2 | 60.3 | 83.8 | 1849.3 | 26.663 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | ndcg@10 | 0.682 | 0.374 | -0.308 [-0.370, -0.243] | 16 / 96 / 78 |
| `abstract_reranker` vs `shipped` | recall@50 | 0.821 | 0.821 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `abstract_reranker` vs `shipped` | mrr@10 | 0.651 | 0.328 | -0.324 [-0.394, -0.252] | 16 / 96 / 78 |
| `title_abstract_chunk_1536` vs `shipped` | ndcg@10 | 0.682 | 0.694 | +0.012 [-0.013, +0.038] | 20 / 16 / 154 |
| `title_abstract_chunk_1536` vs `shipped` | recall@50 | 0.821 | 0.821 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `title_abstract_chunk_1536` vs `shipped` | mrr@10 | 0.651 | 0.670 | +0.019 [-0.011, +0.050] | 20 / 16 / 154 |
| `title_abstract_chunk` vs `shipped` | ndcg@10 | 0.682 | 0.705 | +0.024 [-0.001, +0.051] | 24 / 12 / 154 |
| `title_abstract_chunk` vs `shipped` | recall@50 | 0.821 | 0.821 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `title_abstract_chunk` vs `shipped` | mrr@10 | 0.651 | 0.683 | +0.031 [+0.002, +0.063] | 24 / 12 / 154 |

## Validation — 190 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 0.526 [0.458, 0.595] | 0.842 [0.789, 0.895] | 0.350 [0.289, 0.404] | 0.295 [0.236, 0.350] | 0.053 | 0.911 | 0 | 0 |
| `shipped` | 0.774 [0.711, 0.826] | 0.842 [0.789, 0.895] | 0.698 [0.639, 0.756] | 0.673 [0.612, 0.734] | 0.077 | 0.911 | 0 | 0 |
| `title_abstract_chunk` | 0.784 [0.721, 0.842] | 0.842 [0.789, 0.895] | 0.721 [0.660, 0.778] | 0.701 [0.638, 0.759] | 0.078 | 0.911 | 0 | 0 |
| `title_abstract_chunk_1536` | 0.779 [0.721, 0.837] | 0.842 [0.789, 0.895] | 0.719 [0.659, 0.775] | 0.699 [0.637, 0.757] | 0.078 | 0.911 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `abstract_reranker` | 563.3 | 611.4 | 645.3 | 42.8 | 60.7 | 556.6 | 16.293 |
| `shipped` | 1145.5 | 1320.1 | 2037.2 | 91.5 | 108.0 | 1190.1 | 20.618 |
| `title_abstract_chunk` | 1349.2 | 1443.1 | 1479.9 | 41.8 | 56.6 | 1381.9 | 22.37 |
| `title_abstract_chunk_1536` | 1653.2 | 1875.5 | 1996.2 | 42.6 | 61.3 | 1813.2 | 24.508 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | ndcg@10 | 0.698 | 0.350 | -0.348 [-0.408, -0.292] | 10 / 106 / 74 |
| `abstract_reranker` vs `shipped` | recall@50 | 0.842 | 0.842 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `abstract_reranker` vs `shipped` | mrr@10 | 0.673 | 0.295 | -0.378 [-0.441, -0.319] | 10 / 106 / 74 |
| `title_abstract_chunk_1536` vs `shipped` | ndcg@10 | 0.698 | 0.719 | +0.021 [+0.000, +0.043] | 22 / 13 / 155 |
| `title_abstract_chunk_1536` vs `shipped` | recall@50 | 0.842 | 0.842 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `title_abstract_chunk_1536` vs `shipped` | mrr@10 | 0.673 | 0.699 | +0.026 [-0.002, +0.055] | 22 / 13 / 155 |
| `title_abstract_chunk` vs `shipped` | ndcg@10 | 0.698 | 0.721 | +0.023 [+0.000, +0.047] | 24 / 12 / 154 |
| `title_abstract_chunk` vs `shipped` | recall@50 | 0.842 | 0.842 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `title_abstract_chunk` vs `shipped` | mrr@10 | 0.673 | 0.701 | +0.028 [-0.001, +0.058] | 24 / 12 / 154 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03. A candidate with any degraded query is never chosen (`require_undegraded`).

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Degraded | Promoted |
|---|---|---|---|---|---|---|

| Ablation | Change | Difference [interval] | Non-inferior | Applies to the choice | Adopted |
|---|---|---|---|---|---|
| `abstract_reranker` vs `shipped` | the former search page's reranker, title + abstract at a 1.5 s rerank deadline | -0.348 [-0.408, -0.292] | no | yes | no |
| `title_abstract_chunk` vs `shipped` | the reranker reads title + abstract + the chunk that found the paper, 1,024-token pairs | +0.023 [+0.000, +0.047] | yes | yes | **yes** |
| `title_abstract_chunk_1536` vs `shipped` | title + abstract + chunk with 1,536-token pairs, so less of the chunk is cut | +0.021 [+0.000, +0.043] | yes | yes | **yes** |

**Chosen on validation: `shipped` (mode `hybrid_rerank`); ablations adopted: `title_abstract_chunk`, `title_abstract_chunk_1536`.**
