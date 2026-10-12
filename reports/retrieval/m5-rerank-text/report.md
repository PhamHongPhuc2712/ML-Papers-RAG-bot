# Retrieval experiment — m5-rerank-text

Rendered by `eval report` from `reports/retrieval/m5-rerank-text/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation |
|---|---|---|
| Run | `m5-rerank-text-development-20261012T100315Z` | `m5-rerank-text-validation-20261012T101403Z` |
| Code | `f7d2f2e46b` | `f7d2f2e46b` + uncommitted `a515d28ee5` |
| Corpus release | `m2-20260924T095724Z` | `m2-20260924T095724Z` |
| Dataset | `57e034f2272a`, all | `57e034f2272a`, all |
| Queries | 190 | 190 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 98% / max 100%; memory 4498–6539 MiB; 0 GPU processes seen; before the run 0% busy | util p50 98% / max 100%; memory 4490–6457 MiB; 0 GPU processes seen; before the run 0% busy |
| Locked test | no | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 190 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 0.526 [0.458, 0.595] | 0.821 [0.768, 0.868] | 0.374 [0.317, 0.434] | 0.328 [0.270, 0.388] | 0.053 | 0.884 | 1 | 4 |
| `title_chunk_1024` | 0.779 [0.721, 0.837] | 0.821 [0.768, 0.874] | 0.682 [0.619, 0.738] | 0.651 [0.585, 0.710] | 0.078 | 0.889 | 0 | 0 |
| `title_chunk_512` | 0.737 [0.674, 0.789] | 0.821 [0.768, 0.874] | 0.654 [0.596, 0.709] | 0.627 [0.565, 0.687] | 0.074 | 0.889 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 637.2 | 1182.1 | 2400.3 | 266.5 | 571.5 | 564.6 | 33.535 |
| `title_chunk_1024` | 1076.7 | 1249.5 | 1345.0 | 70.2 | 94.4 | 1158.0 | 20.695 |
| `title_chunk_512` | 691.2 | 734.9 | 1323.1 | 64.2 | 82.8 | 655.7 | 15.769 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `title_chunk_1024` vs `title_chunk_512` | ndcg@10 | 0.654 | 0.682 | +0.028 [-0.007, +0.064] | 22 / 29 / 139 |
| `title_chunk_1024` vs `title_chunk_512` | recall@50 | 0.821 | 0.821 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `title_chunk_1024` vs `title_chunk_512` | mrr@10 | 0.627 | 0.651 | +0.024 [-0.016, +0.064] | 22 / 29 / 139 |
| `title_chunk_512` vs `hybrid_rerank` | ndcg@10 | 0.374 | 0.654 | +0.280 [+0.212, +0.348] | 91 / 22 / 77 |
| `title_chunk_512` vs `hybrid_rerank` | recall@50 | 0.821 | 0.821 | +0.000 [-0.016, +0.011] | 1 / 1 / 188 |
| `title_chunk_512` vs `hybrid_rerank` | mrr@10 | 0.328 | 0.627 | +0.300 [+0.224, +0.374] | 91 / 22 / 77 |

## Validation — 190 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 0.526 [0.458, 0.595] | 0.842 [0.789, 0.895] | 0.350 [0.289, 0.404] | 0.295 [0.237, 0.351] | 0.053 | 0.911 | 1 | 1 |
| `title_chunk_1024` | 0.774 [0.711, 0.826] | 0.842 [0.789, 0.895] | 0.698 [0.639, 0.756] | 0.673 [0.612, 0.734] | 0.077 | 0.911 | 0 | 0 |
| `title_chunk_512` | 0.768 [0.705, 0.826] | 0.842 [0.789, 0.895] | 0.664 [0.604, 0.722] | 0.631 [0.567, 0.692] | 0.077 | 0.911 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 611.7 | 725.3 | 1934.1 | 104.9 | 178.9 | 558.3 | 20.729 |
| `title_chunk_1024` | 1117.7 | 1284.5 | 1370.9 | 35.1 | 56.0 | 1190.3 | 19.5 |
| `title_chunk_512` | 686.7 | 711.2 | 725.5 | 30.9 | 50.4 | 651.6 | 15.402 |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `title_chunk_1024` vs `title_chunk_512` | ndcg@10 | 0.664 | 0.698 | +0.034 [-0.000, +0.069] | 25 / 22 / 143 |
| `title_chunk_1024` vs `title_chunk_512` | recall@50 | 0.842 | 0.842 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `title_chunk_1024` vs `title_chunk_512` | mrr@10 | 0.631 | 0.673 | +0.042 [+0.003, +0.082] | 25 / 22 / 143 |
| `title_chunk_512` vs `hybrid_rerank` | ndcg@10 | 0.350 | 0.664 | +0.314 [+0.253, +0.376] | 101 / 17 / 72 |
| `title_chunk_512` vs `hybrid_rerank` | recall@50 | 0.842 | 0.842 | +0.000 [+0.000, +0.000] | 0 / 0 / 190 |
| `title_chunk_512` vs `hybrid_rerank` | mrr@10 | 0.295 | 0.631 | +0.336 [+0.273, +0.400] | 101 / 17 / 72 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03. A candidate with any degraded query is never chosen (`require_undegraded`).

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Degraded | Promoted |
|---|---|---|---|---|---|---|
| `hybrid_rerank` → `title_chunk_512` | +0.314 [+0.253, +0.376] | 0.71 | yes | yes | 0 | no |
| `hybrid_rerank` → `title_chunk_1024` | +0.348 [+0.292, +0.408] | 1.28 | yes | yes | 0 | no |

**Chosen on validation: `hybrid_rerank` (mode `hybrid_rerank`); ablations adopted: none.**
