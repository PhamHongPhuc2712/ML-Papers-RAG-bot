# Retrieval experiment — orb-c256

Rendered by `eval report` from `reports/retrieval/orb-c256/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | retrieval |
|---|---|
| Run | `orb-c256-retrieval-20261009T042830Z` |
| Code | `74e30d6cfe` + uncommitted `2e2d4e647f` |
| Corpus release | `orb-c256` |
| Dataset | `c0ec514fbde7`, all |
| Queries | 1914 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 91% / max 100%; memory 4108–5526 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Retrieval — 1914 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid` | 0.980 [0.971, 0.989] | 0.993 [0.986, 0.998] | 0.959 [0.948, 0.969] | 0.952 [0.941, 0.963] | 0.098 | 0.994 | 0 | 0 |
| `hybrid_rerank` | 0.872 [0.852, 0.890] | 0.993 [0.986, 0.998] | 0.784 [0.762, 0.805] | 0.756 [0.732, 0.780] | 0.085 | 0.994 | 0 | 0 |
| `hybrid_rerank_chunk` | 0.983 [0.976, 0.991] | 0.993 [0.986, 0.998] | 0.958 [0.948, 0.967] | 0.949 [0.938, 0.960] | 0.098 | 0.994 | 0 | 0 |
| `hybrid_rerank_chunk_2500` | 0.983 [0.976, 0.991] | 0.993 [0.986, 0.998] | 0.958 [0.948, 0.967] | 0.949 [0.938, 0.960] | 0.098 | 0.994 | 0 | 0 |
| `hybrid_rerank_title_chunk_2500` | 0.986 [0.978, 0.993] | 0.993 [0.986, 0.998] | 0.965 [0.956, 0.974] | 0.958 [0.948, 0.969] | 0.098 | 0.994 | 0 | 0 |
| `hybrid_rerank_v1` | 0.874 [0.855, 0.891] | 0.991 [0.985, 0.997] | 0.785 [0.763, 0.805] | 0.756 [0.733, 0.779] | 0.085 | 0.994 | 0 | 0 |
| `hybrid_v1` | 0.976 [0.966, 0.985] | 0.991 [0.985, 0.997] | 0.948 [0.936, 0.959] | 0.938 [0.924, 0.951] | 0.098 | 0.994 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid` | 32.2 | 40.8 | 332.9 | 11.7 | 40.3 | — | 1.009 |
| `hybrid_rerank` | 542.3 | 605.2 | 821.6 | 12.4 | 39.0 | 563.9 | 6.109 |
| `hybrid_rerank_chunk` | 379.4 | 403.4 | 617.0 | 12.3 | 38.8 | 359.6 | 4.365 |
| `hybrid_rerank_chunk_2500` | 379.6 | 407.2 | 1228.4 | 12.6 | 40.8 | 360.9 | 4.21 |
| `hybrid_rerank_title_chunk_2500` | 406.8 | 432.4 | 1015.2 | 12.4 | 39.0 | 388.4 | 4.454 |
| `hybrid_rerank_v1` | 540.8 | 606.6 | 832.8 | 12.0 | 42.3 | 563.8 | 6.127 |
| `hybrid_v1` | 28.4 | 38.6 | 650.2 | 10.4 | 38.0 | — | 2.957 |

nDCG@10 by query set:

| Variant | orb_abstractive (n) | orb_extractive (n) |
|---|---|---|
| `hybrid` | 0.967 (893) | 0.946 (1021) |
| `hybrid_rerank` | 0.858 (893) | 0.663 (1021) |
| `hybrid_rerank_chunk` | 0.963 (893) | 0.943 (1021) |
| `hybrid_rerank_chunk_2500` | 0.963 (893) | 0.943 (1021) |
| `hybrid_rerank_title_chunk_2500` | 0.971 (893) | 0.948 (1021) |
| `hybrid_rerank_v1` | 0.858 (893) | 0.662 (1021) |
| `hybrid_v1` | 0.960 (893) | 0.932 (1021) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `hybrid_rerank_chunk_2500` vs `hybrid` | ndcg@10 | 0.959 | 0.958 | -0.002 [-0.010, +0.006] | 55 / 66 / 266 |
| `hybrid_rerank_chunk_2500` vs `hybrid` | recall@50 | 0.993 | 0.993 | +0.000 [+0.000, +0.000] | 0 / 0 / 387 |
| `hybrid_rerank_chunk_2500` vs `hybrid` | mrr@10 | 0.952 | 0.949 | -0.003 [-0.013, +0.006] | 53 / 68 / 266 |
| `hybrid_rerank_chunk` vs `hybrid` | ndcg@10 | 0.959 | 0.958 | -0.002 [-0.010, +0.006] | 55 / 66 / 266 |
| `hybrid_rerank_chunk` vs `hybrid` | recall@50 | 0.993 | 0.993 | +0.000 [+0.000, +0.000] | 0 / 0 / 387 |
| `hybrid_rerank_chunk` vs `hybrid` | mrr@10 | 0.952 | 0.949 | -0.003 [-0.013, +0.006] | 53 / 68 / 266 |
| `hybrid_rerank_title_chunk_2500` vs `hybrid` | ndcg@10 | 0.959 | 0.965 | +0.006 [-0.003, +0.015] | 63 / 53 / 271 |
| `hybrid_rerank_title_chunk_2500` vs `hybrid` | recall@50 | 0.993 | 0.993 | +0.000 [+0.000, +0.000] | 0 / 0 / 387 |
| `hybrid_rerank_title_chunk_2500` vs `hybrid` | mrr@10 | 0.952 | 0.958 | +0.006 [-0.004, +0.016] | 61 / 55 / 271 |
| `hybrid_rerank` vs `hybrid_rerank_v1` | ndcg@10 | 0.785 | 0.784 | -0.001 [-0.004, +0.002] | 67 / 57 / 263 |
| `hybrid_rerank` vs `hybrid_rerank_v1` | recall@50 | 0.991 | 0.993 | +0.001 [-0.001, +0.004] | 5 / 1 / 381 |
| `hybrid_rerank` vs `hybrid_rerank_v1` | mrr@10 | 0.756 | 0.756 | -0.000 [-0.003, +0.003] | 68 / 56 / 263 |
| `hybrid` vs `hybrid_v1` | ndcg@10 | 0.948 | 0.959 | +0.011 [+0.005, +0.019] | 70 / 36 / 281 |
| `hybrid` vs `hybrid_v1` | recall@50 | 0.991 | 0.993 | +0.001 [-0.001, +0.004] | 5 / 1 / 381 |
| `hybrid` vs `hybrid_v1` | mrr@10 | 0.938 | 0.952 | +0.014 [+0.006, +0.023] | 71 / 34 / 282 |
