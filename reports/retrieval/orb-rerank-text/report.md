# Retrieval experiment — orb-rerank-text

Rendered by `eval report` from `reports/retrieval/orb-rerank-text/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | retrieval |
|---|---|
| Run | `orb-rerank-text-retrieval-20261008T061622Z` |
| Code | `52b0f1e661` + uncommitted `b62277507d` |
| Corpus release | `orb-v1` |
| Dataset | `c0ec514fbde7`, all |
| Queries | 1914 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 99% / max 100%; memory 4278–9978 MiB; 2 GPU processes seen; before the run 0% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Retrieval — 1914 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid` | 0.976 [0.966, 0.985] | 0.991 [0.985, 0.997] | 0.947 [0.935, 0.959] | 0.938 [0.924, 0.951] | 0.098 | 0.993 | 1 | 1 |
| `hybrid_rerank` | 0.874 [0.855, 0.891] | 0.991 [0.985, 0.997] | 0.785 [0.763, 0.805] | 0.756 [0.733, 0.779] | 0.085 | 0.994 | 0 | 0 |
| `hybrid_rerank_chunk` | 0.985 [0.978, 0.992] | 0.991 [0.985, 0.997] | 0.965 [0.954, 0.974] | 0.958 [0.946, 0.969] | 0.098 | 0.994 | 0 | 10 |
| `hybrid_rerank_chunk_2500` | 0.985 [0.978, 0.992] | 0.991 [0.985, 0.997] | 0.965 [0.954, 0.974] | 0.958 [0.946, 0.969] | 0.098 | 0.994 | 0 | 0 |
| `hybrid_rerank_title_chunk` | 0.984 [0.977, 0.991] | 0.991 [0.985, 0.997] | 0.966 [0.958, 0.975] | 0.961 [0.951, 0.970] | 0.098 | 0.994 | 0 | 379 |
| `hybrid_rerank_title_chunk_2500` | 0.986 [0.979, 0.992] | 0.991 [0.985, 0.997] | 0.968 [0.959, 0.976] | 0.962 [0.952, 0.971] | 0.098 | 0.994 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid` | 39.8 | 70.6 | 1116.9 | 18.1 | 69.2 | — | 2.702 |
| `hybrid_rerank` | 565.6 | 657.4 | 1487.0 | 14.0 | 55.2 | 603.0 | 6.836 |
| `hybrid_rerank_chunk` | 1259.9 | 1419.2 | 2518.3 | 16.4 | 54.9 | 1334.6 | 14.151 |
| `hybrid_rerank_chunk_2500` | 1234.8 | 1408.7 | 2322.8 | 12.4 | 45.2 | 1349.3 | 12.731 |
| `hybrid_rerank_title_chunk` | 1256.5 | 2139.7 | 2682.4 | 13.7 | 627.7 | 1301.2 | 13.01 |
| `hybrid_rerank_title_chunk_2500` | 1244.2 | 1558.2 | 2260.9 | 11.1 | 42.1 | 1513.8 | 12.901 |

nDCG@10 by query set:

| Variant | orb_abstractive (n) | orb_extractive (n) |
|---|---|---|
| `hybrid` | 0.959 (893) | 0.932 (1021) |
| `hybrid_rerank` | 0.858 (893) | 0.662 (1021) |
| `hybrid_rerank_chunk` | 0.970 (893) | 0.953 (1021) |
| `hybrid_rerank_chunk_2500` | 0.970 (893) | 0.953 (1021) |
| `hybrid_rerank_title_chunk` | 0.974 (893) | 0.951 (1021) |
| `hybrid_rerank_title_chunk_2500` | 0.974 (893) | 0.954 (1021) |
