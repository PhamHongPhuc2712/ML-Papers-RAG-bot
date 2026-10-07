# Retrieval experiment — orb-text

Rendered by `eval report` from `reports/retrieval/orb-text/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | retrieval |
|---|---|
| Run | `orb-text-retrieval-20261007T110552Z` |
| Code | `f5c40b7386` + uncommitted `7cc22de1e8` |
| Corpus release | `orb-v1` |
| Dataset | `c0ec514fbde7`, all |
| Queries | 1914 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 96% / max 100%; memory 3689–4947 MiB; 0 GPU processes seen; before the run 0% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Retrieval — 1914 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.976 [0.967, 0.986] | 0.987 [0.979, 0.995] | 0.942 [0.929, 0.954] | 0.931 [0.916, 0.945] | 0.097 | 0.989 | 0 | 0 |
| `bm25_papers` | 0.777 [0.753, 0.802] | 0.853 [0.833, 0.871] | 0.693 [0.669, 0.718] | 0.666 [0.641, 0.693] | 0.075 | 0.864 | 0 | 0 |
| `dense` | 0.963 [0.952, 0.974] | 0.981 [0.973, 0.989] | 0.917 [0.903, 0.930] | 0.902 [0.887, 0.917] | 0.096 | 0.987 | 0 | 0 |
| `dense_papers` | 0.762 [0.738, 0.788] | 0.854 [0.833, 0.874] | 0.677 [0.653, 0.705] | 0.650 [0.624, 0.679] | 0.073 | 0.872 | 0 | 0 |
| `hybrid` | 0.976 [0.966, 0.985] | 0.991 [0.985, 0.997] | 0.948 [0.936, 0.959] | 0.938 [0.924, 0.951] | 0.098 | 0.994 | 0 | 0 |
| `hybrid_papers` | 0.809 [0.787, 0.831] | 0.882 [0.863, 0.900] | 0.728 [0.706, 0.753] | 0.703 [0.679, 0.728] | 0.078 | 0.923 | 0 | 0 |
| `hybrid_rerank` | 0.874 [0.855, 0.891] | 0.991 [0.985, 0.997] | 0.785 [0.763, 0.805] | 0.756 [0.733, 0.779] | 0.085 | 0.994 | 0 | 0 |
| `hybrid_rerank_papers` | 0.838 [0.816, 0.858] | 0.882 [0.863, 0.900] | 0.762 [0.740, 0.785] | 0.738 [0.714, 0.762] | 0.081 | 0.923 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25` | 5.3 | 6.9 | 222.1 | 6.7 | — | — | 0.473 |
| `bm25_papers` | 2.5 | 3.4 | 7.1 | 3.2 | — | — | 0.549 |
| `dense` | 24.4 | 32.0 | 255.0 | — | 31.7 | — | 0.839 |
| `dense_papers` | 18.9 | 29.9 | 197.3 | — | 29.7 | — | 0.559 |
| `hybrid` | 27.4 | 35.5 | 301.1 | 9.3 | 35.0 | — | 0.664 |
| `hybrid_papers` | 21.1 | 30.9 | 44.3 | 4.7 | 30.4 | — | 0.284 |
| `hybrid_rerank` | 534.7 | 594.4 | 836.4 | 11.1 | 35.9 | 558.6 | 5.69 |
| `hybrid_rerank_papers` | 531.3 | 612.3 | 755.6 | 6.3 | 31.1 | 583.4 | 5.646 |

nDCG@10 by query set:

| Variant | orb_abstractive (n) | orb_extractive (n) |
|---|---|---|
| `bm25` | 0.948 (893) | 0.932 (1021) |
| `bm25_papers` | 0.785 (893) | 0.552 (1021) |
| `dense` | 0.937 (893) | 0.886 (1021) |
| `dense_papers` | 0.778 (893) | 0.517 (1021) |
| `hybrid` | 0.960 (893) | 0.932 (1021) |
| `hybrid_papers` | 0.812 (893) | 0.596 (1021) |
| `hybrid_rerank` | 0.858 (893) | 0.662 (1021) |
| `hybrid_rerank_papers` | 0.848 (893) | 0.627 (1021) |
