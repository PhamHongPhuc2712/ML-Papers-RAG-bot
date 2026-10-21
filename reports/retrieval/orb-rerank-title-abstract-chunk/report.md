# Retrieval experiment — orb-rerank-title-abstract-chunk

Rendered by `eval report` from `reports/retrieval/orb-rerank-title-abstract-chunk/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | retrieval |
|---|---|
| Run | `orb-rerank-title-abstract-chunk-retrieval-20261021T033107Z` |
| Code | `cb7bbbfa43` + uncommitted `9d2b38644a` |
| Corpus release | `orb-v1` |
| Dataset | `c0ec514fbde7`, all |
| Queries | 1914 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 99% / max 100%; memory 3407–4356 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Retrieval — 1914 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid` | 0.976 [0.966, 0.985] | 0.991 [0.985, 0.997] | 0.948 [0.936, 0.959] | 0.938 [0.924, 0.951] | 0.098 | 0.994 | 0 | 0 |
| `shipped` | 0.986 [0.979, 0.992] | 0.991 [0.985, 0.997] | 0.968 [0.959, 0.976] | 0.962 [0.952, 0.971] | 0.098 | 0.994 | 0 | 0 |
| `title_abstract_chunk` | 0.983 [0.975, 0.990] | 0.991 [0.985, 0.997] | 0.966 [0.957, 0.975] | 0.961 [0.951, 0.971] | 0.098 | 0.994 | 0 | 1 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid` | 27.7 | 40.1 | 785.3 | 10.6 | 39.6 | — | 1.235 |
| `shipped` | 1218.3 | 1325.5 | 2049.5 | 11.7 | 37.3 | 1272.4 | 12.916 |
| `title_abstract_chunk` | 1403.7 | 1459.8 | 2384.2 | 11.6 | 36.3 | 1418.7 | 14.396 |

nDCG@10 by query set:

| Variant | orb_abstractive (n) | orb_extractive (n) |
|---|---|---|
| `hybrid` | 0.960 (893) | 0.932 (1021) |
| `shipped` | 0.974 (893) | 0.954 (1021) |
| `title_abstract_chunk` | 0.974 (893) | 0.946 (1021) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `title_abstract_chunk` vs `shipped` | ndcg@10 | 0.968 | 0.966 | -0.002 [-0.006, +0.003] | 31 / 40 / 316 |
| `title_abstract_chunk` vs `shipped` | recall@50 | 0.991 | 0.991 | +0.000 [+0.000, +0.000] | 0 / 0 / 387 |
| `title_abstract_chunk` vs `shipped` | mrr@10 | 0.962 | 0.961 | -0.001 [-0.007, +0.005] | 31 / 40 / 316 |
