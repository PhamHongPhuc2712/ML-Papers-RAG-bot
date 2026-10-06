# Retrieval experiment — m2-release

Rendered by `eval report` from `reports/retrieval/m2-release/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | test |
|---|---|
| Run | `m2-release-test-20261006T091458Z` |
| Code | `912497474e` + uncommitted `4e606bb000` |
| Corpus release | `m2-20260924T095724Z` |
| Dataset | `dfcb643e87b5`, in_domain |
| Queries | 48 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 24% / max 99%; memory 3813–4388 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | yes |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Test — 48 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.688 [0.562, 0.833] | 0.792 [0.667, 0.896] | 0.494 [0.381, 0.607] | 0.433 [0.317, 0.546] | 0.069 | 0.833 | 0 | 0 |
| `dense` | 0.458 [0.312, 0.604] | 0.604 [0.458, 0.750] | 0.343 [0.218, 0.466] | 0.307 [0.189, 0.428] | 0.061 | 0.625 | 12 | 0 |
| `hybrid` | 0.771 [0.646, 0.876] | 0.875 [0.771, 0.958] | 0.630 [0.515, 0.746] | 0.587 [0.463, 0.713] | 0.077 | 0.938 | 0 | 0 |
| `hybrid_rerank` | 0.750 [0.625, 0.875] | 0.875 [0.771, 0.958] | 0.582 [0.467, 0.704] | 0.528 [0.409, 0.660] | 0.075 | 0.938 | 0 | 0 |
| `hybrid_rerank_papers` | 0.708 [0.583, 0.833] | 0.792 [0.667, 0.896] | 0.558 [0.445, 0.683] | 0.510 [0.393, 0.641] | 0.072 | 0.875 | 1 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25` | 163.6 | 259.4 | 403.8 | 259.1 | — | — | 20.101 |
| `dense` | 474.5 | 1000.7 | 1726.9 | — | 887.7 | — | 23.245 |
| `hybrid` | 60.5 | 101.2 | 202.0 | 94.7 | 100.7 | — | 16.36 |
| `hybrid_rerank` | 623.4 | 736.7 | 1477.2 | 97.4 | 104.3 | 538.6 | 18.282 |
| `hybrid_rerank_papers` | 591.8 | 695.7 | 1006.9 | 23.5 | 118.6 | 562.0 | 10.235 |

nDCG@10 by query set:

| Variant | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|
| `bm25` | 0.000 (1) | 0.483 (31) | 0.546 (16) |
| `dense` | 0.000 (1) | 0.263 (31) | 0.521 (16) |
| `hybrid` | 0.000 (1) | 0.636 (31) | 0.657 (16) |
| `hybrid_rerank` | 0.631 (1) | 0.580 (31) | 0.582 (16) |
| `hybrid_rerank_papers` | 0.631 (1) | 0.558 (31) | 0.553 (16) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `bm25` | 0.382 (7) | 0.513 (41) |
| `dense` | 0.194 (7) | 0.369 (41) |
| `hybrid` | 0.590 (7) | 0.637 (41) |
| `hybrid_rerank` | 0.407 (7) | 0.612 (41) |
| `hybrid_rerank_papers` | 0.401 (7) | 0.585 (41) |
