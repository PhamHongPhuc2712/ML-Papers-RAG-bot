# Retrieval experiment — e3-first-stage

Rendered by `eval report` from `reports/retrieval/e3-first-stage/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development |
|---|---|
| Run | `e3-first-stage-development-20261006T060219Z` |
| Code | `eaf4bb450c` + uncommitted `bcb03c2596` |
| Corpus release | `litsearch-v1` |
| Dataset | `dfcb643e87b5`, all |
| Queries | 359 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 97% / max 100%; memory 3627–5121 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 359 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25_deep1000` | 0.547 [0.495, 0.597] | 0.678 [0.630, 0.726] | 0.411 [0.368, 0.457] | 0.373 [0.330, 0.418] | 0.058 | 0.884 | 0 | 0 |
| `dense_deep1000` | 0.579 [0.526, 0.628] | 0.727 [0.681, 0.773] | 0.440 [0.393, 0.481] | 0.401 [0.353, 0.443] | 0.060 | 0.899 | 0 | 0 |
| `hybrid_rerank` | 0.694 [0.649, 0.741] | 0.784 [0.743, 0.826] | 0.556 [0.511, 0.600] | 0.517 [0.468, 0.563] | 0.072 | 0.860 | 0 | 0 |
| `hybrid_rerank_pool300` | 0.680 [0.635, 0.727] | 0.799 [0.759, 0.837] | 0.550 [0.504, 0.595] | 0.513 [0.464, 0.559] | 0.071 | 0.910 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25_deep1000` | 8.7 | 10.8 | 16.0 | 10.0 | — | — | 0.289 |
| `dense_deep1000` | 33.7 | 60.9 | 136.9 | — | 59.9 | — | 0.546 |
| `hybrid_rerank` | 494.5 | 755.2 | 1240.8 | 14.3 | 57.5 | 727.0 | 18.212 |
| `hybrid_rerank_pool300` | 507.3 | 757.6 | 889.1 | 9.8 | 40.3 | 731.4 | 5.621 |

nDCG@10 by query set:

| Variant | inline_acl (n) | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|---|
| `bm25_deep1000` | 0.328 (59) | 0.301 (152) | 0.544 (93) | 0.581 (55) |
| `dense_deep1000` | 0.396 (59) | 0.436 (152) | 0.468 (93) | 0.448 (55) |
| `hybrid_rerank` | 0.474 (59) | 0.500 (152) | 0.637 (93) | 0.664 (55) |
| `hybrid_rerank_pool300` | 0.475 (59) | 0.489 (152) | 0.631 (93) | 0.666 (55) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `bm25_deep1000` | 0.271 (103) | 0.468 (256) |
| `dense_deep1000` | 0.341 (103) | 0.479 (256) |
| `hybrid_rerank` | 0.422 (103) | 0.611 (256) |
| `hybrid_rerank_pool300` | 0.418 (103) | 0.604 (256) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `hybrid_rerank_pool300` vs `hybrid_rerank` | ndcg@10 | 0.556 | 0.550 | -0.006 [-0.016, +0.003] | 8 / 16 / 335 |
| `hybrid_rerank_pool300` vs `hybrid_rerank` | recall@50 | 0.784 | 0.799 | +0.015 [-0.006, +0.036] | 11 / 5 / 343 |
| `hybrid_rerank_pool300` vs `hybrid_rerank` | mrr@10 | 0.517 | 0.513 | -0.004 [-0.012, +0.002] | 8 / 15 / 336 |
