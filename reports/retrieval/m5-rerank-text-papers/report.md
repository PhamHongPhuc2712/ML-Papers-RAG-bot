# Retrieval experiment — m5-rerank-text-papers

Rendered by `eval report` from `reports/retrieval/m5-rerank-text-papers/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | validation |
|---|---|
| Run | `m5-rerank-text-papers-validation-20261012T102408Z` |
| Code | `f7d2f2e46b` + uncommitted `b0c0ff0716` |
| Corpus release | `m2-20260924T095724Z` |
| Dataset | `dfcb643e87b5`, in_domain |
| Queries | 52 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 95% / max 100%; memory 4499–5430 MiB; 0 GPU processes seen; before the run 1% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Validation — 52 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 0.712 [0.596, 0.827] | 0.904 [0.808, 0.981] | 0.576 [0.450, 0.686] | 0.535 [0.409, 0.650] | 0.071 | 0.962 | 0 | 0 |
| `title_chunk_1024` | 0.615 [0.481, 0.750] | 0.904 [0.808, 0.981] | 0.511 [0.386, 0.634] | 0.477 [0.351, 0.606] | 0.062 | 0.962 | 0 | 1 |
| `title_chunk_512` | 0.788 [0.673, 0.885] | 0.904 [0.808, 0.981] | 0.582 [0.476, 0.695] | 0.518 [0.402, 0.647] | 0.079 | 0.962 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 651.5 | 764.3 | 815.7 | 150.1 | 224.4 | 532.7 | 21.147 |
| `title_chunk_1024` | 1065.2 | 1222.5 | 3000.4 | 40.3 | 153.1 | 1107.2 | 22.022 |
| `title_chunk_512` | 671.6 | 713.6 | 719.9 | 40.3 | 53.1 | 652.8 | 16.547 |

nDCG@10 by query set:

| Variant | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|
| `hybrid_rerank` | 0.333 (3) | 0.572 (31) | 0.622 (18) |
| `title_chunk_1024` | 0.333 (3) | 0.518 (31) | 0.529 (18) |
| `title_chunk_512` | 0.333 (3) | 0.604 (31) | 0.584 (18) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `hybrid_rerank` | 0.233 (4) | 0.604 (48) |
| `title_chunk_1024` | 0.250 (4) | 0.533 (48) |
| `title_chunk_512` | 0.333 (4) | 0.602 (48) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `title_chunk_1024` vs `title_chunk_512` | ndcg@10 | 0.582 | 0.511 | -0.071 [-0.137, +0.002] | 6 / 18 / 28 |
| `title_chunk_1024` vs `title_chunk_512` | recall@50 | 0.904 | 0.904 | +0.000 [+0.000, +0.000] | 0 / 0 / 52 |
| `title_chunk_1024` vs `title_chunk_512` | mrr@10 | 0.518 | 0.477 | -0.041 [-0.113, +0.039] | 6 / 18 / 28 |
| `title_chunk_512` vs `hybrid_rerank` | ndcg@10 | 0.576 | 0.582 | +0.006 [-0.079, +0.099] | 12 / 13 / 27 |
| `title_chunk_512` vs `hybrid_rerank` | recall@50 | 0.904 | 0.904 | +0.000 [+0.000, +0.000] | 0 / 0 / 52 |
| `title_chunk_512` vs `hybrid_rerank` | mrr@10 | 0.535 | 0.518 | -0.017 [-0.127, +0.093] | 12 / 13 / 27 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03. A candidate with any degraded query is never chosen (`require_undegraded`).

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Degraded | Promoted |
|---|---|---|---|---|---|---|
| `hybrid_rerank` → `title_chunk_512` | +0.006 [-0.079, +0.099] | 0.71 | no | yes | 0 | no |
| `hybrid_rerank` → `title_chunk_1024` | -0.065 [-0.167, +0.044] | 1.22 | no | yes | 1 | no |

**Chosen on validation: `hybrid_rerank` (mode `hybrid_rerank`); ablations adopted: none.**
