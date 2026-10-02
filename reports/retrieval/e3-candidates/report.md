# Retrieval experiment — e3-candidates

Rendered by `eval report` from `reports/retrieval/e3-candidates/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development |
|---|---|
| Run | `e3-candidates-development-20261002T084151Z` |
| Code | `1467e20baa` |
| Corpus release | `litsearch-v1` |
| Dataset | `dfcb643e87b5`, all |
| Queries | 359 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 33% / max 54%; memory 2110–2182 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 359 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid` | 0.597 [0.545, 0.644] | 0.784 [0.743, 0.826] | 0.474 [0.427, 0.517] | 0.441 [0.392, 0.484] | 0.063 | 0.860 | 0 | 0 |
| `hybrid_pool200` | 0.605 [0.553, 0.653] | 0.784 [0.741, 0.826] | 0.477 [0.430, 0.520] | 0.443 [0.394, 0.486] | 0.064 | 0.885 | 0 | 0 |
| `hybrid_pool300` | 0.609 [0.558, 0.657] | 0.799 [0.759, 0.837] | 0.481 [0.433, 0.522] | 0.445 [0.396, 0.486] | 0.064 | 0.910 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid` | 29.8 | 57.2 | 141.3 | 11.6 | 56.7 | — | 5.752 |
| `hybrid_pool200` | 28.3 | 39.8 | 55.8 | 8.0 | 39.2 | — | 0.604 |
| `hybrid_pool300` | 30.4 | 45.7 | 181.8 | 10.1 | 44.7 | — | 0.491 |

nDCG@10 by query set:

| Variant | inline_acl (n) | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|---|
| `hybrid` | 0.414 (59) | 0.409 (152) | 0.568 (93) | 0.562 (55) |
| `hybrid_pool200` | 0.414 (59) | 0.416 (152) | 0.569 (93) | 0.561 (55) |
| `hybrid_pool300` | 0.417 (59) | 0.416 (152) | 0.576 (93) | 0.566 (55) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `hybrid` | 0.320 (103) | 0.537 (256) |
| `hybrid_pool200` | 0.320 (103) | 0.541 (256) |
| `hybrid_pool300` | 0.325 (103) | 0.543 (256) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `hybrid_pool200` vs `hybrid` | ndcg@10 | 0.474 | 0.477 | +0.003 [-0.002, +0.008] | 8 / 5 / 346 |
| `hybrid_pool200` vs `hybrid` | recall@50 | 0.784 | 0.784 | +0.000 [-0.019, +0.017] | 6 / 6 / 347 |
| `hybrid_pool200` vs `hybrid` | mrr@10 | 0.441 | 0.443 | +0.002 [-0.001, +0.004] | 8 / 5 / 346 |
| `hybrid_pool300` vs `hybrid` | ndcg@10 | 0.474 | 0.481 | +0.006 [+0.000, +0.013] | 11 / 5 / 343 |
| `hybrid_pool300` vs `hybrid` | recall@50 | 0.784 | 0.799 | +0.015 [-0.006, +0.036] | 11 / 5 / 343 |
| `hybrid_pool300` vs `hybrid` | mrr@10 | 0.441 | 0.445 | +0.004 [+0.000, +0.010] | 10 / 5 / 344 |

## Analysis

Written 2026-10-02. A development-only diagnostic for P2.6 step 2: there is no decision
block, and nothing here chooses. `reports/m2-retrieval-gaps.md` (step 2) has the reading,
and `gaps.md` beside this file has the slices.

- The 100-per-branch row reproduces E3's recorded hybrid run, so the system measured is the
  shipped one.
- A bigger pool finds gold the fusion then buries past rank 100. Never-pooled gold falls
  from 14.0% to 9.0% at 300 per branch, but gold within the fused top 100 rises only from
  82.8% to 84.3%.
- The 9–14% never pooled sits mostly in broad and citation-derived queries, out of reach of
  a larger pool. That is a first-stage model problem (P2.6 step 4).
