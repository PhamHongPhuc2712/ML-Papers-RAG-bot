# Retrieval experiment — e3-rerank-depth

Rendered by `eval report` from `reports/retrieval/e3-rerank-depth/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation |
|---|---|---|
| Run | `e3-rerank-depth-development-20261002T084328Z` | `e3-rerank-depth-validation-20261002T085409Z` |
| Code | `1467e20baa` + uncommitted `c199546c27` | `1467e20baa` + uncommitted `22e5fe6ee1` |
| Corpus release | `litsearch-v1` | `litsearch-v1` |
| Dataset | `dfcb643e87b5`, all | `dfcb643e87b5`, all |
| Queries | 359 | 120 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 97% / max 100%; memory 3198–4688 MiB; 1 GPU processes seen; before the run 0% busy | util p50 97% / max 100%; memory 3198–4261 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 359 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 0.694 [0.649, 0.741] | 0.784 [0.743, 0.826] | 0.556 [0.511, 0.600] | 0.517 [0.468, 0.563] | 0.072 | 0.860 | 0 | 0 |
| `hybrid_rerank_depth100` | 0.682 [0.635, 0.728] | 0.807 [0.766, 0.845] | 0.550 [0.505, 0.594] | 0.513 [0.465, 0.559] | 0.071 | 0.860 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 497.1 | 749.6 | 1223.1 | 13.5 | 48.0 | 718.8 | 16.69 |
| `hybrid_rerank_depth100` | 918.4 | 1219.5 | 1479.3 | 8.9 | 40.8 | 1192.7 | 8.925 |

nDCG@10 by query set:

| Variant | inline_acl (n) | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|---|
| `hybrid_rerank` | 0.474 (59) | 0.500 (152) | 0.637 (93) | 0.664 (55) |
| `hybrid_rerank_depth100` | 0.471 (59) | 0.495 (152) | 0.633 (93) | 0.647 (55) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `hybrid_rerank` | 0.422 (103) | 0.611 (256) |
| `hybrid_rerank_depth100` | 0.420 (103) | 0.603 (256) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `hybrid_rerank_depth100` vs `hybrid_rerank` | ndcg@10 | 0.556 | 0.550 | -0.006 [-0.014, +0.002] | 6 / 32 / 321 |
| `hybrid_rerank_depth100` vs `hybrid_rerank` | recall@50 | 0.784 | 0.807 | +0.023 [+0.003, +0.045] | 14 / 5 / 340 |
| `hybrid_rerank_depth100` vs `hybrid_rerank` | mrr@10 | 0.517 | 0.513 | -0.004 [-0.009, +0.001] | 6 / 30 / 323 |

## Validation — 120 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 0.592 [0.504, 0.671] | 0.696 [0.613, 0.771] | 0.469 [0.390, 0.545] | 0.434 [0.354, 0.515] | 0.060 | 0.787 | 0 | 0 |
| `hybrid_rerank_depth100` | 0.592 [0.504, 0.679] | 0.729 [0.650, 0.800] | 0.468 [0.389, 0.545] | 0.433 [0.353, 0.515] | 0.060 | 0.787 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 514.8 | 851.4 | 1438.8 | 14.5 | 49.6 | 790.6 | 7.829 |
| `hybrid_rerank_depth100` | 957.4 | 1307.6 | 1645.6 | 9.8 | 43.2 | 1274.9 | 9.509 |

nDCG@10 by query set:

| Variant | inline_acl (n) | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|---|
| `hybrid_rerank` | 0.347 (20) | 0.436 (51) | 0.502 (31) | 0.642 (18) |
| `hybrid_rerank_depth100` | 0.340 (20) | 0.434 (51) | 0.502 (31) | 0.647 (18) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `hybrid_rerank` | 0.296 (23) | 0.510 (97) |
| `hybrid_rerank_depth100` | 0.307 (23) | 0.506 (97) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `hybrid_rerank_depth100` vs `hybrid_rerank` | ndcg@10 | 0.469 | 0.468 | -0.001 [-0.009, +0.008] | 1 / 4 / 115 |
| `hybrid_rerank_depth100` vs `hybrid_rerank` | recall@50 | 0.696 | 0.729 | +0.033 [-0.008, +0.075] | 5 / 1 / 114 |
| `hybrid_rerank_depth100` vs `hybrid_rerank` | mrr@10 | 0.434 | 0.433 | -0.001 [-0.005, +0.003] | 1 / 4 / 115 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03. A candidate with any degraded query is never chosen (`require_undegraded`).

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Degraded | Promoted |
|---|---|---|---|---|---|---|
| `hybrid_rerank` → `hybrid_rerank_depth100` | -0.001 [-0.009, +0.008] | 1.31 | no | yes | 0 | no |

**Chosen on validation: `hybrid_rerank` (mode `hybrid_rerank`); ablations adopted: none.**

## Analysis

Written 2026-10-02 after the two clean runs above. The rule in
`configs/experiments/e3-rerank-depth.yaml` was committed in `a18a991` before either run and
was not changed afterwards. The locked test split has **not** been run.

### Decision: keep the rerank depth at 50

The rule promotes depth 100 only if three things hold. Its paired nDCG@10 interval must lie
wholly above zero, its p95 must be within 3 s, and neither variant may degrade a query. On
the 120 validation queries, the second and third held but the first did not:

| Validation | Difference [95% interval] | Wins / losses |
|---|---|---|
| nDCG@10 | −0.001 [−0.009, +0.008] | 1 / 4 |
| Recall@50 | +0.033 [−0.008, +0.075] | 5 / 1 |
| Recall@100 | +0.042 [+0.008, +0.083] | 5 / 0 |

Development, which was not used to choose, agrees: nDCG@10 −0.006 [−0.014, +0.002], with 6
wins and 32 losses.

### What deeper reranking does

- **It reaches more gold papers and promotes them badly.** The deeper head gives the
  cross-encoder every gold paper the fused top 100 holds. On development that is 82.8% of
  them, against 78.4% in the top 50. Recall@100 rises by +0.044 [+0.024, +0.065] with no
  query losing. But the 50 extra candidates also bring distractors the cross-encoder ranks
  above gold. Recall@10 falls by 0.013 on development, and nDCG@10 does not move on either
  split. The cross-encoder, not the cut at 50, now limits the top of the page.
- **It costs half a second.** The rerank stage p95 rose from 719 to 1,193 ms on development,
  and from 791 to 1,275 ms on validation. Neither run timed out a single rerank.

### For the LLM reranking plan

LitSearch's GPT-4o reranker reads the top 100. Our cross-encoder at depth 100 hands an LLM
4.2–4.4 more points of gold than depth 50 does. Whether the LLM can use them is the LLM
pilot's question, not this experiment's: an LLM variant may set its own rerank depth.

### Runs

Both runs were clean. The GPU was 0% busy beforehand, and during each run the only process
on it besides ours was the idle one present beforehand. Utilization p50 was 97%, all our
reranker's. Both manifests record code `1467e20` plus uncommitted, non-code changes: the
ceiling run's outputs and the P2.6 report.
