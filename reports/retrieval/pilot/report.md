# Retrieval experiment — m2-retrieval-pilot

Rendered by `eval report` from `reports/retrieval/pilot/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation |
|---|---|---|
| Run | `m2-retrieval-pilot-development-20260930T093637Z` | `m2-retrieval-pilot-validation-20260930T093951Z` |
| Code | `4c54044765` + uncommitted `f5e8cedc4f` | `4c54044765` + uncommitted `1100fd6d78` |
| Corpus release | `m2-20260924T095724Z` | `m2-20260924T095724Z` |
| Dataset | `dfcb643e87b5`, in_domain | `dfcb643e87b5`, in_domain |
| Queries | 150 | 52 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 95% / max 100%; memory 10173–11600 MiB; 2 GPU processes seen | util p50 97% / max 100%; memory 10204–11174 MiB; 2 GPU processes seen |
| Locked test | no | no |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 150 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.633 [0.560, 0.707] | 0.740 [0.673, 0.807] | 0.485 [0.417, 0.553] | 0.438 [0.371, 0.506] | 0.063 | 0.780 | 0 | 0 |
| `dense` | 0.593 [0.513, 0.673] | 0.767 [0.700, 0.833] | 0.450 [0.378, 0.522] | 0.406 [0.335, 0.480] | 0.059 | 0.813 | 0 | 0 |
| `dense_bge_small` | 0.600 [0.520, 0.680] | 0.753 [0.687, 0.820] | 0.448 [0.379, 0.520] | 0.400 [0.330, 0.473] | 0.060 | 0.787 | 0 | 0 |
| `hybrid` | 0.693 [0.620, 0.767] | 0.807 [0.740, 0.867] | 0.551 [0.484, 0.620] | 0.506 [0.436, 0.578] | 0.069 | 0.900 | 0 | 0 |
| `hybrid_rerank` | 0.740 [0.667, 0.807] | 0.807 [0.740, 0.867] | 0.585 [0.523, 0.655] | 0.536 [0.473, 0.609] | 0.074 | 0.900 | 0 | 2 |
| `hybrid_rerank_bge_small` | 0.753 [0.687, 0.820] | 0.833 [0.773, 0.887] | 0.599 [0.535, 0.663] | 0.550 [0.481, 0.619] | 0.075 | 0.880 | 0 | 3 |
| `hybrid_rerank_pair512` | 0.740 [0.667, 0.807] | 0.807 [0.740, 0.867] | 0.585 [0.523, 0.655] | 0.536 [0.473, 0.609] | 0.074 | 0.900 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25` | 5.9 | 8.3 | 12.4 | 8.0 | — | — | 1.65 |
| `dense` | 31.7 | 53.9 | 893.1 | — | 53.5 | — | 2.464 |
| `dense_bge_small` | 18.9 | 29.9 | 56.9 | — | 29.6 | — | 0.827 |
| `hybrid` | 28.9 | 46.1 | 54.9 | 7.9 | 45.7 | — | 0.571 |
| `hybrid_rerank` | 572.3 | 687.9 | 1573.4 | 10.1 | 60.4 | 573.6 | 6.826 |
| `hybrid_rerank_bge_small` | 553.2 | 1282.8 | 1547.5 | 16.4 | 32.6 | 631.8 | 5.824 |
| `hybrid_rerank_pair512` | 558.0 | 633.9 | 1366.8 | 9.7 | 76.3 | 561.3 | 6.497 |

nDCG@10 by query set:

| Variant | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|
| `bm25` | 0.477 (3) | 0.559 (93) | 0.360 (54) |
| `dense` | 0.667 (3) | 0.504 (93) | 0.345 (54) |
| `dense_bge_small` | 0.667 (3) | 0.497 (93) | 0.352 (54) |
| `hybrid` | 0.833 (3) | 0.629 (93) | 0.400 (54) |
| `hybrid_rerank` | 0.655 (3) | 0.657 (93) | 0.458 (54) |
| `hybrid_rerank_bge_small` | 0.649 (3) | 0.675 (93) | 0.465 (54) |
| `hybrid_rerank_pair512` | 0.655 (3) | 0.657 (93) | 0.458 (54) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `dense_bge_small` vs `dense` | ndcg@10 | 0.450 | 0.448 | -0.002 [-0.055, +0.058] | 34 / 37 / 79 |
| `dense_bge_small` vs `dense` | recall@50 | 0.767 | 0.753 | -0.013 [-0.067, +0.047] | 9 / 11 / 130 |
| `dense_bge_small` vs `dense` | mrr@10 | 0.406 | 0.400 | -0.006 [-0.065, +0.057] | 34 / 37 / 79 |
| `dense` vs `bm25` | ndcg@10 | 0.485 | 0.450 | -0.035 [-0.103, +0.031] | 35 / 42 / 73 |
| `dense` vs `bm25` | recall@50 | 0.740 | 0.767 | +0.027 [-0.047, +0.093] | 17 / 13 / 120 |
| `dense` vs `bm25` | mrr@10 | 0.438 | 0.406 | -0.032 [-0.104, +0.040] | 35 / 42 / 73 |
| `hybrid_rerank_bge_small` vs `hybrid_rerank` | ndcg@10 | 0.585 | 0.599 | +0.013 [-0.013, +0.042] | 15 / 7 / 128 |
| `hybrid_rerank_bge_small` vs `hybrid_rerank` | recall@50 | 0.807 | 0.833 | +0.027 [-0.020, +0.073] | 8 / 4 / 138 |
| `hybrid_rerank_bge_small` vs `hybrid_rerank` | mrr@10 | 0.536 | 0.550 | +0.013 [-0.013, +0.042] | 15 / 7 / 128 |
| `hybrid_rerank_pair512` vs `hybrid_rerank` | ndcg@10 | 0.585 | 0.585 | +0.000 [+0.000, +0.000] | 0 / 0 / 150 |
| `hybrid_rerank_pair512` vs `hybrid_rerank` | recall@50 | 0.807 | 0.807 | +0.000 [+0.000, +0.000] | 0 / 0 / 150 |
| `hybrid_rerank_pair512` vs `hybrid_rerank` | mrr@10 | 0.536 | 0.536 | +0.000 [+0.000, +0.000] | 0 / 0 / 150 |
| `hybrid_rerank` vs `hybrid` | ndcg@10 | 0.551 | 0.585 | +0.035 [-0.009, +0.079] | 30 / 24 / 96 |
| `hybrid_rerank` vs `hybrid` | recall@50 | 0.807 | 0.807 | +0.000 [+0.000, +0.000] | 0 / 0 / 150 |
| `hybrid_rerank` vs `hybrid` | mrr@10 | 0.506 | 0.536 | +0.030 [-0.018, +0.080] | 30 / 24 / 96 |
| `hybrid` vs `dense` | ndcg@10 | 0.450 | 0.551 | +0.101 [+0.047, +0.152] | 49 / 13 / 88 |
| `hybrid` vs `dense` | recall@50 | 0.767 | 0.807 | +0.040 [-0.007, +0.093] | 10 / 4 / 136 |
| `hybrid` vs `dense` | mrr@10 | 0.406 | 0.506 | +0.100 [+0.045, +0.157] | 49 / 13 / 88 |

## Validation — 52 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.558 [0.423, 0.692] | 0.692 [0.558, 0.808] | 0.446 [0.317, 0.566] | 0.411 [0.283, 0.539] | 0.056 | 0.750 | 0 | 0 |
| `dense` | 0.596 [0.462, 0.731] | 0.712 [0.577, 0.827] | 0.428 [0.312, 0.538] | 0.374 [0.263, 0.488] | 0.060 | 0.769 | 0 | 0 |
| `dense_bge_small` | 0.519 [0.385, 0.654] | 0.712 [0.596, 0.827] | 0.388 [0.284, 0.502] | 0.346 [0.241, 0.457] | 0.052 | 0.731 | 0 | 0 |
| `hybrid` | 0.635 [0.500, 0.750] | 0.769 [0.654, 0.865] | 0.510 [0.386, 0.619] | 0.470 [0.345, 0.583] | 0.063 | 0.846 | 0 | 0 |
| `hybrid_rerank` | 0.635 [0.500, 0.750] | 0.769 [0.654, 0.865] | 0.516 [0.396, 0.626] | 0.477 [0.356, 0.590] | 0.063 | 0.846 | 0 | 0 |
| `hybrid_rerank_bge_small` | 0.635 [0.500, 0.750] | 0.750 [0.635, 0.846] | 0.532 [0.408, 0.644] | 0.499 [0.373, 0.615] | 0.063 | 0.846 | 0 | 0 |
| `hybrid_rerank_pair512` | 0.635 [0.500, 0.750] | 0.769 [0.654, 0.865] | 0.516 [0.396, 0.626] | 0.477 [0.356, 0.590] | 0.063 | 0.846 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25` | 6.5 | 13.7 | 31.6 | 13.5 | — | — | 1.508 |
| `dense` | 28.9 | 43.3 | 52.8 | — | 43.0 | — | 0.595 |
| `dense_bge_small` | 17.5 | 28.8 | 30.4 | — | 28.5 | — | 0.53 |
| `hybrid` | 32.0 | 47.9 | 51.5 | 8.0 | 47.4 | — | 0.607 |
| `hybrid_rerank` | 546.4 | 603.8 | 632.8 | 10.7 | 42.4 | 557.6 | 6.07 |
| `hybrid_rerank_bge_small` | 544.9 | 601.3 | 614.6 | 10.9 | 29.5 | 570.4 | 5.687 |
| `hybrid_rerank_pair512` | 545.3 | 609.6 | 1201.0 | 8.8 | 39.1 | 565.9 | 5.619 |

nDCG@10 by query set:

| Variant | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|
| `bm25` | 0.333 (3) | 0.438 (31) | 0.478 (18) |
| `dense` | 0.144 (3) | 0.404 (31) | 0.516 (18) |
| `dense_bge_small` | 0.000 (3) | 0.453 (31) | 0.341 (18) |
| `hybrid` | 0.333 (3) | 0.502 (31) | 0.553 (18) |
| `hybrid_rerank` | 0.333 (3) | 0.530 (31) | 0.521 (18) |
| `hybrid_rerank_bge_small` | 0.333 (3) | 0.543 (31) | 0.546 (18) |
| `hybrid_rerank_pair512` | 0.333 (3) | 0.530 (31) | 0.521 (18) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `dense_bge_small` vs `dense` | ndcg@10 | 0.428 | 0.388 | -0.040 [-0.133, +0.044] | 11 / 15 / 26 |
| `dense_bge_small` vs `dense` | recall@50 | 0.712 | 0.712 | +0.000 [-0.115, +0.115] | 5 / 5 / 42 |
| `dense_bge_small` vs `dense` | mrr@10 | 0.374 | 0.346 | -0.029 [-0.124, +0.062] | 11 / 15 / 26 |
| `dense` vs `bm25` | ndcg@10 | 0.446 | 0.428 | -0.018 [-0.137, +0.109] | 11 / 17 / 24 |
| `dense` vs `bm25` | recall@50 | 0.692 | 0.712 | +0.019 [-0.096, +0.135] | 5 / 4 / 43 |
| `dense` vs `bm25` | mrr@10 | 0.411 | 0.374 | -0.037 [-0.153, +0.088] | 11 / 17 / 24 |
| `hybrid_rerank_bge_small` vs `hybrid_rerank` | ndcg@10 | 0.516 | 0.532 | +0.016 [+0.001, +0.039] | 4 / 0 / 48 |
| `hybrid_rerank_bge_small` vs `hybrid_rerank` | recall@50 | 0.769 | 0.750 | -0.019 [-0.096, +0.058] | 2 / 3 / 47 |
| `hybrid_rerank_bge_small` vs `hybrid_rerank` | mrr@10 | 0.477 | 0.499 | +0.021 [+0.000, +0.052] | 4 / 0 / 48 |
| `hybrid_rerank_pair512` vs `hybrid_rerank` | ndcg@10 | 0.516 | 0.516 | +0.000 [+0.000, +0.000] | 0 / 0 / 52 |
| `hybrid_rerank_pair512` vs `hybrid_rerank` | recall@50 | 0.769 | 0.769 | +0.000 [+0.000, +0.000] | 0 / 0 / 52 |
| `hybrid_rerank_pair512` vs `hybrid_rerank` | mrr@10 | 0.477 | 0.477 | +0.000 [+0.000, +0.000] | 0 / 0 / 52 |
| `hybrid_rerank` vs `hybrid` | ndcg@10 | 0.510 | 0.516 | +0.006 [-0.060, +0.071] | 10 / 9 / 33 |
| `hybrid_rerank` vs `hybrid` | recall@50 | 0.769 | 0.769 | +0.000 [+0.000, +0.000] | 0 / 0 / 52 |
| `hybrid_rerank` vs `hybrid` | mrr@10 | 0.470 | 0.477 | +0.008 [-0.073, +0.084] | 10 / 9 / 33 |
| `hybrid` vs `dense` | ndcg@10 | 0.428 | 0.510 | +0.082 [-0.010, +0.169] | 16 / 5 / 31 |
| `hybrid` vs `dense` | recall@50 | 0.712 | 0.769 | +0.058 [+0.000, +0.135] | 3 / 0 / 49 |
| `hybrid` vs `dense` | mrr@10 | 0.374 | 0.470 | +0.096 [-0.003, +0.191] | 16 / 5 / 31 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03.

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Promoted |
|---|---|---|---|---|---|
| `bm25` → `dense` | -0.018 [-0.137, +0.109] | 0.04 | no | yes | no |
| `bm25` → `hybrid` | +0.064 [-0.009, +0.138] | 0.05 | no | yes | no |
| `bm25` → `hybrid_rerank` | +0.070 [-0.010, +0.151] | 0.60 | no | yes | no |

| Ablation | Change | Difference [interval] | Non-inferior | Applies to the choice | Adopted |
|---|---|---|---|---|---|
| `hybrid_rerank_pair512` vs `hybrid_rerank` | reranker pair budget 1,024 -> 512 tokens (spec §7) | +0.000 [+0.000, +0.000] | yes | no | no |
| `dense_bge_small` vs `dense` | embedding BGE-M3 (1,024-d) -> BGE-small-en-v1.5 (384-d) | -0.040 [-0.133, +0.044] | no | no | no |
| `hybrid_rerank_bge_small` vs `hybrid_rerank` | embedding BGE-M3 (1,024-d) -> BGE-small-en-v1.5 (384-d) | +0.016 [+0.001, +0.039] | yes | no | no |

**Chosen on validation: `bm25` (mode `bm25`); ablations adopted: none.**

## Analysis

Written 2026-09-30 after the development and validation runs above. The locked test split
has **not** been run.

### Decision: retain BM25 as the first search configuration

The rules were fixed in `configs/experiments/retrieval.yaml` before any validation number
existed. A costlier mode replaces the current one only if the paired 95% interval of its
nDCG@10 gain lies wholly above zero. On validation neither fusion mode clears that bar:

| vs `bm25` on nDCG@10 | Validation, 52 queries | Development, 150 queries (not used to choose) |
|---|---|---|
| `hybrid` | +0.064 [−0.009, +0.138] | +0.065 [+0.022, +0.105] |
| `hybrid_rerank` | +0.070 [−0.010, +0.151] | +0.100 [+0.046, +0.153] |
| `hybrid_rerank` vs `hybrid` | +0.006 [−0.060, +0.071] | +0.035 [−0.009, +0.079] |

The development rows were computed from `development/per_query.parquet` with the same
bootstrap: same seed, same family resampling. They are shown because the reader should
see them, not because they may choose.

**What the evidence says, as distinct from what the rule decided.** Fusion's gain over
BM25 is about +0.065 nDCG@10 on both splits. On development its interval excludes zero
comfortably; on validation the same effect size is inconclusive because 52 queries are
too few. Hybrid costs almost nothing in time: p95 48 ms against BM25's 14 ms. The
reranker's own contribution is small and unconfirmed on both splits, at about 550 ms per
query. So the rule retains BM25 for lack of validation evidence, not because fusion was
shown not to help. Changing the rule now to reach the expected answer would make the
experiment worthless. The way forward is more evidence: E3 over LitSearch's own corpus
(597 queries) and E4's hand-authored queries, then the locked test once, for the release
decision.

### Ablations

- **Reranker pair budget 1,024 → 512: identical, and non-inferior.** All 202 queries,
  development and validation, tie exactly. The budget barely binds on paper text: in the
  reranker's tokenizer, title plus abstract is 315 tokens at the median, 434 at p95 and
  499 at p99. Only 1.6% of the 85,729 papers exceed 480 tokens; at 1,024, only 3 are cut.
  The latency saving is correspondingly nil: rerank p95 on validation was 566 ms at 512
  tokens and 558 ms at 1,024. The comparison that would matter is chunk reranking in P4, where chunks run
  to 1,200 tokens. Retain 1,024, the spec's default; nothing was bought by the change.
- **BGE-M3 → BGE-small-en-v1.5, dense alone: not shown non-inferior.** Validation
  −0.040 [−0.133, +0.044]; development −0.002 [−0.055, +0.058]. Both lower bounds pass the
  −0.03 margin, so the rule cannot rule out a real loss, and BGE-M3 is retained.
- **BGE-M3 → BGE-small inside `hybrid_rerank`: non-inferior, even marginally ahead.**
  Validation +0.016 [+0.001, +0.039], with 4 wins and 0 losses; development +0.013
  [−0.013, +0.042]. It would be adopted if `hybrid_rerank` were the chosen mode; it is not,
  so no pin changes. Worth carrying forward: under fusion and reranking, the dense model's
  size matters little. BGE-small's paper vectors are 132 MB against 351 MB; it built in
  168 s against 13 min; dense-stage p95 is 29 ms against 42 ms. The smaller model
  truncates 171 papers at its 512-position limit; BGE-M3 truncates none.

### What else the runs show

- **The rerank depth of 50 loses candidates.** On development, 90.0% of gold papers are
  somewhere in the union of the two branches' top 100, but only 80.7% are in the fused
  top 50 the reranker sees. On validation the figures are 84.6% and 76.9%. A reranker
  cannot recover what it is not shown. A depth-100 run is the obvious next ablation; the
  cost is roughly double the reranker time.
- **Dense alone is not better than BM25 here** (validation −0.018, development −0.035).
  These are known-item queries written from a specific paper, so they share vocabulary
  with its title and abstract — the setting where exact BM25 is strong. The two branches
  find different papers, which is why fusing them helps.
- **Judged coverage is about 6% at rank 10.** LitSearch labels the one paper each query
  was written from, so nine of ten returned papers are unjudged, not irrelevant. The
  metrics measure how high the known paper ranks, not how useful the rest of the page is.
- **No query failed.** A few rerank calls timed out and were served in fused order:
  2 and 3 of 150 on development, none on validation.
- **The GPU was ours during the timed runs.** Another project's process held 7.8 GB of
  GPU memory but was idle, at 0% utilization, whenever sampled outside our runs: before and
  after the development run, and after the validation run. Utilization
  during the runs (p50 95%) matches our own reranking load. The rerank-stage p95 of
  558–632 ms matches P2.3's uncontended pilot (567 ms).

### Limits

- 52 validation and 48 test queries are small instruments. The intervals above are the
  honest measure of that.
- The queries are LitSearch's author-written known-item sets: `manual_acl` and
  `manual_iclr`, plus 7 `inline_nonacl` queries whose single gold paper happens to be in
  the corpus. Topic discovery and "keeping up" are not represented (see
  `reports/m2-labels.md`).
- Grades are binary, so nDCG@10 carries little more information than MRR@10.
