# Retrieval experiment — e3-litsearch

Rendered by `eval report` from `reports/retrieval/e3-litsearch/<split>/metrics.json`; edit `analysis.md`, not this file. Per-query rows are in `<split>/per_query.parquet`, manifests in `<split>/manifest.json`.

| | development | validation | test |
|---|---|---|---|
| Run | `e3-litsearch-development-20261001T055140Z` | `e3-litsearch-validation-20261001T055954Z` | `e3-litsearch-test-20261006T091138Z` |
| Code | `07ebd3ad89` + uncommitted `b13dd40568` | `07ebd3ad89` + uncommitted `3776bf8675` | `912497474e` + uncommitted `277805bcf4` |
| Corpus release | `litsearch-v1` | `litsearch-v1` | `litsearch-v1` |
| Dataset | `dfcb643e87b5`, all | `dfcb643e87b5`, all | `dfcb643e87b5`, all |
| Queries | 359 | 120 | 118 |
| Hardware | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 | NVIDIA GeForce RTX 3080 Laptop GPU; torch 2.14.0+cu130 |
| GPU during the run | util p50 83% / max 100%; memory 3744–5987 MiB; 1 GPU processes seen; before the run 0% busy | util p50 86% / max 100%; memory 3776–4794 MiB; 1 GPU processes seen; before the run 0% busy | util p50 97% / max 100%; memory 3813–5083 MiB; 1 GPU processes seen; before the run 0% busy |
| Locked test | no | no | yes |

Timing: Queries run one at a time through SearchService.rank with the production deadlines of configs/search.yaml; seconds are wall-clock around rank(), which includes both candidate branches, fusion, the reranked head's metadata read and the reranker, and excludes HTTP and page hydration. Before its first timed query each variant is warmed with the service's own warm-up and then 10 queries from another split (never the test split), whose results are discarded: GPU kernels warm per input shape, and without this the first variant to use a model would pay for every later one. p50/p95 interpolate linearly over every query of the split, failed ones included. GPU utilization and memory are sampled every second for the whole run, and other processes on the GPU are listed.

Cost: 0.00 USD metered — self-hosted models and services; no priced model call is made.

## Development — 359 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.547 [0.495, 0.597] | 0.678 [0.630, 0.726] | 0.411 [0.368, 0.457] | 0.373 [0.330, 0.418] | 0.058 | 0.714 | 0 | 0 |
| `dense` | 0.574 [0.520, 0.624] | 0.719 [0.676, 0.765] | 0.434 [0.389, 0.476] | 0.396 [0.350, 0.439] | 0.060 | 0.773 | 0 | 0 |
| `hybrid` | 0.597 [0.545, 0.644] | 0.784 [0.743, 0.826] | 0.474 [0.427, 0.517] | 0.441 [0.392, 0.484] | 0.063 | 0.860 | 0 | 0 |
| `hybrid_rerank` | 0.694 [0.649, 0.741] | 0.784 [0.743, 0.826] | 0.556 [0.511, 0.600] | 0.517 [0.468, 0.563] | 0.072 | 0.860 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25` | 6.9 | 14.5 | 48.2 | 14.3 | — | — | 3.473 |
| `dense` | 32.9 | 60.0 | 260.5 | — | 59.6 | — | 5.058 |
| `hybrid` | 35.1 | 53.1 | 72.6 | 9.2 | 52.7 | — | 0.868 |
| `hybrid_rerank` | 524.6 | 777.7 | 1418.6 | 13.4 | 99.4 | 735.8 | 6.642 |

nDCG@10 by query set:

| Variant | inline_acl (n) | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|---|
| `bm25` | 0.328 (59) | 0.301 (152) | 0.544 (93) | 0.581 (55) |
| `dense` | 0.381 (59) | 0.436 (152) | 0.457 (93) | 0.449 (55) |
| `hybrid` | 0.414 (59) | 0.409 (152) | 0.568 (93) | 0.562 (55) |
| `hybrid_rerank` | 0.474 (59) | 0.500 (152) | 0.637 (93) | 0.664 (55) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `bm25` | 0.271 (103) | 0.468 (256) |
| `dense` | 0.343 (103) | 0.471 (256) |
| `hybrid` | 0.320 (103) | 0.537 (256) |
| `hybrid_rerank` | 0.422 (103) | 0.611 (256) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `dense` vs `bm25` | ndcg@10 | 0.411 | 0.434 | +0.023 [-0.019, +0.065] | 93 / 76 / 190 |
| `dense` vs `bm25` | recall@50 | 0.678 | 0.719 | +0.041 [-0.010, +0.091] | 54 / 37 / 268 |
| `dense` vs `bm25` | mrr@10 | 0.373 | 0.396 | +0.023 [-0.022, +0.066] | 93 / 73 / 193 |
| `hybrid_rerank` vs `hybrid` | ndcg@10 | 0.474 | 0.556 | +0.082 [+0.050, +0.117] | 98 / 58 / 203 |
| `hybrid_rerank` vs `hybrid` | recall@50 | 0.784 | 0.784 | +0.000 [+0.000, +0.000] | 0 / 0 / 359 |
| `hybrid_rerank` vs `hybrid` | mrr@10 | 0.441 | 0.517 | +0.076 [+0.039, +0.115] | 98 / 56 / 205 |
| `hybrid` vs `dense` | ndcg@10 | 0.434 | 0.474 | +0.040 [+0.010, +0.069] | 90 / 46 / 223 |
| `hybrid` vs `dense` | recall@50 | 0.719 | 0.784 | +0.065 [+0.034, +0.096] | 33 / 10 / 316 |
| `hybrid` vs `dense` | mrr@10 | 0.396 | 0.441 | +0.045 [+0.013, +0.077] | 89 / 45 / 225 |

## Validation — 120 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.499 [0.410, 0.588] | 0.649 [0.564, 0.725] | 0.360 [0.288, 0.431] | 0.319 [0.249, 0.390] | 0.051 | 0.690 | 0 | 0 |
| `dense` | 0.526 [0.435, 0.610] | 0.625 [0.537, 0.700] | 0.385 [0.306, 0.457] | 0.341 [0.266, 0.412] | 0.054 | 0.692 | 0 | 0 |
| `hybrid` | 0.536 [0.449, 0.621] | 0.696 [0.613, 0.771] | 0.415 [0.339, 0.492] | 0.377 [0.303, 0.455] | 0.055 | 0.787 | 0 | 0 |
| `hybrid_rerank` | 0.592 [0.504, 0.671] | 0.696 [0.613, 0.771] | 0.469 [0.390, 0.545] | 0.434 [0.354, 0.515] | 0.060 | 0.787 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25` | 5.5 | 7.5 | 8.1 | 7.3 | — | — | 1.696 |
| `dense` | 24.8 | 34.9 | 50.3 | — | 34.7 | — | 1.15 |
| `hybrid` | 27.9 | 43.3 | 52.1 | 7.2 | 42.9 | — | 0.532 |
| `hybrid_rerank` | 503.1 | 812.3 | 860.7 | 9.2 | 42.6 | 785.1 | 5.57 |

nDCG@10 by query set:

| Variant | inline_acl (n) | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|---|
| `bm25` | 0.292 (20) | 0.262 (51) | 0.421 (31) | 0.606 (18) |
| `dense` | 0.286 (20) | 0.358 (51) | 0.334 (31) | 0.664 (18) |
| `hybrid` | 0.318 (20) | 0.351 (51) | 0.417 (31) | 0.698 (18) |
| `hybrid_rerank` | 0.347 (20) | 0.436 (51) | 0.502 (31) | 0.642 (18) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `bm25` | 0.229 (23) | 0.391 (97) |
| `dense` | 0.288 (23) | 0.408 (97) |
| `hybrid` | 0.304 (23) | 0.441 (97) |
| `hybrid_rerank` | 0.296 (23) | 0.510 (97) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `dense` vs `bm25` | ndcg@10 | 0.360 | 0.385 | +0.026 [-0.047, +0.089] | 30 / 29 / 61 |
| `dense` vs `bm25` | recall@50 | 0.649 | 0.625 | -0.024 [-0.099, +0.046] | 10 / 14 / 96 |
| `dense` vs `bm25` | mrr@10 | 0.319 | 0.341 | +0.022 [-0.050, +0.089] | 30 / 29 / 61 |
| `hybrid_rerank` vs `hybrid` | ndcg@10 | 0.415 | 0.469 | +0.054 [+0.007, +0.101] | 29 / 16 / 75 |
| `hybrid_rerank` vs `hybrid` | recall@50 | 0.696 | 0.696 | +0.000 [+0.000, +0.000] | 0 / 0 / 120 |
| `hybrid_rerank` vs `hybrid` | mrr@10 | 0.377 | 0.434 | +0.056 [+0.006, +0.112] | 29 / 16 / 75 |
| `hybrid` vs `dense` | ndcg@10 | 0.385 | 0.415 | +0.029 [-0.012, +0.069] | 30 / 15 / 75 |
| `hybrid` vs `dense` | recall@50 | 0.625 | 0.696 | +0.071 [+0.021, +0.121] | 11 / 1 / 108 |
| `hybrid` vs `dense` | mrr@10 | 0.341 | 0.377 | +0.037 [-0.011, +0.079] | 31 / 14 / 75 |

### Decision

Pre-registered rules (`decision` in the config): primary metric `ndcg@10`; a costlier mode is promoted only if its paired interval lies above zero and its p95 is within 3.0 s; a cheaper ablation is adopted only if its interval rules out losing more than 0.03.

| Step | Difference [interval] | p95 s | Gain supported | Fits budget | Promoted |
|---|---|---|---|---|---|
| `bm25` → `dense` | +0.026 [-0.047, +0.089] | 0.03 | no | yes | no |
| `bm25` → `hybrid` | +0.055 [+0.001, +0.104] | 0.04 | yes | yes | **yes** |
| `hybrid` → `hybrid_rerank` | +0.054 [+0.007, +0.101] | 0.81 | yes | yes | **yes** |

**Chosen on validation: `hybrid_rerank` (mode `hybrid_rerank`); ablations adopted: none.**

## Test — 118 queries

Mean over queries with a 95% bootstrap interval over query families. Failed queries score zero and stay in every denominator.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | Candidate recall | Failures | Degraded |
|---|---|---|---|---|---|---|---|---|
| `bm25` | 0.555 [0.470, 0.644] | 0.674 [0.593, 0.763] | 0.411 [0.339, 0.490] | 0.367 [0.294, 0.447] | 0.058 | 0.742 | 0 | 0 |
| `dense` | 0.551 [0.462, 0.644] | 0.746 [0.665, 0.822] | 0.406 [0.332, 0.484] | 0.367 [0.291, 0.448] | 0.058 | 0.822 | 0 | 0 |
| `hybrid` | 0.640 [0.559, 0.720] | 0.805 [0.737, 0.873] | 0.478 [0.405, 0.554] | 0.436 [0.362, 0.512] | 0.066 | 0.898 | 0 | 0 |
| `hybrid_rerank` | 0.682 [0.597, 0.763] | 0.805 [0.737, 0.873] | 0.550 [0.473, 0.621] | 0.516 [0.432, 0.589] | 0.070 | 0.898 | 0 | 0 |

Latency of `rank()` in milliseconds; stage columns are p95.

| Variant | p50 | p95 | max | lexical | dense | rerank | warm-up s |
|---|---|---|---|---|---|---|---|
| `bm25` | 6.1 | 18.1 | 48.3 | 17.6 | — | — | 5.973 |
| `dense` | 28.0 | 60.9 | 98.3 | — | 60.6 | — | 2.419 |
| `hybrid` | 29.3 | 42.1 | 52.3 | 8.2 | 41.4 | — | 0.54 |
| `hybrid_rerank` | 520.6 | 810.3 | 1141.4 | 9.1 | 38.4 | 785.0 | 6.6 |

nDCG@10 by query set:

| Variant | inline_acl (n) | inline_nonacl (n) | manual_acl (n) | manual_iclr (n) |
|---|---|---|---|---|
| `bm25` | 0.221 (19) | 0.395 (50) | 0.434 (31) | 0.614 (18) |
| `dense` | 0.275 (19) | 0.397 (50) | 0.433 (31) | 0.521 (18) |
| `hybrid` | 0.373 (19) | 0.447 (50) | 0.498 (31) | 0.638 (18) |
| `hybrid_rerank` | 0.384 (19) | 0.518 (50) | 0.625 (31) | 0.687 (18) |

nDCG@10 by specificity:

| Variant | 0 (n) | 1 (n) |
|---|---|---|
| `bm25` | 0.273 (29) | 0.456 (89) |
| `dense` | 0.326 (29) | 0.432 (89) |
| `hybrid` | 0.392 (29) | 0.506 (89) |
| `hybrid_rerank` | 0.491 (29) | 0.569 (89) |

Paired differences, candidate minus baseline, 95% interval over the same family resamples:

| Candidate vs baseline | Metric | Baseline | Candidate | Difference [interval] | Wins / losses / ties |
|---|---|---|---|---|---|
| `dense` vs `bm25` | ndcg@10 | 0.411 | 0.406 | -0.005 [-0.077, +0.066] | 34 / 28 / 56 |
| `dense` vs `bm25` | recall@50 | 0.674 | 0.746 | +0.072 [-0.030, +0.170] | 22 / 13 / 83 |
| `dense` vs `bm25` | mrr@10 | 0.367 | 0.367 | -0.001 [-0.073, +0.070] | 34 / 28 / 56 |
| `hybrid_rerank` vs `hybrid` | ndcg@10 | 0.478 | 0.550 | +0.072 [+0.011, +0.134] | 34 / 21 / 63 |
| `hybrid_rerank` vs `hybrid` | recall@50 | 0.805 | 0.805 | +0.000 [+0.000, +0.000] | 0 / 0 / 118 |
| `hybrid_rerank` vs `hybrid` | mrr@10 | 0.436 | 0.516 | +0.080 [+0.010, +0.149] | 34 / 21 / 63 |
| `hybrid` vs `dense` | ndcg@10 | 0.406 | 0.478 | +0.072 [+0.018, +0.117] | 39 / 13 / 66 |
| `hybrid` vs `dense` | recall@50 | 0.746 | 0.805 | +0.059 [-0.008, +0.127] | 11 / 4 / 103 |
| `hybrid` vs `dense` | mrr@10 | 0.367 | 0.436 | +0.069 [+0.017, +0.114] | 39 / 13 / 66 |

## Analysis

Written 2026-10-01 after the two clean runs above. The locked test split has **not** been
run.

### Decision: the pre-registered rule promotes hybrid, then reranking

The rules in `configs/experiments/e3-litsearch.yaml` are P2.5's, copied unchanged, fixed
before any E3 number existed. On the 120 validation queries both steps clear them:

| Step on validation | nDCG@10 difference [95% interval] | p95 | Promoted |
|---|---|---|---|
| `bm25` → `dense` | +0.026 [−0.047, +0.089] | 35 ms | no |
| `bm25` → `hybrid` | +0.055 [+0.001, +0.104] | 43 ms | yes |
| `hybrid` → `hybrid_rerank` | +0.054 [+0.007, +0.101] | 812 ms | yes |

Development (359 queries, not used to choose) agrees and is sharper. Hybrid gains +0.040
[+0.010, +0.069] over dense and +0.063 over BM25; reranking adds +0.082 [+0.050, +0.117]
over hybrid, with 98 wins against 58 losses.

**Against P2.5.** On our corpus, P2.5's 52 in-domain validation queries saw fusion's gain
over BM25 (+0.064) but could not confirm it, so its rule kept BM25. E3 has 2.3 times the
validation queries and confirms both fusion and reranking. The two experiments agree in
direction everywhere; they differ in what 52 queries can show. E3 is the evidence the
benchmarks plan designates as primary for G2, so the configuration it selects is
`hybrid_rerank`:
- BGE-M3 and exact BM25, fused by RRF (k = 60);
- the top 50 reranked by `bge-reranker-v2-m3` at a 1,024-token pair budget;
- p95 of 0.81 s against a 3 s budget.

The locked test split, run once, is what confirms it.

### What the query sets show

- **BM25 and dense win on different queries.** On development, dense beats BM25 on the
  citation-derived sets: `inline_nonacl` 0.436 against 0.301, `inline_acl` 0.381 against
  0.328. BM25 beats dense on the author-written ones: `manual_acl` 0.544 against 0.457,
  `manual_iclr` 0.581 against 0.449. Author-written queries describe one known paper in its
  own vocabulary; citation-derived ones paraphrase what a cited work did. P2.5's in-domain
  slice is almost all author-written, which is why dense alone lost to BM25 there. Fusion
  keeps the better branch on each kind, and the reranker adds on all four sets on
  development.
- **Broad queries are harder for everything.** Specificity 0 scores 0.13–0.22 below
  specificity 1 for every variant on development: `hybrid_rerank` 0.422 against 0.611, and
  dense has the smallest gap (0.343 against 0.471).
- **The depth-50 rerank cut loses candidates again.** Gold papers are in the union of the
  two branches' top 100 for 86.0% of development queries and 78.7% of validation queries,
  but in the fused top 50 the reranker sees for only 78.4% and 69.6%. Reranking deeper is
  the next ablation, as it was in P2.5.
- **The reranker helps more here than on our corpus.** Its gain over hybrid on development
  is +0.082 here against +0.035 in P2.5. With different corpora, queries and sizes, this is
  recorded, not explained.

### Running on a shared GPU

This host's GPU also ran another project's training jobs during E3: a cross-encoder
distillation, then cross-encoder training. Three attempts ran with one of them on the GPU.
Each recorded the other process in its manifest and was set aside:

| Attempt | Rerank timeouts, falling back to fused order | Every other query |
|---|---|---|
| development, first | 60 of 359 | ranked identically to the clean run |
| development, second | 183 of 359 | ranked identically to the clean run |
| validation, first | 31 of 120 | ranked identically to the clean run |

BM25, dense and hybrid ranked every query identically in all three. The runs above are
clean: 0% utilization before each, and no other process on the GPU during it. This is
also the plan's reproducibility case, met on the real corpus: a rerun with the same
revisions reproduces every ranking that did not miss a deadline.

### Limits

- **Judged coverage is about 6% at rank 10.** LitSearch labels the paper or papers each
  query was written about; the rest of a page is unjudged, not wrong.
- **Title and abstract only**, the shape the corpus is packaged in. No full-text
  configuration was run.
- **These metrics are not comparable to published LitSearch tables without checking
  definitions.** Our cutoffs (Recall@10/50, nDCG@10, MRR@10), deduplication and scoring
  are ours.
- **120 validation queries is still a modest instrument.** The lower bound of the hybrid
  interval sits at +0.001.
