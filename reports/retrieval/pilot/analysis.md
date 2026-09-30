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
