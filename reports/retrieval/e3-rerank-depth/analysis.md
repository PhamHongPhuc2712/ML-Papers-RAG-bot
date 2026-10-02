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
