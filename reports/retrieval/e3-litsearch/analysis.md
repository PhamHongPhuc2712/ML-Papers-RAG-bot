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
