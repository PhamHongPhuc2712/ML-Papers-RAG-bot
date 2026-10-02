# P2.6 evidence — the retrieval gaps E3 and P2.5 exposed

Date: 2026-10-01
Task: P2.6 of the retrieval plan (urgent, before the locked test split and before P3.1)
Status: **in progress.** Step 1, the offline analysis, is done. The harness can now run the
step 2 ceiling and the step 3 depth-100 ablation, and the depth-100 rule is written. The
ceiling run is pending an idle GPU, and the depth-100 runs have not started. The locked
test split has **not** been run.
Host: WSL2 Linux, 12 cores / 23 GB RAM, RTX 3080 Laptop 16 GB.

## What changed in the harness

| Piece | Where | Why |
|---|---|---|
| `eval gaps`: recall at LitSearch's published cutoffs, where each gold paper is lost (returned / pooled but cut / never pooled), gold rank bands, missed query ids per slice. Reads only recorded `per_query.parquet`; refuses the test split | `evaluation/gaps.py`, `cli.py`, `tests/unit/test_gaps.py` | Step 1, and the gap table must be regenerated after the final run |
| A variant may override `candidates.per_branch` (any mode), `rerank.depth` and `deadlines_seconds.rerank` (`hybrid_rerank` only); anything else is refused when the config loads. Every variant's effective settings and their digest go in the manifest | `evaluation/retrieval.py` | The plan's prerequisite for steps 2–4 |
| `eval compare` lists variants whose search settings differ between the two runs | `evaluation/regression.py` | Runs that searched differently are never compared as equal |
| `require_undegraded` decision rule, opt-in: a step is promoted only if neither side degraded a query | `evaluation/retrieval.py`, `evaluation/report.py` | Encodes "any rerank timeout on a clean GPU sets the run aside" |

**Deviation:** the plan's file list names `evaluation/retrieval.py` and the eval-runner
integration test only. The offline analysis got its own module and a unit test file,
because it is reread from every recorded run and has nothing to do with running one. The
three config-validation tests went into `tests/unit/test_regression.py`, beside the
existing experiment-config tests, so offline CI runs them. Re-rendering the recorded E3 and
P2.5 reports after these changes produces no diff: their decisions are unchanged.

## Step 1 — offline, no GPU

### LitSearch's definitions, checked

- **Cutoffs:** R@20 for broad questions; R@5 and R@20 for specific questions. The
  inline-citation and author-written sets are reported apart. All over **titles and
  abstracts** (arXiv 2407.18940v2, Table 3 and its caption).
- **Recall:** the share of a query's gold papers in the top k
  (`utils.calculate_recall` in `princeton-nlp/LitSearch`: a set intersection over the
  number of golds). That is our `recall_at_k`, deduplication included.
- **Specificity:** 0 is broad ("no more than 20 papers fit"), 1 is specific ("no more
  than 5"), from the paper's annotation rubric. Its Table 2 counts, 155 broad and 442
  specific, match the data. This settles the scale E1 recorded as undocumented.
- **License, unchanged:** the GitHub repository is MIT-licensed, but the license text
  covers "the Software" only. The dataset still declares none, so local-use-only stands.

### Beside the published table

Our development and validation splits together are 479 of the 597 queries; the locked test
split is excluded, so the samples overlap rather than match. The cells are percentages.
Every interval is in `reports/retrieval/e3-litsearch/gaps.md`, and most span ±7–10 points.

| System | Inline broad R@20 | Inline specific R@5 | Inline specific R@20 | Author broad R@20 | Author specific R@5 | Author specific R@20 |
|---|---|---|---|---|---|---|
| ours `bm25` | 40.1 | 37.2 | 56.8 | 42.9 | 62.1 | 75.1 |
| ours `dense` (BGE-M3) | 53.1 | 49.7 | 62.0 | 50.0 | 55.0 | 69.8 |
| ours `hybrid` | 43.6 | 48.9 | 67.1 | 60.7 | 66.3 | 77.5 |
| ours `hybrid_rerank` | 54.1 | 57.6 | 72.8 | 60.7 | 72.2 | 83.4 |
| published BM25 | 37.4 | 38.5 | 55.8 | 48.6 | 62.6 | 73.5 |
| published Instructor-XL | 56.3 | 48.9 | 60.0 | 57.1 | 55.9 | 70.1 |
| published E5-large-v2 | 55.8 | 50.4 | 63.9 | 54.3 | 62.6 | 75.8 |
| published GritLM-7B | 69.7 | 67.7 | 77.9 | 74.3 | 82.5 | 89.1 |
| published GPT-4o reranking (w/ GritLM) | 74.7 | 73.2 | 79.9 | 77.1 | 85.8 | 92.4 |

- **Our BM25 reproduces the paper's BM25.** Every published cell is inside our interval.
  This is independent evidence that the harness, the labels-by-corpusid path and the exact
  BM25 are measuring what the paper measured.
- **`hybrid_rerank` beats E5-large-v2 and Instructor-XL in five of six cells.** The
  exception is inline broad R@20, where it is 1.7–2.2 points lower, well inside the interval.
- **It trails GritLM-7B in all six cells, by 5–16 points.** GritLM is a 7B embedding model;
  the stack's two models are BGE-M3 (568M) and a 568M cross-encoder. Inline broad is the
  widest gap.
- **Fusion reorders broad citation queries but loses nothing that matters to the
  reranker.** Diagnosed on development only (79 queries), hybrid trails dense alone at R@20
  by −0.096 [−0.158, −0.039], but not at R@50 (+0.023 [−0.032, +0.082]). Equal-weight RRF
  pushes dense's hits from the top 20 into ranks 21–50, still inside the reranked head.
  Fusion weighting is therefore not a recall lever for `hybrid_rerank`.

### Where the gold paper is lost

Shares of gold papers, averaged per query. The pool is the union of each branch's top 100;
for a single-branch mode it is that branch's list. Regenerated from the recorded runs:

| Run | Queries | Returned in top 50 | In pool, cut at 50 | Not in pool |
|---|---|---|---|---|
| E3 development, `hybrid_rerank` | 359 | 78.4 | 7.6 | 14.0 |
| E3 validation, `hybrid_rerank` | 120 | 69.6 | 9.2 | 21.2 |
| P2.5 development, `hybrid_rerank` | 150 | 80.7 | 9.3 | 10.0 |
| P2.5 validation, `hybrid_rerank` | 52 | 76.9 | 7.7 | 15.4 |

These match the plan's table. The plan's 9.1 and 21.3 for E3 validation are the same values,
9.17 and 21.25, rounded differently.

- **The branches are complementary.** On E3 development, BM25's own top 100 misses 28.6% of
  golds and dense's misses 22.7%, but their union misses only 14.0%.
- **Broad queries account for most of the pool misses.** On E3 development, broad
  questions miss 24.6% of their golds from the pool against 9.8% for specific ones. By
  set, `inline_acl` misses 21.0% and the author-written sets 8.6–9.1%.
- **On our corpus, the depth-50 cut falls hardest on ICLR.** In P2.5 development,
  `manual_iclr` loses 20.4% of its golds between rank 50 and the pool's edge, against 3.2%
  for `manual_acl`. One plausible reason, not yet tested: our corpus holds about 13k ICLR
  papers, many on neighbouring topics, so more near-duplicates compete for the top 50.
  Depth 100 would act most there.

Gold rank bands and the query ids behind every miss, per slice, are in
`reports/retrieval/e3-litsearch/gaps.md` and `reports/retrieval/pilot/gaps.md`.

## Step 2 — the ceilings

Pending: `configs/experiments/e3-candidates.yaml` (hybrid only, development, pools of 100,
200 and 300 per branch, rankings kept 100 deep). It waits for an idle GPU. Another
project's training jobs were using the GPU when it was due.

## Step 3 — rerank depth 100

Pre-registered in `configs/experiments/e3-rerank-depth.yaml`: depth 100 with a 2.5 s rerank
deadline, against the shipped depth 50 at 1.5 s. It is promoted only if:
- the paired nDCG@10 interval lies wholly above zero;
- the end-to-end p95 is ≤ 3 s;
- neither variant degrades a single query (`require_undegraded`).

The rule must be committed before the validation run. Not run yet.

## Commands and results

```text
uv run --env-file .env.test --project backend pytest backend/tests -q           -> 477 passed
uv run --project backend pytest backend/tests -m "not integration" -q           -> 325 passed
uv run --env-file .env.test --project backend pytest backend/tests/integration/test_eval_runner.py -q
                                                                                -> 8 passed
uv run --project backend python -m copilot.cli eval smoke                       -> passed, reproducible
uv run --project backend ruff check backend                                     -> All checks passed
uv run --project backend mypy --config-file backend/pyproject.toml backend/src  -> no issues in 54 files
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/e3-litsearch
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/pilot
```
