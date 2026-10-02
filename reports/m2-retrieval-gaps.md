# P2.6 evidence — the retrieval gaps E3 and P2.5 exposed

Date: 2026-10-01
Task: P2.6 of the retrieval plan (urgent, before the locked test split and before P3.1)
Status: **in progress.** Steps 1–3 are done. Step 1 is the offline analysis. Step 2 is the
ceiling run. Step 3, the depth-100 rerank, was pre-registered in `a18a991` and is not
promoted on validation. Step 4 (first-stage recall), the GPU-sharing policy and the final
E3 re-run remain. The locked test split has **not** been run.
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

Run `e3-candidates-development-20261002T084151Z`, code `1467e20`, clean. The GPU was 0% busy
before the run and used only by our own dense encoding during it (p50 33%, max 54%). The
only other process listed, pid 27, was present and idle beforehand. Hybrid only, no
reranker, the 359 development queries, rankings kept 100 deep.

| Pool per branch | Recall@10 | Recall@50 | Recall@100 | nDCG@10 | Gold in top 100 | Pooled, beyond 100 | Not in pool | p95 |
|---|---|---|---|---|---|---|---|---|
| 100 (shipped) | 0.597 | 0.784 | 0.828 | 0.474 | 82.8 | 3.2 | 14.0 | 57 ms |
| 200 | 0.605 | 0.784 | 0.843 | 0.478 | 84.3 | 4.3 | 11.5 | 40 ms |
| 300 | 0.609 | 0.799 | 0.840 | 0.481 | 84.0 | 7.1 | 9.0 | 46 ms |

- **The 100-candidate row reproduces E3's recorded hybrid run to the third decimal.** That
  is Recall@10 0.597, Recall@50 0.784 and nDCG@10 0.474. Today's LLM-mode changes left the
  existing modes as they were.
- **A deeper rerank is the cheap lever.** At the shipped pool, 82.8% of gold papers sit in
  the fused top 100, against 78.4% in the top 50 the reranker sees today. Reranking 100 can
  reach 4.4 more points of gold.
- **A bigger pool finds gold the fusion then buries.** Tripling the pool cuts never-pooled
  gold from 14.0% to 9.0%. But most of what it adds lands past fused rank 100: the pooled
  share beyond 100 grows from 3.2% to 7.1%. Each added paper comes from deep in one branch
  only, so RRF ranks it low. Combined with a depth-100 rerank, a 200 pool adds at most 1.5
  points (84.3% against 82.8%).
- **What remains is a model problem.** At pool 300, 9.0% of gold papers are in neither
  branch's top 300. They are mostly broad and citation-derived queries: with pool 100,
  broad queries miss 24.6% of their gold from the pool and `inline_acl` 21.0%. A larger
  pool does not reach them; a stronger first-stage model (P2.6 step 4) might.

Development only: a diagnostic, never a choice. `reports/retrieval/e3-candidates/gaps.md`
has the slices and gold rank bands.

## Step 3 — rerank depth 100: not promoted

Pre-registered in `configs/experiments/e3-rerank-depth.yaml` and committed in `a18a991`
before either run: depth 100 with a 2.5 s rerank deadline, against the shipped depth 50 at
1.5 s. It is promoted only if:
- the paired nDCG@10 interval lies wholly above zero;
- the end-to-end p95 is ≤ 3 s;
- neither variant degrades a single query (`require_undegraded`).

Runs `e3-rerank-depth-development-20261002T084328Z` and
`e3-rerank-depth-validation-20261002T085409Z` were both clean. The GPU was idle beforehand;
no rerank timed out and no query degraded.

| Paired vs depth 50 | Development (359) | Validation (120) |
|---|---|---|
| nDCG@10 | −0.006 [−0.014, +0.002], 6 wins / 32 losses | **−0.001 [−0.009, +0.008]**, 1 / 4 |
| Recall@10 | −0.013 [−0.036, +0.008] | +0.000 [−0.025, +0.025] |
| Recall@50 | +0.023 [+0.003, +0.045] | +0.033 [−0.008, +0.075] |
| Recall@100 | +0.044 [+0.024, +0.065], 17 / 0 | +0.042 [+0.008, +0.083], 5 / 0 |
| End-to-end p95 | 750 → 1,220 ms | 851 → 1,308 ms |

**Decision: the rerank depth stays at 50.** The gain is not supported, while the run fits
the 3 s budget and has no degraded query. Reranking 100 hands the cross-encoder the 4.4
points of gold that the cut at 50 had dropped. But it also brings in distractors it ranks
above them, so the top 10 does not improve. At the top of the page the cross-encoder, not
the depth, is now the limit. `reports/retrieval/e3-rerank-depth/report.md` has the full
tables. Spec §7's "rerank at most 50" stands unchanged.

For the LLM reranking plan, the wider head is still a live option. An LLM variant may set
its own rerank depth. At 100, it would see 4.2–4.4 more points of gold than at 50.

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
