# P2.6 evidence — the retrieval gaps E3 and P2.5 exposed

Date: 2026-10-01, completed 2026-10-06
Task: P2.6 of the retrieval plan (urgent, before the locked test split and before P3.1)
Status: **done.** Step 1 is the offline analysis. Step 2 is the ceiling run. Step 3, the
depth-100 rerank, was pre-registered in `a18a991` and not promoted. Step 4 read the misses,
ruled out a larger pool, and found the one lever that moves first-stage recall on our
corpus: candidates found through the chunk collection, pre-registered in `8654f22` and
**promoted on validation** (Recall@50 +0.135 [+0.058, +0.231], nDCG@10 +0.060 [+0.025,
+0.097]). It is adopted as `candidates.source: chunks` in `configs/search.yaml`, with spec
§7 amended. The GPU policy is recorded, and the final E3 validation re-run reproduces the
recorded E3 to four decimals. The locked test split has **not** been run.
Host: WSL2 Linux, 12 cores / 23 GB RAM, RTX 3080 Laptop 16 GB.

## What changed in the harness

| Piece | Where | Why |
|---|---|---|
| `eval gaps`: recall at LitSearch's published cutoffs, where each gold paper is lost (returned / pooled but cut / never pooled), gold rank bands, missed query ids per slice. Reads only recorded `per_query.parquet`; refuses the test split | `evaluation/gaps.py`, `cli.py`, `tests/unit/test_gaps.py` | Step 1, and the gap table must be regenerated after the final run |
| A variant may override `candidates.per_branch` (any mode), `rerank.depth` and `deadlines_seconds.rerank` (`hybrid_rerank` only); anything else is refused when the config loads. Every variant's effective settings and their digest go in the manifest | `evaluation/retrieval.py` | The plan's prerequisite for steps 2–4 |
| `eval compare` lists variants whose search settings differ between the two runs | `evaluation/regression.py` | Runs that searched differently are never compared as equal |
| `require_undegraded` decision rule, opt-in: a step is promoted only if neither side degraded a query | `evaluation/retrieval.py`, `evaluation/report.py` | Encodes "any rerank timeout on a clean GPU sets the run aside" |
| Candidates through chunks: both retrievers take `group_papers` over the chunk collection (Qdrant's grouped query, each paper scored by its best evidence chunk, references excluded) and `UnionRetriever` fuses a paper list with a chunk list by RRF; `candidates.source` in `configs/search.yaml`; a variant's `candidates:` field, resolved and recorded per variant, in its search digest when not `papers`; refused when the release has no chunk collection | `search/chunks.py`, `search/dense.py`, `search/sparse.py`, `search/service.py`, `search/api.py`, `evaluation/retrieval.py` | Step 4's lever on our corpus, then its adoption |
| `guards` in a decision rule, opt-in: a step is promoted only if every guard metric's paired interval rules out a drop beyond its `max_drop` | `evaluation/retrieval.py`, `evaluation/report.py` | A first-stage rule whose primary is recall must not promote a candidate that loses the top of the page |

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

## Step 4 — first-stage recall

### The misses, read one by one (offline)

The 48 gold papers that `hybrid` never pooled on E3 development (every query whose
`candidate_recall` is 0 in the recorded run) were read beside their queries, with the
query text from `DATA_DIR` and the gold's title and abstract from the `litsearch-v1`
snapshot. Nothing of either is reproduced here. A one-off script also counted, for every
gold paper of the split, the share of the query's content words (stop words removed, the
P2.1 tokenizer) that occur in the gold's title and abstract.

| Gold papers, `hybrid`, E3 development | n | Query words found in the gold text, median |
|---|---|---|
| In the returned top 50 | 297 | 0.41 |
| Pooled but cut at 50 | 40 | 0.24 |
| Never pooled | 48 | 0.18 |

Of the never-pooled golds, 98% share under half of the query's content words with the
gold's title and abstract, against 67% of the found ones. Reading them, four kinds:

1. **The query describes something only the body says** — a hyperparameter recipe, a
   binarization step, an evaluation protocol, a tool, a component the paper adds to a
   baseline. The abstract names none of it. This is the largest group, and it is the
   citation-derived (`inline_*`) sets almost entirely: the query paraphrases the citing
   sentence, and the citing sentence cites the paper for a detail. On LitSearch's corpus,
   which is title and abstract only, no first stage over that text can find these; on our
   corpus the body is indexed, which is what the chunk-level variants below test.
2. **The gold is a general paper cited for a side point** — a sentence-embedding paper for
   a question about matching to a knowledge base, an alignment model for a question about
   simultaneous translation, a retrieval model for a question about distillation. The
   query and the gold are both reasonable; the label is the loose part. No ranking fixes
   these, and they are a reason the judged-coverage caveat matters.
3. **Vocabulary mismatch on a findable paper** — the query says in plain words what the
   title says in the paper's own terms (a safety-constrained world model, iterative
   prompting for ambiguous questions, a Pareto-frontier result the query calls mutual
   learning). These are the author-written misses, and the ones a stronger dense model
   is for. BGE-M3 trails GritLM-7B by 5–16 points at LitSearch's cutoffs (step 1), and
   nothing cheaper than a stronger model addresses this group.
4. **Empty gold text.** Two never-pooled golds have no title and no abstract in LitSearch's
   `corpus_clean`, and a third has no title. They are unfindable by any system over that
   corpus and should be read as a floor on every published number too.

### A larger pool, and how deep the misses are (E3, development)

Run `e3-first-stage-development-20261006T060217Z`, clean: GPU idle beforehand (p50 0%,
1,257 MiB held by an idle Windows-side process, as in every clean run before), no other
compute process during it, 0 degraded and 0 failed queries. Config
`configs/experiments/e3-first-stage.yaml`; development only, no decision block.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Gold in pool | p95 |
|---|---|---|---|---|---|---|
| `hybrid_rerank` (shipped) | 0.694 | 0.784 | 0.556 | 0.517 | 86.0% | 755 ms |
| `hybrid_rerank_pool300` | 0.680 | 0.799 | 0.550 | 0.513 | 91.0% | 758 ms |
| `bm25_deep1000` | 0.547 | 0.678 | 0.411 | 0.373 | 88.4% | 11 ms |
| `dense_deep1000` | 0.579 | 0.727 | 0.440 | 0.401 | 89.9% | 61 ms |

- **The shipped row reproduces E3's recorded development run** (nDCG@10 0.556; gold
  returned / cut / never pooled 78.4 / 7.6 / 14.0, to the decimal), after the chunk-level
  code and the candidate-source field were added. Nothing moved for the paper-level path.
- **A 300-per-branch pool is not a lever, by the same mechanism as depth 100.** Paired
  against the shipped stack: nDCG@10 −0.006 [−0.016, +0.003] with 8 wins and 16 losses,
  Recall@10 −0.014 [−0.033, +0.003], Recall@50 +0.015 [−0.006, +0.036]. The pool holds
  5 more points of gold (91.0% against 86.0%), the reranked top 50 gains 1.5, and the
  top 10 loses a little: the extra candidates come from deep in one branch, RRF ranks
  them low, and the few that reach the reranker displace as much as they add. It costs
  nothing in latency (p95 758 against 755 ms), so it is not promoted on latency grounds
  either; it is simply not better. No validation run is owed to a change that loses on
  development.
- **Reranking far deeper would not reach the rest.** Run alone to 1,000 candidates, BM25
  holds 88.4% of gold papers somewhere in its list and dense 89.9%. Against the 100-deep
  union's 86.0%, ten times the depth in either branch recovers at most 4 points, and
  10–12% of gold papers are beyond rank 1,000 of a branch. Those are the queries read in
  the section above: the paper's text does not say what the query says. On this corpus
  they are a model problem, not a depth problem.

### Candidates through chunks, on our corpus (development)

The lever the misses point at exists only on our corpus, whose chunk collection holds
every paper's body. A paper found through its chunks is scored by its best evidence
chunk (Qdrant's grouped query over `paper_chunks_<release>`, references excluded);
`chunks` replaces both branches' collection, `both` fuses each branch's paper list with
its chunk list by RRF before the branches are fused. The reranker is unchanged and still
scores titles and abstracts. This is also E4's chunk-level retrieval on the in-domain
slice, shared with that task as its plan asks.

Run `m2-first-stage-development-20261006T062016Z`, clean: GPU idle beforehand (p50 0%),
only the idle Windows-side process seen during it, 0 degraded queries. A first attempt
(`...-20261006T061147Z`, set aside) produced the same three reranked rows to the third
decimal. Config `configs/experiments/m2-first-stage.yaml`; in-domain development, 150
queries.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Gold in pool | p95 |
|---|---|---|---|---|---|---|
| `hybrid_rerank` (shipped) | 0.740 | 0.807 | 0.585 | 0.536 | 90.0% | 621 ms |
| `hybrid_rerank_chunks` | 0.773 | 0.840 | 0.606 | 0.553 | 91.3% | 640 ms |
| `hybrid_rerank_both` | 0.767 | **0.880** | 0.597 | 0.543 | 92.7% | 638 ms |
| `bm25_chunks` (branch alone) | 0.667 | 0.800 | 0.517 | 0.470 | 85.3% | 180 ms |
| `dense_chunks` (branch alone) | 0.573 | 0.720 | 0.432 | 0.387 | 79.3% | 337 ms |

Paired against the shipped stack, 95% intervals over query families:

| Candidate | nDCG@10 | Recall@10 | Recall@50 |
|---|---|---|---|
| `hybrid_rerank_both` | +0.012 [−0.006, +0.032], 12 / 9 | +0.027 [+0.000, +0.060] | **+0.073 [+0.027, +0.127]**, 13 wins / 2 losses |
| `hybrid_rerank_chunks` | +0.020 [−0.006, +0.046], 24 / 10 | +0.033 [−0.007, +0.080] | +0.033 [−0.020, +0.093] |

- **This is the first lever that moves first-stage recall.** With `both`, gold papers
  returned in the top 50 rise from 80.7% to 88.0%; never-pooled gold falls from 10.0% to
  7.3% and gold cut at 50 from 9.3% to 4.7%. The slice that lost most to the depth-50 cut
  gains most: `manual_iclr` goes from 66.7% to 81.5% returned. Recall@50 is also what a
  page can show, since `hybrid_rerank` serves only its reranked head.
- **The top 10 moves less, and 150 queries cannot confirm it.** Both candidates improve
  nDCG@10 and MRR@10 with more wins than losses, and both intervals include zero.
- **It costs 17–19 ms at p95** (two more Qdrant queries per request), nothing against the
  3 s budget. Alone, the dense chunk branch timed out at its 1 s branch deadline on 2 of
  150 queries (the first attempt: 2 dense and 1 BM25); inside the hybrid modes no query
  degraded in either attempt. A branch over 3.4 M on-disk vectors sits closer to its
  deadline than one over 86k in RAM, and the diagnostic reports that rather than hiding it.
- **`bm25_chunks` alone nearly matches the shipped hybrid at Recall@50** (0.800 against
  0.807): a paper's body says most of what its abstract says and more. `dense_chunks`
  alone is the weakest branch, as it was at paper level.

**Pre-registered rule, written after this run and before any validation number.** The
`decision` block in `m2-first-stage.yaml`, committed before the validation run: primary
metric Recall@50, because a first stage is judged by what it puts in front of the
reranker and what the page can show; steps `hybrid_rerank` → `hybrid_rerank_chunks` →
`hybrid_rerank_both` in cost order; a step is promoted only if its paired Recall@50
interval lies wholly above zero, its nDCG@10 interval rules out a loss beyond 0.03 (a
*guard*, a new opt-in rule in the harness, so the top of the page is checked by the
machine rather than by prose), its p95 fits 3 s, and neither side degraded a query. The
52 in-domain validation queries are few, and the rule was chosen knowing that.

### Validation: the rule promotes `hybrid_rerank_chunks`

Run `m2-first-stage-validation-20261006T062929Z`, clean: GPU idle beforehand, only the
idle Windows-side process seen, 0 failed and 0 degraded queries. The rule was committed in
`8654f22` before this run existed. In-domain validation, 52 queries.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Gold in pool | p95 |
|---|---|---|---|---|---|---|
| `hybrid_rerank` (paper level, as shipped) | 0.635 | 0.769 | 0.516 | 0.477 | 84.6% | 699 ms |
| `hybrid_rerank_chunks` | **0.712** | **0.904** | **0.576** | **0.535** | 96.2% | 597 ms |
| `hybrid_rerank_both` | 0.615 | 0.846 | 0.514 | 0.481 | 94.2% | 684 ms |
| `bm25_chunks` (branch alone) | 0.615 | 0.885 | 0.506 | 0.469 | 92.3% | 101 ms |
| `dense_chunks` (branch alone) | 0.538 | 0.788 | 0.440 | 0.409 | 82.7% | 164 ms |

| Step | Recall@50 (primary) | Guard: nDCG@10 interval low | p95 | Promoted |
|---|---|---|---|---|
| `hybrid_rerank` → `hybrid_rerank_chunks` | **+0.135 [+0.058, +0.231]**, 7 wins / 0 losses | +0.025 (holds; the gain is +0.060 [+0.025, +0.097], 10 / 1) | 0.60 s | **yes** |
| `hybrid_rerank_chunks` → `hybrid_rerank_both` | −0.058 [−0.135, +0.000], 0 / 3 | −0.101 (fails; −0.062 [−0.101, −0.026], 0 / 11) | 0.68 s | no |

**Decision: both branches search the chunk collection.** `hybrid_rerank_chunks` is chosen;
`both` is not promoted over it. Where the gold paper is lost under the chosen stage, on
validation: returned in the top 50 76.9% → 92.3%, cut at 50 7.7% → 1.9%, never pooled
15.4% → 5.8% (`reports/retrieval/m2-first-stage/gaps.md`). On validation every metric
cleared zero, which 150 development queries had not managed for the top 10; the
intervals are wide (52 queries), and the ordering of `chunks` against `both` reversed
between the splits, so the rule chose between them on evidence the splits do not agree
on. What both splits agree on is the step that matters: chunk-level candidates beat
title-and-abstract candidates on Recall@50 with intervals above zero on both, and never
lose the top of the page.

Observations recorded for later tasks, not decisions: `dense` alone through chunks is
weaker than `dense` alone over abstracts on development (Recall@50 0.720 against 0.767),
so the single-branch `dense` mode gets a worse first stage than before while the served
default improves; and the dense chunk branch alone touched its 1 s deadline on 1–2 of 150
development queries, never inside the hybrid modes and never on validation.

**Adopted** (spec §7 amended in the same commit): `candidates.source: chunks` in
`configs/search.yaml`, read by the search service and the API; the cache identity names
the source, so no ordering cached under the paper collection is reused; a release
without a chunk collection — LitSearch's — stays at paper level, and the harness records
the resolved source per variant. P2.5's two BGE-small variants name `candidates: papers`
explicitly, because that release holds a paper collection only. P2.5's in-domain slice was
re-measured under the final configuration by this very run and its development
counterpart (`hybrid_rerank` there is the previous configuration, `hybrid_rerank_chunks`
the new one), so no separate re-run of `retrieval.yaml` is owed.

## GPU contention policy

**Option (a), no code.** Evaluation runs and demos happen only on an otherwise idle GPU.
Every run samples the GPU for 5 s before loading a model and records
`timing.gpu.before_run.busy`, lists the compute processes seen, and counts degraded
queries; a run that was busy beforehand, shared the GPU with another compute process, or
degraded a query is set aside and redone. The search API already returns `warnings`
(`rerank_timeout` and the rest) on every page, so a contended demo is visible rather than
silently worse. Recorded in the tracker on 2026-10-05, when another project's
cross-encoder training held 16 of 16 GB for over five hours and the step-4 runs waited;
two further runs of its pipeline interrupted the first attempt, which was set aside.

## Final E3 validation, under the final configuration

LitSearch's release has no chunk collection, so its final configuration is the paper-level
stack E3 chose, and this re-run is the reproduction check the plan asks for. Run
`e3-litsearch-validation-20261006T064454Z` (`reports/retrieval/e3-final/`), clean: GPU
idle beforehand, only the idle Windows-side process seen, 0 failed and 0 degraded
queries. A first attempt lost one BM25 query to a 1.25 s branch timeout and was set aside.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | p95 |
|---|---|---|---|---|---|
| `bm25` | 0.4986 | 0.6486 | 0.3598 | 0.3188 | 7 ms |
| `dense` | 0.5264 | 0.6250 | 0.3854 | 0.3408 | 65 ms |
| `hybrid` | 0.5361 | 0.6958 | 0.4148 | 0.3774 | 34 ms |
| `hybrid_rerank` | 0.5917 | 0.6958 | 0.4688 | 0.4339 | 815 ms |

`eval compare` against the recorded E3 validation run: every gated metric of every variant
equal to four decimals, `search_changed` empty. The pre-registered rule decides as it did:
hybrid over BM25 +0.055 [+0.001, +0.104], reranking over hybrid +0.054 [+0.007, +0.101],
rerank-stage p95 782 ms. The gap table, regenerated from this run:

| Run | Queries | Returned in top 50 | In pool, cut at 50 | Not in pool |
|---|---|---|---|---|
| E3 validation, `hybrid_rerank` (paper level; LitSearch has no chunks) | 120 | 69.6 | 9.2 | 21.2 |
| Our corpus, validation, `hybrid_rerank` paper level (P2.5's configuration) | 52 | 76.9 | 7.7 | 15.4 |
| Our corpus, validation, `hybrid_rerank` through chunks (**adopted**) | 52 | **92.3** | **1.9** | **5.8** |
| Our corpus, development, paper level | 150 | 80.7 | 9.3 | 10.0 |
| Our corpus, development, through chunks | 150 | 84.0 | 7.3 | 8.7 |

On LitSearch's corpus the three gaps the plan opened stand where steps 1–3 left them: the
first stage misses 21% of gold on validation and no depth or pool reaches it, so the
remaining lever there is a stronger embedding model (GritLM-7B trails us by 5–16 points
the other way), which is M6 work. On our corpus, where the body of every paper is
indexed, the first-stage miss falls from 15.4% to 5.8% and the depth-50 cut from 7.7% to
1.9% on validation. The locked test split has not been run; it waits for the user's
go-ahead, on an idle GPU, once, under this configuration.

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

Step 4 and the close of the task (2026-10-05 and 2026-10-06), every measured run behind
an idle-GPU gate (four consecutive samples at or under 20% utilization, under 2.5 GB used,
no Linux compute process) and judged afterwards by its manifest:

```text
uv run --env-file .env.test --project backend pytest backend/tests -q           -> 573 passed
uv run --project backend pytest backend/tests -m "not integration" -q           -> 391 passed
uv run --project backend python -m copilot.cli eval smoke                       -> passed, reproducible
uv run --project backend ruff check backend                                     -> All checks passed
uv run --project backend mypy --config-file backend/pyproject.toml backend/src  -> no issues in 58 files
uv run --env-file .env --project backend python -m copilot.cli eval retrieval --config configs/experiments/e3-first-stage.yaml --split development --out reports/retrieval/e3-first-stage
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/e3-first-stage --focus hybrid_rerank_pool300
uv run --env-file .env --project backend python -m copilot.cli eval retrieval --config configs/experiments/m2-first-stage.yaml --split development --out reports/retrieval/m2-first-stage
git commit 8654f22   (the pre-registered rule, before the validation run)
uv run --env-file .env --project backend python -m copilot.cli eval retrieval --config configs/experiments/m2-first-stage.yaml --split validation --out reports/retrieval/m2-first-stage
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/m2-first-stage --focus hybrid_rerank_both
uv run --env-file .env --project backend python -m copilot.cli eval retrieval --config configs/experiments/e3-litsearch.yaml --split validation --out reports/retrieval/e3-final
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/e3-final
uv run --project backend python -m copilot.cli eval compare --baseline reports/retrieval/e3-litsearch/validation/metrics.json --candidate reports/retrieval/e3-final/validation/metrics.json
```

Set aside, not reported: the first m2-first-stage development attempt (2 dense-chunk and
1 BM25-chunk branch timeouts; its three reranked rows equal the kept run's), one
e3-first-stage attempt stopped after another project's job took the GPU mid-run, and the
first e3-final attempt (one BM25 branch timeout). The full suite on 2026-10-05 had one
transient failure in the readiness health test under host load; it passed three times in
a row on rerun and in the final 573.
