# E3 evidence — the four retrieval baselines on LitSearch's own corpus

Date: 2026-10-01
Task: E3 of the evaluation benchmarks plan
Status: **done.** All four baselines were run on one frozen snapshot, development (359 queries)
and validation (120), on a GPU no other process was using. The pre-registered rule
promotes hybrid, then reranking. The locked test split has not been run.
Host: WSL2 Linux, 12 cores / 23 GB RAM, RTX 3080 Laptop 16 GB; torch 2.14.0+cu130.
Benchmark: `princeton-nlp/LitSearch` at `9573fb284a1026c998df47024b888a163f0f0e25`.
Code: branch `phuc` from `07ebd3a`, plus this task's commit.

The generated report, with every table, interval and slice, is
[`retrieval/e3-litsearch/report.md`](retrieval/e3-litsearch/report.md). Its analysis
section is the interpretation. This file records how the evidence was produced.

## What exists

| Piece | Where |
|---|---|
| `write_litsearch_snapshot`: `corpus_clean` as a schema-2 snapshot keyed by `uuid5(corpusid)`; `local_paths` verifies the pinned shards offline | `evaluation/litsearch.py` |
| `eval litsearch-snapshot` | `cli.py` |
| Scoring by corpusid (`dataset.labels`), paper text from the release's snapshot (`corpus.papers`), the specificity slice, injectable models, a GPU check before every run | `evaluation/retrieval.py`, `evaluation/report.py` |
| The experiment: four baselines, P2.5's decision rule unchanged | `configs/experiments/e3-litsearch.yaml` |
| Acceptance tests against the test services | `tests/integration/test_eval_runner.py` |

## The corpus as its own release

```text
eval litsearch-snapshot
  -> ${DATA_DIR}/exports/litsearch-v1/: 64,183 papers, chunks table empty, 11 s
  -> 574 distinct gold papers, 0 missing from the corpus; papers digest 0b985bec…b9ed
search build-index --manifest litsearch-v1/manifest.json --release litsearch-v1 --collections papers
  -> dev_paper_abstracts_litsearch-v1: 64,183 points, 1,024-d, status green, 525 MB
  -> digest equal to the snapshot's; 270 documents cut at 1,024 tokens
  -> BM25 statistics: 107,953 terms, mean length 137.3
```

Each document is title plus abstract, as the benchmark packages it; no full-text
configuration was run. Qdrant keys points by UUID, so a document's id is
`uuid5(LITSEARCH_NAMESPACE, "litsearch:corpusid:<id>")`. Gold labels map through the same
function and need no lookup table. The snapshot marks every row
`redistribution: unknown` and lives under `DATA_DIR` only: LitSearch declares no license.
`litsearch-v1` is staged beside the live release and never activated. Its chunk collection
does not exist, and serving is unaffected.

The encode took 24 minutes rather than the roughly 10 that BGE-M3's own paper build
implies. Another project was training a cross-encoder on the same GPU at the time;
nothing about the vectors depends on that.

## The runs, and the ones set aside

This host's GPU is shared. During E3, another project ran two training jobs back to back:
cross-encoder distillation, then cross-encoder training. Every run now samples the GPU for
5 s before loading a model and lists every process seen while it runs. A watcher started a
split only after two minutes with no other process on the GPU, then discarded any run that
shared it after all.

| Attempt | Outcome |
|---|---|
| development and validation, first | another process on the GPU throughout; set aside |
| development, second | a training job started as it began; set aside |
| **development, third** | clean: 0% before, no other process during |
| **validation, second** | clean: 0% before, no other process during |

The set-aside attempts lost 60, 183 and 31 queries to the reranker's 1.5 s deadline; those
queries fell back to fused order. Every other query in them — every BM25, dense and hybrid
query, and every rerank that met its deadline — ranked identically to the clean runs.

## Results on validation (120 queries)

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Judged@10 | p50 | p95 |
|---|---|---|---|---|---|---|---|
| `bm25` | 0.499 | 0.649 | 0.360 [0.288, 0.431] | 0.319 | 0.051 | 6 ms | 8 ms |
| `dense` | 0.526 | 0.625 | 0.385 [0.306, 0.457] | 0.341 | 0.054 | 25 ms | 35 ms |
| `hybrid` | 0.536 | 0.696 | 0.415 [0.339, 0.492] | 0.377 | 0.055 | 28 ms | 43 ms |
| `hybrid_rerank` | 0.592 | 0.696 | 0.469 [0.390, 0.545] | 0.434 | 0.060 | 503 ms | 812 ms |

No query failed and none was degraded, in any variant, on either split.

**Decision under the pre-registered rule:** `bm25` → `hybrid` promoted, +0.055
[+0.001, +0.104]; `hybrid` → `hybrid_rerank` promoted, +0.054 [+0.007, +0.101], with p95
0.81 s against 3 s. The selected configuration is `hybrid_rerank`. P2.5's in-domain run
retained BM25 for lack of power on 52 queries and points the same way. The
[analysis](retrieval/e3-litsearch/analysis.md) compares the two and slices by query set
and specificity.

## Acceptance cases

| Case | Evidence |
|---|---|
| A run against the mini fixture produces a manifest with every required field | `test_a_run_writes_a_complete_manifest_scored_by_corpusid`; `validate_manifest` runs on every real run before anything is written |
| Rerunning with the same seed and revisions reproduces the metrics exactly | `test_a_rerun_reproduces_every_metric_and_ranking`; on the real corpus, every query that met its deadline ranked identically across all five attempts |
| A deliberately broken baseline surfaces as a counted failure, not as a zero | `test_a_broken_baseline_is_a_counted_failure_not_a_quiet_zero`: an embedder the release was not built with fails all four queries as `candidates_unavailable`, counted by code and present in every per-query row |
| Report Recall@50, MRR@10, nDCG@10, judged coverage and p50/p95, sliced by `query_set` and `specificity` | the generated report |
| Bootstrap with 1,000 resamples | every interval: query families, seed 42, paired across variants |

The test file also covers the snapshot's id rule and gold count, a corrupted shard and a
duplicate corpusid being refused, and the test split staying locked.

## Deviations from the plan, and why

- **No `runner.py`, and no second `eval retrieval` interface.** P2.5 had already built the
  harness this task describes, so E3 is an experiment config on it (recorded in the
  benchmarks plan, 2026-09-30). `eval retrieval --config configs/experiments/e3-litsearch.yaml
  --split <split>` replaces the plan's `--benchmark litsearch --corpus litsearch --baseline`
  flags. All four baselines run together, keeping them on one manifest, as G2 asks.
- **The plan's manifest fields map onto the harness's.** The benchmark revision travels as
  `dataset.sources` (`litsearch@9573fb28…`) beside the dataset digest. For a packaged corpus
  the parser and chunker versions read `none: packaged title and abstract`. Query errors
  are counted per variant under `failures`.
- **The splits are `development` and `validation`**, E2's names for what the plan calls
  `dev` and `val`.

## Commands and results

```text
uv run --project backend ruff check backend                                     -> All checks passed
uv run --project backend mypy --config-file backend/pyproject.toml backend/src  -> no issues in 53 files
uv run --env-file .env.test --project backend pytest backend/tests/integration/test_eval_runner.py -q
                                                                                -> 7 passed
uv run --env-file .env.test --project backend pytest backend/tests -q           -> 455 passed
uv run --project backend pytest backend/tests -m "not integration" -q           -> 304 passed
uv run --project backend python -m copilot.cli eval smoke                       -> passed, reproducible
```
