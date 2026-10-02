# Retrieval experiments

Each directory here is one experiment config run over one or more frozen splits. Everything
in it is written by the harness (`backend/src/copilot/evaluation/retrieval.py`), except
`analysis.md`, which is the human reading of the numbers.

```text
reports/retrieval/<experiment>/
  report.md                      rendered by `eval report` from the files below
  analysis.md                    written by hand; included in report.md verbatim
  gaps.md, gaps.json             rendered by `eval gaps`: where the gold paper is lost (P2.6)
  <split>/manifest.json          what produced the run: code, corpus, dataset, models, hardware
  <split>/metrics.json           per-variant summaries, paired comparisons, the decision
  <split>/per_query.parquet      one row per variant and query: ranked ids, labels, scores, timing
```

`eval gaps` reads only the recorded `per_query.parquet` files, so it reruns nothing and needs
no service, model or GPU. Per variant it reports recall at LitSearch's published cutoffs
(R@20 broad, R@5 and R@20 specific, set beside the paper's Table 3 when the run is over
LitSearch's own corpus); the share of gold papers returned within the depth, pooled but
cut, or never pooled; the gold rank distribution; and, for one focus variant, the query ids
that lost their gold, per slice. It refuses the test split.

## Producing a run

```bash
uv run --env-file .env --project backend python -m copilot.cli eval retrieval \
  --config configs/experiments/retrieval.yaml --split validation --out reports/retrieval/pilot
uv run --project backend python -m copilot.cli eval report --out reports/retrieval/pilot
uv run --project backend python -m copilot.cli eval compare \
  --baseline OLD/validation/metrics.json --candidate NEW/validation/metrics.json
uv run --project backend python -m copilot.cli eval smoke
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/e3-litsearch
```

A variant may override three search knobs, and only these (P2.6): `candidates.per_branch`
for any mode, and `rerank.depth` and `deadlines_seconds.rerank` for `hybrid_rerank`. They go
in a `search:` block shaped like `configs/search.yaml`. Each variant's effective settings and
their digest are in the manifest (`variants.<name>.search`, `search_sha256`), and `eval
compare` lists any variant whose settings differ between the two runs. A decision block may
set `require_undegraded: true`; a step is then promoted only if neither side degraded a
single query.

`eval retrieval` needs the core services, the `models` dependency group and the pinned
models under `DATA_DIR`. `eval smoke` needs none of them: it ranks the synthetic fixture in
`data/fixtures/retrieval-smoke/` with the exact BM25 oracle and the fixture models, and
compares against frozen outputs. That makes it the check CI can run.

## Rules the harness enforces

- **A run must say what produced it.** A manifest missing any version — git commit,
  corpus release and manifest digest, dataset digest, split, parser and chunker versions,
  each variant's embedding and reranker identity, search config digest, seed, hardware,
  timing method — is rejected before anything is written. A run from uncommitted code
  records the digest of the uncommitted change.
- **Runs over different corpora, datasets or splits are never compared.** `eval compare`
  refuses them; within one corpus and split it flags a drop of more than 0.03 absolute in
  Recall@10 or nDCG@10.
- **A failed query stays in every denominator.** It scores zero, its time counts toward
  latency, and the failure is counted by code.
- **Tune on development, choose on validation, lock test.** Only a validation run
  produces a decision. The test split runs only with `--locked-test`, for a release
  decision, and no decision is ever computed from it.
- **Unlabelled is unjudged.** Every score travels with judged coverage. LitSearch labels
  the one paper each query was written from, so recall here is recall against that paper,
  not against every paper that would answer the query.

## What is not in git

LitSearch's query text: it declares no license, so it stays under
`${DATA_DIR}/benchmarks/litsearch-dataset/` and is re-attached at run time. The per-query
files carry query ids, never query text.
