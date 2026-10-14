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
no service, model or GPU. Per variant it reports the share of gold papers returned within
the depth, pooled but cut, or never pooled; the gold rank distribution; and, for one focus variant, the query ids
that lost their gold, per slice. It refuses the test split.

## Producing a run

```bash
uv run --env-file .env --project backend python -m copilot.cli eval retrieval \
  --config configs/experiments/retrieval.yaml --split validation --out reports/retrieval/sections
uv run --project backend python -m copilot.cli eval report --out reports/retrieval/sections
uv run --project backend python -m copilot.cli eval compare \
  --baseline OLD/validation/metrics.json --candidate NEW/validation/metrics.json
uv run --project backend python -m copilot.cli eval smoke
uv run --project backend python -m copilot.cli eval gaps --run reports/retrieval/orb-text
```

A variant may override four search knobs, and only these: `candidates.per_branch` for any
mode, and `rerank.depth`, `deadlines_seconds.rerank` and `rerank.text` for `hybrid_rerank`
(the first three from P2.6; `rerank.text` added 2026-10-08, one of `abstract`, `chunk`,
`title_chunk`, and refused unless the variant's candidates come through chunks). They go
in a `search:` block shaped like `configs/search.yaml`. Each variant's effective settings and
their digest are in the manifest (`variants.<name>.search`, `search_sha256`), and `eval
compare` lists any variant whose settings differ between the two runs. A decision block may
set `require_undegraded: true`; a step is then promoted only if neither side degraded a
single query. It may also list `guards`, each a metric with a `max_drop`: a step is promoted
only if every guard's paired interval rules out losing more than that, so a rule whose
primary is a recall metric cannot promote a candidate that loses the top of the page.

A variant may also say where its first stage finds candidates with `candidates:` (P2.6
step 4): `papers`, the title-and-abstract collection; `chunks`, the release's chunk
collection with each paper scored by its best evidence chunk (references excluded); or
`both`, each branch's RRF of its paper list and its chunk list. Unset, a variant follows
`candidates.source` in `configs/search.yaml` — `chunks` since 2026-10-06 — except over a
`papers: snapshot` corpus, which was never chunked and stays at paper level. The resolved
source is recorded per variant (`variants.<name>.candidates`); a source other than `papers`
also appears in the variant's search settings as `candidates_source` and changes its digest,
so runs recorded before the decision still read as the same search. A run naming `chunks` or
`both` is refused before anything loads when the release has no chunk collection
(`chunks_not_built`). The reranker scores titles and abstracts unless the variant sets
`rerank.text` to `chunk` or `title_chunk`, in which case it scores the chunk each paper was
found by; `abstract` leaves every recorded digest unchanged, the other two change it.

A `hybrid_rerank_llm` variant names the hosted model it uses with `llm:`, a key in
`configs/llm.yaml`; no other mode may. Such a run needs `--max-spend-usd`. It is refused before
anything loads when its worst case passes that budget, the worst case being every query's
whole head at 300 words and the model's whole output budget. It stops, writing nothing, once
the spend ledger shows its real spend past the budget. Every call is billed to the variant's
own ledger run. The manifest records what each LLM variant was and what it cost
(`variants.<name>.llm`: model, served models, prompt digest, calls, cache hits, billed tokens
and cost), and the total goes in `cost.metered_usd`. Answers are cached under
`${DATA_DIR}/cache/llm/`, so a rerun is free and identical. Warm-up never calls the LLM, and
`eval compare` lists variants whose served model or prompt changed (`llm_changed`).

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
- **A benchmark read as a whole uses the `retrieval` split** (Open RAG Bench's text slice):
  every query in one split, no decision ever, warm-up from the split's own last queries,
  which the manifest says; `eval gaps` diagnoses it like development. `eval orb-sections
  --run DIR --database-url …copilot_orb` then scores each variant's rank-1 hit against the
  gold section, writing `sections.md` and `sections.json` beside the run, with the share of
  queries whose section mapped at all.
- **Tune on development, choose on validation, lock test.** Only a validation run
  produces a decision. The test split runs only with `--locked-test`, for a release
  decision, and no decision is ever computed from it.
- **Unlabelled is unjudged.** Every score travels with judged coverage. ORB and S1 label
  the one paper each question was written from, so recall here is recall against that paper,
  not against every paper that would answer the query.

## What is not in git

Question text: ORB's is CC-BY-NC-4.0 (`${DATA_DIR}/benchmarks/orb/`) and S1's is derived from
corpus papers (`${DATA_DIR}/benchmarks/sections-v1/`), so both stay under `DATA_DIR` and are
re-attached at run time. LitSearch, the first benchmark, was retired on 2026-10-14; its
experiment folders were deleted, and git history keeps them. The per-query
files carry query ids, never query text.
