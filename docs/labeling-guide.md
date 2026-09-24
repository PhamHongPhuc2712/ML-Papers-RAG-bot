# Retrieval labeling guide

What the evaluation dataset contains, what its labels mean, and what they cannot
support. Read this before quoting a retrieval number.

## Where the labels come from

They are not ours. The 597 queries and their gold papers are LitSearch
(Ajith et al., EMNLP 2024), pinned at revision `9573fb284a1026c998df47024b888a163f0f0e25`.
Spec §11 originally called for 150 hand-judged query families; using an
expert-checked benchmark that already exists is both cheaper and comparable with
published work. The decision and its cost are recorded in the progress tracker.

## The files

`data/fixtures/retrieval/`, one JSON object per line.

**The repository copy carries no query text.** LitSearch declares no license, so
its strings stay under `${DATA_DIR}/benchmarks/`, while the split assignment —
the thing that must be frozen, reviewable and diffable — lives in git.
`datasets.hydrate` re-attaches the text at run time from the pinned parquet, and
refuses rather than leaving a query empty.

| File | One row per | Key fields |
|---|---|---|
| `queries.jsonl` | query | `query_id`, `query_set`, `workflow`, `specificity`, `quality`, `in_domain`, `source`, `author` (`query` omitted in the repository copy) |
| `qrels.jsonl` | gold paper | `query_id`, `corpusid`, `grade`, `paper_id` |
| `splits.jsonl` | query | `query_id`, `family_id`, `split` |

`qrels.jsonl` carries two identifiers on purpose. `corpusid` is LitSearch's own
id, used when the run searches LitSearch's 64,183-document corpus. `paper_id` is
ours, present only where matching found the paper, used when the run searches
our corpus. One judgement, two runs.

## What a grade means

**Grades are binary.** `grade: 1` marks the paper the query was built from.
There is no 2 or 3, and no 0 rows at all.

This is the most important limitation in this document. LitSearch labels *one*
paper per query out of 64,183, and our own corpus holds many further papers that
would answer the same question perfectly well. Every unlabelled result is
**unjudged**, not irrelevant.

Consequences you must not paper over:

- **Recall is recall against the judged pool.** Never describe it as recall
  against everything relevant.
- **nDCG over binary gains carries less information** than the graded 0–3 scale
  spec §11 describes. It is reported for comparability and should not be read as
  a fine-grained quality signal.
- **Report judged coverage beside every score.** `evaluation.metrics.evaluate`
  returns it as a required field precisely so it cannot be dropped.
- A new system's unjudged results trigger fresh blind assessment, not a silent
  zero.

## Query sets and workflows

| `query_set` | Count | `workflow` | How it was built |
|---|---:|---|---|
| `manual_acl` | 155 | `known_item` | Author-written about their own recent paper |
| `manual_iclr` | 91 | `known_item` | Author-written, ICLR |
| `inline_acl` | 98 | `related_work` | GPT-4-generated from a citation sentence |
| `inline_nonacl` | 253 | `related_work` | GPT-4-generated from a citation sentence |

`workflow` is derived from construction, not judged per query. Spec §11 also
names a **keeping up** workflow, which LitSearch does not exercise at all — that
gap is what the hand-authored in-domain queries in E4 exist to fill.

## `in_domain`, and why partial does not count

A query is `in_domain` only when **every** one of its gold papers is in our
corpus. A query with two of three golds present is not in-domain and is excluded
from in-domain recall.

This is deliberate. Recall computed against a partly-present gold set is bounded
by our coverage rather than by our retrieval, and it fails downward silently —
the system looks worse and nothing says why.

Coverage is nearly binary by query set: the author-written sets are ~99% covered
and the citation-derived sets are ~1%, because an inline query's gold paper is
the work being *cited* and therefore predates our 2023–2026 window. Measured
figures are in `reports/e1-litsearch-coverage.md`.

## Splits

60 / 20 / 20 development / validation / test, stratified by `query_set` with
seed 42, assigned from sorted query ids so the result never depends on row
order.

**The test split is scored, never tuned against.** Prompt changes, ranking
changes and parameter sweeps look only at development and validation.

`family_id` equals `query_id`. LitSearch has no rewrite families; inventing them
would fake a grouping the data does not have. The validator still enforces that a
family cannot span two splits, so the rule is in place if a future dataset has
real families.

## Annotations we do not stratify on

`specificity` (0 for 155 queries, 1 for 442) and `quality` (1 for 294, 2 for 303)
are described on the dataset card only as "specificity and quality annotations",
with no scale definition in the published artifacts. They travel with each record
so results can be sliced by them, but splits are stratified on `query_set` alone
rather than on semantics we would be guessing at.

## Validating a dataset

```bash
uv run --project backend python -m copilot.cli eval validate-dataset --path data/fixtures/retrieval
```

It refuses a family spanning two splits, a qrel for an unknown query, a query
with no qrel, a query with no split, a duplicate query id and a negative grade.
On success it prints the dataset's shape, so a passing run says what it accepted
rather than only that it accepted something.

## Adding hand-authored queries later

E4 adds 30 in-domain queries covering topic discovery, related work and keeping
up. Those carry graded 0–3 judgements, a real `author`, and a `source` that is
not `litsearch@...`. Pool candidates blindly across baselines before judging,
record the rationale, and freeze family assignments before any tuning.
