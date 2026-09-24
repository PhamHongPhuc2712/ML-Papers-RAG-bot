# M2 labels — the frozen retrieval evaluation dataset

Built 2026-09-24 from LitSearch at revision `9573fb284a1026c998df47024b888a163f0f0e25`,
matched against the finished corpus (85,729 papers in `copilot_v2`), split with seed 42.

`docs/labeling-guide.md` is the reader's guide; this is the record of what was produced.

## Shape

```
uv run --project backend python -m copilot.cli eval validate-dataset --path data/fixtures/retrieval
{"families": 597, "in_domain": 250, "qrels": 639, "queries": 597,
 "splits": {"development": 359, "test": 118, "validation": 120}}
```

| | |
|---|---:|
| Queries | 597 |
| Gold judgements | 639 (251 resolve to a paper in our corpus) |
| Families | 597 — a query is its own family |
| In-domain (every gold present) | 250 |

## Splits, stratified by query set

60 / 20 / 20 with seed 42, cut **inside each query set** rather than over the pool,
because the sets differ by a factor of 2.6 in size and by a factor of ~35 in corpus
coverage. Assignment reads sorted query ids, so it does not depend on row order.

| Query set | Development | Validation | Test | Total | In-domain |
|---|---:|---:|---:|---:|---:|
| `manual_acl` | 93 | 31 | 31 | 155 | 155 |
| `manual_iclr` | 55 | 18 | 18 | 91 | 88 |
| `inline_nonacl` | 152 | 51 | 50 | 253 | 7 |
| `inline_acl` | 59 | 20 | 19 | 98 | 0 |

The in-domain slice lands at **150 development / 52 validation / 48 test**.

**The in-domain test split is 48 queries.** That is small, and it is the number that
matters when reading an in-domain result: bootstrap intervals over 48 binary-labelled
queries are wide. E3's run over LitSearch's own corpus, where all 597 are answerable,
stays the primary G2 evidence; the in-domain slice is the in-domain signal, not the
headline.

## Judgement rules

- **Grades are binary.** `grade: 1` marks the paper the query was built from. There are
  no 0 rows and no graded 2 or 3 — LitSearch labels one paper out of 64,183.
- **Unlabelled is unjudged, not irrelevant.** Every metric result carries
  `judged_coverage` as a required field so a score cannot be reported without it.
- **A query is in-domain only when every gold is present.** 250 queries qualify; none are
  partial, because LitSearch's multi-gold queries either matched entirely or not at all.
- **`workflow` is derived from construction**, not judged per query: author-written sets
  are `known_item`, citation-derived sets are `related_work`.

## Query text is not in the repository

`data/fixtures/retrieval/queries.jsonl` carries ids, annotations, splits and provenance —
**not the query strings**. LitSearch declares no license (E1), so its text stays under
`${DATA_DIR}/benchmarks/litsearch-dataset/` and `datasets.hydrate` re-attaches it at run
time from the pinned parquet, raising rather than leaving a query empty.

This keeps the thing that must be frozen and reviewable — the split assignment — in git,
without redistributing content whose terms are unknown. The full copy with text exists on
this machine and is reproducible from the pinned revision by anyone who fetches it
themselves.

## Known limitations

- **48 in-domain test queries** is a small instrument. Report intervals, not point
  estimates alone.
- **`inline_acl` has zero corpus coverage** and `inline_nonacl` has 2.8%. Those 351
  queries are only usable against LitSearch's own corpus.
- **No "keeping up" workflow.** LitSearch exercises known-item and related-work search.
  The workflow this project exists for — what appeared recently that I should read — is
  not represented, and the 30 hand-authored queries in E4 are the only planned cover.
- **`specificity` and `quality` have no documented scale** in the published artifacts.
  They travel as metadata for slicing; splits are not stratified on them.
- **Binary nDCG** carries less information than the graded 0–3 scale spec §11 describes.
  It is reported for comparability with published LitSearch results.

## Reproducing

```bash
# 1. fetch the benchmark (refuses any revision that is not a 40-char commit)
uv run --env-file .env --project backend python -c "
from pathlib import Path; import os
from copilot.evaluation.litsearch import fetch_litsearch
fetch_litsearch(Path(os.environ['DATA_DIR']) / 'benchmarks' / 'litsearch')"

# 2. validate what is in the repository
uv run --project backend python -m copilot.cli eval validate-dataset --path data/fixtures/retrieval
```

The validator refuses a family spanning two splits, a qrel for an unknown query, a query
with no qrel, a query with no split, a duplicate id and a negative grade. On success it
prints the shape it accepted rather than only that it accepted something.
