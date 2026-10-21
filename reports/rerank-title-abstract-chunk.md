# The reranker reads title + abstract + passage (2026-10-20/21)

## Results in plain words

- **Shipped since 2026-10-21, the developer's decision**: `configs/search.yaml` `rerank.text:
  title_abstract_chunk`. The cross-encoder now reads each candidate's title and abstract, then the
  passage (chunk) that found the paper. Before it read title + passage (`title_chunk`, U1).
- Against title + passage (nDCG@10, paired difference [95% interval over query families]):
  - **S1**, passage questions, our corpus: +0.024 [−0.001, +0.051] on development, +0.023
    [+0.000, +0.047] on validation. Not worse, perhaps slightly better.
  - **S3**, "find papers about X", our corpus: +0.056 [+0.031, +0.082] on development (0.583 against
    0.527). The old title + abstract reranker still leads there (0.616).
  - **Open RAG Bench**, 1,914 passage questions, another corpus: −0.002 [−0.006, +0.003].
    No difference.
- Cost: about 0.15 s more per search (ORB p95 1.46 s against 1.33 s), inside the 2.5 s rerank
  deadline, which is unchanged.
- Measurement, as U4 was: S1 and S3 are model-written with one gold paper per query, so S3
  undercounts every other relevant paper. Not a judged paper-finding measure.

## What the reranker reads

`search/service.py` `_rerank_texts`: `f"{title}\n{abstract}\n{chunk}"` (the abstract is omitted when
the paper has none). The pair budget (1,024 tokens, `configs/models.yaml`) truncates the end of the
text, so the passage's tail is cut, never the abstract. A 1,536-token variant, which cuts less of the
passage, was no better on any set and slower (p95 about 1.9 s), so the budget stays.
Test: `test_title_abstract_chunk_reranking_reads_the_abstract_before_the_chunk`
(`backend/tests/integration/test_search_service.py`), which failed before the mode existed.

## Runs

Release `m2-20260924T095724Z` (after the bibliography relabel, `reports/references-relabel.md`) and
ORB's `orb-v1`; code `cb7bbbf` plus this change (uncommitted, recorded as dirty in each manifest).
Configs `configs/experiments/rerank-title-abstract-chunk-{sections,topics}.yaml` and
`orb-rerank-title-abstract-chunk.yaml`. Test splits unread. Each run warmed first with a 30-query
partial run outside `reports/`.

### nDCG@10

| Set, split | queries | title + abstract (old) | title + passage (was shipped) | **title + abstract + passage** | same, 1,536-token pairs | no reranker |
|---|---|---|---|---|---|---|
| S1 development | 190 | 0.374 | 0.682 | **0.705** | 0.694 | — |
| S1 validation | 190 | 0.350 | 0.698 | **0.721** | 0.719 | — |
| S3 development | 349 | 0.616 | 0.527 | **0.583** | 0.583 | — |
| ORB retrieval | 1,914 | (0.785, O3) | 0.968 | **0.966** | — | 0.948 |

Folders: `reports/retrieval/rerank-title-abstract-chunk-{sections,topics}/`,
`reports/retrieval/orb-rerank-title-abstract-chunk/`. The harness's `decision` field is not read.
S1's title + passage and title + abstract rows reproduce U4's exactly, so the relabel did not move them.
ORB's title + passage and no-reranker rows reproduce `orb-rerank-text` (0.968, 0.947).

Degraded queries: none in S1; one per new variant in S3 development (a `dense_timeout`, a
`rerank_timeout`); one of 1,914 in ORB's new variant. Every recorded run had the GPU at 0% before it.

### S3 validation: not recorded

Two attempts were set aside, both under the GPU policy:
1. 2026-10-20 ~15:16 local: the laptop was unplugged during the run; the GPU was power-capped at about
   40 W and 930–1,140 MHz. The title + passage variant ran first and degraded 348 of 349 queries. The
   other three ran at normal speed with no degraded query: title + abstract 0.663, title + abstract +
   passage 0.582, 1,536-token pairs 0.581.
2. Rerun on mains power with the battery at 16% and charging: every variant except the 1,536-token one
   ran slow (title + abstract median 1.93 s against 0.54 s); 259 and 349 queries degraded.

Both are kept outside the repository. A clean S3 validation run of all four variants is still owed for
the record; U4 measured title + passage there at 0.563 before the relabel.

### Latency of `rank()` on ORB (ms)

| Variant | p50 | p95 | max |
|---|---|---|---|
| no reranker | 27.7 | 40.1 | 785.3 |
| title + passage | 1,218.3 | 1,325.5 | 2,049.5 |
| title + abstract + passage | 1,403.7 | 1,459.8 | 2,384.2 |

## What changed with the switch

- `configs/search.yaml` `rerank.text: title_abstract_chunk`; the 2.5 s rerank deadline stays.
- `configs/experiments/orb-text-qa-u2-overview.yaml` pins `title_chunk` at 2.5 s, the stage U2 ran
  under, so the recorded experiment keeps its meaning. Other experiment configs that read
  `configs/search.yaml` unpinned (U4's `shipped`) now mean the new stage if run again.
- ORB's answering numbers (frozen 400 `flat` 0.880) were measured with title + passage and are not
  re-run here.
