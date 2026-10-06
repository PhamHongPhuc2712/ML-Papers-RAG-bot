# Retrieval gaps — m2-first-stage

Rendered by `eval gaps` from `reports/retrieval/m2-first-stage/<split>/per_query.parquet`; nothing here was rerun. Corpus release `m2-20260924T095724Z`, labels `paper_id`.

LitSearch's cutoffs follow its paper (arXiv 2407.18940v2, Table 3): R@20 for broad questions, R@5 and R@20 for specific ones, inline-citation and author-written sets apart. Recall is the share of a query's gold papers in the top k, as in the benchmark's `calculate_recall` and in ours. Specificity 0 is broad, 1 specific. Cells are percentages with 95% bootstrap intervals over queries.

This run is **not** over LitSearch's corpus, so the published rows are not shown: the same cutoffs over another corpus measure something else.

## Development — 150 queries

### Recall at LitSearch's published cutoffs

| Variant | inline broad R@20 (n=1) | inline specific R@5 (n=2) | inline specific R@20 (n=2) | author broad R@20 (n=24) | author specific R@5 (n=123) | author specific R@20 (n=123) |
|---|---|---|---|---|---|---|
| `hybrid_rerank` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 66.7 [45.8, 83.3] | 68.3 [60.2, 75.6] | 81.3 [74.0, 87.0] |
| `bm25_chunks` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 54.2 [33.3, 75.0] | 61.8 [53.7, 70.7] | 76.4 [69.1, 84.6] |
| `dense_chunks` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 50.0 [29.2, 70.8] | 51.2 [41.5, 59.3] | 66.7 [57.7, 74.0] |
| `hybrid_rerank_chunks` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 66.7 [45.8, 83.3] | 72.4 [63.4, 79.7] | 82.9 [74.8, 88.6] |
| `hybrid_rerank_both` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 79.2 [62.5, 91.8] | 71.5 [62.6, 78.9] | 84.6 [77.2, 90.2] |

### Where the gold paper is lost (depth 50)

Share of gold papers, averaged per query. The pool is the union of each branch's top 100; for a single-branch mode it is that branch's list.

| Variant | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| `hybrid_rerank` | 150 | 80.7 | 9.3 | 10.0 |
| `bm25_chunks` | 150 | 80.0 | 5.3 | 14.7 |
| `dense_chunks` | 150 | 72.0 | 7.3 | 20.7 |
| `hybrid_rerank_chunks` | 150 | 84.0 | 7.3 | 8.7 |
| `hybrid_rerank_both` | 150 | 88.0 | 4.7 | 7.3 |

`hybrid_rerank_both` by slice:

| Slice | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| query_set: `inline_nonacl` | 3 | 100.0 | 0.0 | 0.0 |
| query_set: `manual_acl` | 93 | 91.4 | 2.2 | 6.5 |
| query_set: `manual_iclr` | 54 | 81.5 | 9.3 | 9.3 |
| specificity: `broad` | 25 | 84.0 | 0.0 | 16.0 |
| specificity: `specific` | 125 | 88.8 | 5.6 | 5.6 |

### Gold rank distribution

Counts of gold papers by one-based rank; `absent` means not in the stored ranking.

| Variant | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 150 | 66 | 34 | 11 | 8 | 2 | 29 |
| `bm25_chunks` | 150 | 58 | 31 | 11 | 10 | 10 | 30 |
| `dense_chunks` | 150 | 46 | 29 | 11 | 11 | 11 | 42 |
| `hybrid_rerank_chunks` | 150 | 67 | 39 | 10 | 5 | 5 | 24 |
| `hybrid_rerank_both` | 150 | 65 | 40 | 10 | 11 | 6 | 18 |

`hybrid_rerank_both` by slice:

| Slice | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| query_set: `inline_nonacl` | 3 | 1 | 1 | 1 | 0 | 0 | 0 |
| query_set: `manual_acl` | 93 | 46 | 24 | 7 | 6 | 2 | 8 |
| query_set: `manual_iclr` | 54 | 18 | 15 | 2 | 5 | 4 | 10 |
| specificity: `broad` | 25 | 8 | 7 | 2 | 3 | 1 | 4 |
| specificity: `specific` | 125 | 57 | 33 | 8 | 8 | 5 | 14 |

### Queries whose gold paper `hybrid_rerank_both` lost

Query ids only; the text stays under `DATA_DIR`.

- **inline_nonacl / broad** (1 queries) — not in pool: none; in pool, beyond 50: none
- **inline_nonacl / specific** (2 queries) — not in pool: none; in pool, beyond 50: none
- **manual_acl / broad** (18 queries) — not in pool: litsearch-0467, litsearch-0475; in pool, beyond 50: none
- **manual_acl / specific** (75 queries) — not in pool: litsearch-0378, litsearch-0395, litsearch-0403, litsearch-0411; in pool, beyond 50: litsearch-0360, litsearch-0473
- **manual_iclr / broad** (6 queries) — not in pool: litsearch-0544, litsearch-0557; in pool, beyond 50: none
- **manual_iclr / specific** (48 queries) — not in pool: litsearch-0507, litsearch-0527, litsearch-0554; in pool, beyond 50: litsearch-0510, litsearch-0520, litsearch-0547, litsearch-0565, litsearch-0590
