# Retrieval gaps — e3-litsearch

Rendered by `eval gaps` from `reports/retrieval/e3-final/<split>/per_query.parquet`; nothing here was rerun. Corpus release `litsearch-v1`, labels `corpusid`.

LitSearch's cutoffs follow its paper (arXiv 2407.18940v2, Table 3): R@20 for broad questions, R@5 and R@20 for specific ones, inline-citation and author-written sets apart. Recall is the share of a query's gold papers in the top k, as in the benchmark's `calculate_recall` and in ours. Specificity 0 is broad, 1 specific. Cells are percentages with 95% bootstrap intervals over queries.

This run is over LitSearch's own corpus, title and abstract, so it can sit beside the published rows. They cover all 597 queries; ours cover development and validation only — the locked test split is excluded — so the samples overlap rather than match.

## Validation — 120 queries

### Recall at LitSearch's published cutoffs

| Variant | inline broad R@20 (n=19) | inline specific R@5 (n=52) | inline specific R@20 (n=52) | author broad R@20 (n=4) | author specific R@5 (n=45) | author specific R@20 (n=45) |
|---|---|---|---|---|---|---|
| `bm25` | 41.2 [21.1, 63.2] | 35.6 [22.1, 49.0] | 51.9 [38.5, 64.4] | 25.0 [0.0, 75.0] | 62.2 [48.9, 75.6] | 73.3 [60.0, 84.4] |
| `dense` | 44.7 [23.7, 68.4] | 42.3 [28.8, 55.8] | 51.9 [38.5, 65.4] | 25.0 [0.0, 75.0] | 60.0 [46.7, 73.3] | 68.9 [53.3, 82.2] |
| `hybrid` | 36.0 [15.8, 57.0] | 44.2 [30.8, 57.7] | 57.7 [44.2, 71.2] | 25.0 [0.0, 75.0] | 64.4 [51.1, 77.8] | 71.1 [57.8, 82.2] |
| `hybrid_rerank` | 41.2 [20.2, 63.2] | 47.1 [33.6, 61.5] | 62.5 [49.0, 76.0] | 25.0 [0.0, 75.0] | 64.4 [51.1, 77.8] | 80.0 [68.9, 91.1] |

### Where the gold paper is lost (depth 50)

Share of gold papers, averaged per query. The pool is the union of each branch's top 100; for a single-branch mode it is that branch's list.

| Variant | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| `bm25` | 120 | 64.9 | 4.2 | 31.0 |
| `dense` | 120 | 62.5 | 6.7 | 30.8 |
| `hybrid` | 120 | 69.6 | 9.2 | 21.2 |
| `hybrid_rerank` | 120 | 69.6 | 9.2 | 21.2 |

`hybrid_rerank` by slice:

| Slice | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| query_set: `inline_acl` | 20 | 57.5 | 15.0 | 27.5 |
| query_set: `inline_nonacl` | 51 | 64.7 | 7.8 | 27.5 |
| query_set: `manual_acl` | 31 | 71.0 | 9.7 | 19.4 |
| query_set: `manual_iclr` | 18 | 94.4 | 5.6 | 0.0 |
| specificity: `broad` | 23 | 54.3 | 13.0 | 32.6 |
| specificity: `specific` | 97 | 73.2 | 8.2 | 18.6 |

### Gold rank distribution

Counts of gold papers by one-based rank; `absent` means not in the stored ranking.

| Variant | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| `bm25` | 132 | 27 | 26 | 8 | 10 | 10 | 51 |
| `dense` | 132 | 29 | 29 | 7 | 5 | 8 | 54 |
| `hybrid` | 132 | 35 | 26 | 5 | 6 | 16 | 44 |
| `hybrid_rerank` | 132 | 43 | 20 | 9 | 8 | 8 | 44 |

`hybrid_rerank` by slice:

| Slice | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| query_set: `inline_acl` | 30 | 5 | 3 | 2 | 3 | 2 | 15 |
| query_set: `inline_nonacl` | 53 | 18 | 7 | 2 | 3 | 4 | 19 |
| query_set: `manual_acl` | 31 | 12 | 4 | 4 | 1 | 1 | 9 |
| query_set: `manual_iclr` | 18 | 8 | 6 | 1 | 1 | 1 | 1 |
| specificity: `broad` | 30 | 5 | 4 | 0 | 1 | 5 | 15 |
| specificity: `specific` | 102 | 38 | 16 | 9 | 7 | 3 | 29 |

### Queries whose gold paper `hybrid_rerank` lost

Query ids only; the text stays under `DATA_DIR`.

- **inline_acl / broad** (7 queries) — not in pool: litsearch-0018, litsearch-0030; in pool, beyond 50: none
- **inline_acl / specific** (13 queries) — not in pool: litsearch-0013, litsearch-0041, litsearch-0062, litsearch-0063, litsearch-0074; in pool, beyond 50: litsearch-0035, litsearch-0063, litsearch-0079, litsearch-0093
- **inline_nonacl / broad** (12 queries) — not in pool: litsearch-0136, litsearch-0167, litsearch-0185, litsearch-0228, litsearch-0328; in pool, beyond 50: litsearch-0145, litsearch-0224
- **inline_nonacl / specific** (39 queries) — not in pool: litsearch-0111, litsearch-0124, litsearch-0155, litsearch-0162, litsearch-0168, litsearch-0208, litsearch-0214, litsearch-0221, litsearch-0308, litsearch-0311; in pool, beyond 50: litsearch-0109, litsearch-0157
- **manual_acl / broad** (3 queries) — not in pool: litsearch-0369; in pool, beyond 50: none
- **manual_acl / specific** (28 queries) — not in pool: litsearch-0352, litsearch-0389, litsearch-0431, litsearch-0433, litsearch-0451; in pool, beyond 50: litsearch-0380, litsearch-0429, litsearch-0504
- **manual_iclr / broad** (1 queries) — not in pool: none; in pool, beyond 50: litsearch-0543
- **manual_iclr / specific** (17 queries) — not in pool: none; in pool, beyond 50: none

## validation — 120 queries, beside the paper

| Variant | inline broad R@20 (n=19) | inline specific R@5 (n=52) | inline specific R@20 (n=52) | author broad R@20 (n=4) | author specific R@5 (n=45) | author specific R@20 (n=45) |
|---|---|---|---|---|---|---|
| `bm25` | 41.2 [21.1, 63.2] | 35.6 [22.1, 49.0] | 51.9 [38.5, 64.4] | 25.0 [0.0, 75.0] | 62.2 [48.9, 75.6] | 73.3 [60.0, 84.4] |
| `dense` | 44.7 [23.7, 68.4] | 42.3 [28.8, 55.8] | 51.9 [38.5, 65.4] | 25.0 [0.0, 75.0] | 60.0 [46.7, 73.3] | 68.9 [53.3, 82.2] |
| `hybrid` | 36.0 [15.8, 57.0] | 44.2 [30.8, 57.7] | 57.7 [44.2, 71.2] | 25.0 [0.0, 75.0] | 64.4 [51.1, 77.8] | 71.1 [57.8, 82.2] |
| `hybrid_rerank` | 41.2 [20.2, 63.2] | 47.1 [33.6, 61.5] | 62.5 [49.0, 76.0] | 25.0 [0.0, 75.0] | 64.4 [51.1, 77.8] | 80.0 [68.9, 91.1] |
| *published:* BM25 | 37.4 | 38.5 | 55.8 | 48.6 | 62.6 | 73.5 |
| *published:* GTR-T5-large | 45.7 | 38.5 | 51.5 | 37.1 | 40.8 | 55.9 |
| *published:* Instructor-XL | 56.3 | 48.9 | 60.0 | 57.1 | 55.9 | 70.1 |
| *published:* E5-large-v2 | 55.8 | 50.4 | 63.9 | 54.3 | 62.6 | 75.8 |
| *published:* GritLM-7B | 69.7 | 67.7 | 77.9 | 74.3 | 82.5 | 89.1 |
| *published:* GPT-4o reranking (w/ BM25) | 54.9 | 60.0 | 67.5 | 77.1 | 76.8 | 82.9 |
| *published:* GPT-4o one-hop (w/ BM25) | 62.0 | 64.1 | 71.6 | 74.3 | 73.5 | 77.7 |
| *published:* GPT-4o reranking (w/ GritLM) | 74.7 | 73.2 | 79.9 | 77.1 | 85.8 | 92.4 |
| *published:* GPT-4o one-hop (w/ GritLM) | 72.9 | 70.3 | 78.4 | 74.3 | 84.4 | 87.2 |
