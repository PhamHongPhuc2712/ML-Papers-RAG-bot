# Retrieval gaps — e3-litsearch

Rendered by `eval gaps` from `reports/retrieval/e3-litsearch/<split>/per_query.parquet`; nothing here was rerun. Corpus release `litsearch-v1`, labels `corpusid`.

LitSearch's cutoffs follow its paper (arXiv 2407.18940v2, Table 3): R@20 for broad questions, R@5 and R@20 for specific ones, inline-citation and author-written sets apart. Recall is the share of a query's gold papers in the top k, as in the benchmark's `calculate_recall` and in ours. Specificity 0 is broad, 1 specific. Cells are percentages with 95% bootstrap intervals over queries.

This run is over LitSearch's own corpus, title and abstract, so it can sit beside the published rows. They cover all 597 queries; ours cover development and validation only — the locked test split is excluded — so the samples overlap rather than match.

## Development — 359 queries

### Recall at LitSearch's published cutoffs

| Variant | inline broad R@20 (n=79) | inline specific R@5 (n=132) | inline specific R@20 (n=132) | author broad R@20 (n=24) | author specific R@5 (n=124) | author specific R@20 (n=124) |
|---|---|---|---|---|---|---|
| `bm25` | 39.8 [29.2, 50.2] | 37.9 [29.5, 45.5] | 58.7 [50.7, 66.3] | 45.8 [25.0, 66.7] | 62.1 [54.0, 70.2] | 75.8 [67.7, 83.1] |
| `dense` | 55.1 [44.8, 65.3] | 52.7 [43.9, 60.6] | 65.9 [57.6, 73.5] | 54.2 [33.3, 75.0] | 53.2 [44.4, 62.1] | 70.2 [62.1, 78.2] |
| `hybrid` | 45.5 [34.8, 55.7] | 50.8 [42.8, 58.7] | 70.8 [62.9, 78.0] | 66.7 [50.0, 87.5] | 66.9 [58.9, 75.0] | 79.8 [71.8, 87.1] |
| `hybrid_rerank` | 57.3 [46.1, 67.0] | 61.7 [53.0, 70.1] | 76.9 [69.3, 83.7] | 66.7 [50.0, 87.5] | 75.0 [66.9, 82.3] | 84.7 [77.4, 91.1] |

### Where the gold paper is lost (depth 50)

Share of gold papers, averaged per query. The pool is the union of each branch's top 100; for a single-branch mode it is that branch's list.

| Variant | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| `bm25` | 359 | 67.8 | 3.6 | 28.6 |
| `dense` | 359 | 71.9 | 5.4 | 22.7 |
| `hybrid` | 359 | 78.4 | 7.6 | 14.0 |
| `hybrid_rerank` | 359 | 78.4 | 7.6 | 14.0 |

`hybrid_rerank` by slice:

| Slice | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| query_set: `inline_acl` | 59 | 66.0 | 13.1 | 21.0 |
| query_set: `inline_nonacl` | 152 | 76.6 | 6.9 | 16.4 |
| query_set: `manual_acl` | 93 | 84.9 | 6.5 | 8.6 |
| query_set: `manual_iclr` | 55 | 85.5 | 5.5 | 9.1 |
| specificity: `broad` | 103 | 65.5 | 9.9 | 24.6 |
| specificity: `specific` | 256 | 83.6 | 6.6 | 9.8 |

### Gold rank distribution

Counts of gold papers by one-based rank; `absent` means not in the stored ranking.

| Variant | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| `bm25` | 385 | 105 | 63 | 39 | 19 | 31 | 128 |
| `dense` | 385 | 112 | 74 | 29 | 26 | 32 | 112 |
| `hybrid` | 385 | 129 | 73 | 23 | 31 | 41 | 88 |
| `hybrid_rerank` | 385 | 154 | 79 | 27 | 20 | 17 | 88 |

`hybrid_rerank` by slice:

| Slice | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| query_set: `inline_acl` | 77 | 23 | 13 | 3 | 5 | 5 | 28 |
| query_set: `inline_nonacl` | 160 | 56 | 34 | 12 | 13 | 7 | 38 |
| query_set: `manual_acl` | 93 | 45 | 23 | 7 | 1 | 3 | 14 |
| query_set: `manual_iclr` | 55 | 30 | 9 | 5 | 1 | 2 | 8 |
| specificity: `broad` | 119 | 32 | 22 | 6 | 7 | 8 | 44 |
| specificity: `specific` | 266 | 122 | 57 | 21 | 13 | 9 | 44 |

### Queries whose gold paper `hybrid_rerank` lost

Query ids only; the text stays under `DATA_DIR`.

- **inline_acl / broad** (22 queries) — not in pool: litsearch-0002, litsearch-0009, litsearch-0023, litsearch-0059, litsearch-0070, litsearch-0075, litsearch-0092, litsearch-0095, litsearch-0096; in pool, beyond 50: litsearch-0000, litsearch-0011, litsearch-0067, litsearch-0086, litsearch-0087, litsearch-0092, litsearch-0096
- **inline_acl / specific** (37 queries) — not in pool: litsearch-0003, litsearch-0012, litsearch-0017, litsearch-0053, litsearch-0055, litsearch-0058, litsearch-0061; in pool, beyond 50: litsearch-0071, litsearch-0084
- **inline_nonacl / broad** (57 queries) — not in pool: litsearch-0123, litsearch-0137, litsearch-0142, litsearch-0149, litsearch-0177, litsearch-0179, litsearch-0194, litsearch-0205, litsearch-0213, litsearch-0249, litsearch-0257, litsearch-0260, litsearch-0269; in pool, beyond 50: litsearch-0150, litsearch-0226, litsearch-0257, litsearch-0303
- **inline_nonacl / specific** (95 queries) — not in pool: litsearch-0102, litsearch-0128, litsearch-0132, litsearch-0135, litsearch-0163, litsearch-0166, litsearch-0199, litsearch-0238, litsearch-0244, litsearch-0261, litsearch-0278, litsearch-0297, litsearch-0340; in pool, beyond 50: litsearch-0158, litsearch-0188, litsearch-0206, litsearch-0230, litsearch-0265, litsearch-0310, litsearch-0312
- **manual_acl / broad** (18 queries) — not in pool: litsearch-0404, litsearch-0426, litsearch-0467, litsearch-0475; in pool, beyond 50: litsearch-0363
- **manual_acl / specific** (75 queries) — not in pool: litsearch-0378, litsearch-0409, litsearch-0411, litsearch-0473; in pool, beyond 50: litsearch-0371, litsearch-0395, litsearch-0421, litsearch-0457, litsearch-0502
- **manual_iclr / broad** (6 queries) — not in pool: litsearch-0544, litsearch-0557; in pool, beyond 50: none
- **manual_iclr / specific** (49 queries) — not in pool: litsearch-0510, litsearch-0527, litsearch-0566; in pool, beyond 50: litsearch-0519, litsearch-0575, litsearch-0590

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

## development + validation — 479 queries, beside the paper

| Variant | inline broad R@20 (n=98) | inline specific R@5 (n=184) | inline specific R@20 (n=184) | author broad R@20 (n=28) | author specific R@5 (n=169) | author specific R@20 (n=169) |
|---|---|---|---|---|---|---|
| `bm25` | 40.1 [30.9, 49.6] | 37.2 [30.4, 43.8] | 56.8 [49.2, 63.9] | 42.9 [25.0, 60.8] | 62.1 [54.4, 69.2] | 75.1 [68.6, 81.7] |
| `dense` | 53.1 [43.1, 62.8] | 49.7 [42.7, 56.5] | 62.0 [55.4, 68.5] | 50.0 [32.1, 67.9] | 55.0 [47.9, 62.1] | 69.8 [62.7, 76.9] |
| `hybrid` | 43.6 [33.3, 53.7] | 48.9 [42.1, 55.7] | 67.1 [60.1, 73.9] | 60.7 [46.4, 82.1] | 66.3 [59.2, 73.4] | 77.5 [71.0, 83.4] |
| `hybrid_rerank` | 54.1 [44.2, 63.5] | 57.6 [50.3, 64.4] | 72.8 [66.3, 79.3] | 60.7 [46.4, 82.1] | 72.2 [65.1, 78.7] | 83.4 [78.1, 88.8] |
| *published:* BM25 | 37.4 | 38.5 | 55.8 | 48.6 | 62.6 | 73.5 |
| *published:* GTR-T5-large | 45.7 | 38.5 | 51.5 | 37.1 | 40.8 | 55.9 |
| *published:* Instructor-XL | 56.3 | 48.9 | 60.0 | 57.1 | 55.9 | 70.1 |
| *published:* E5-large-v2 | 55.8 | 50.4 | 63.9 | 54.3 | 62.6 | 75.8 |
| *published:* GritLM-7B | 69.7 | 67.7 | 77.9 | 74.3 | 82.5 | 89.1 |
| *published:* GPT-4o reranking (w/ BM25) | 54.9 | 60.0 | 67.5 | 77.1 | 76.8 | 82.9 |
| *published:* GPT-4o one-hop (w/ BM25) | 62.0 | 64.1 | 71.6 | 74.3 | 73.5 | 77.7 |
| *published:* GPT-4o reranking (w/ GritLM) | 74.7 | 73.2 | 79.9 | 77.1 | 85.8 | 92.4 |
| *published:* GPT-4o one-hop (w/ GritLM) | 72.9 | 70.3 | 78.4 | 74.3 | 84.4 | 87.2 |
