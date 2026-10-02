# Retrieval gaps — e3-candidates

Rendered by `eval gaps` from `reports/retrieval/e3-candidates/<split>/per_query.parquet`; nothing here was rerun. Corpus release `litsearch-v1`, labels `corpusid`.

LitSearch's cutoffs follow its paper (arXiv 2407.18940v2, Table 3): R@20 for broad questions, R@5 and R@20 for specific ones, inline-citation and author-written sets apart. Recall is the share of a query's gold papers in the top k, as in the benchmark's `calculate_recall` and in ours. Specificity 0 is broad, 1 specific. Cells are percentages with 95% bootstrap intervals over queries.

This run is over LitSearch's own corpus, title and abstract, so it can sit beside the published rows. They cover all 597 queries; ours cover development and validation only — the locked test split is excluded — so the samples overlap rather than match.

## Development — 359 queries

### Recall at LitSearch's published cutoffs

| Variant | inline broad R@20 (n=79) | inline specific R@5 (n=132) | inline specific R@20 (n=132) | author broad R@20 (n=24) | author specific R@5 (n=124) | author specific R@20 (n=124) |
|---|---|---|---|---|---|---|
| `hybrid` | 45.5 [34.8, 55.7] | 50.8 [42.8, 58.7] | 70.8 [62.9, 78.0] | 66.7 [50.0, 87.5] | 66.9 [58.9, 75.0] | 79.8 [71.8, 87.1] |
| `hybrid_pool200` | 46.8 [36.0, 56.8] | 52.3 [43.9, 59.9] | 73.1 [65.5, 80.7] | 62.5 [41.7, 83.3] | 66.9 [58.9, 75.0] | 79.0 [71.8, 86.3] |
| `hybrid_pool300` | 51.2 [40.2, 61.6] | 52.3 [43.9, 59.9] | 73.1 [65.9, 80.3] | 66.7 [50.0, 87.5] | 66.9 [58.9, 75.0] | 79.8 [72.6, 87.1] |

### Where the gold paper is lost (depth 100)

Share of gold papers, averaged per query. The pool is the union of each branch's top 100; for a single-branch mode it is that branch's list.

| Variant | Queries | In top 100 | In pool, beyond 100 | Not in pool |
|---|---|---|---|---|
| `hybrid` | 359 | 82.8 | 3.2 | 14.0 |
| `hybrid_pool200` | 359 | 84.3 | 4.3 | 11.5 |
| `hybrid_pool300` | 359 | 84.0 | 7.1 | 9.0 |

`hybrid` by slice:

| Slice | Queries | In top 100 | In pool, beyond 100 | Not in pool |
|---|---|---|---|---|
| query_set: `inline_acl` | 59 | 74.0 | 5.1 | 21.0 |
| query_set: `inline_nonacl` | 152 | 80.6 | 3.0 | 16.4 |
| query_set: `manual_acl` | 93 | 88.2 | 3.2 | 8.6 |
| query_set: `manual_iclr` | 55 | 89.1 | 1.8 | 9.1 |
| specificity: `broad` | 103 | 72.9 | 2.4 | 24.6 |
| specificity: `specific` | 256 | 86.7 | 3.5 | 9.8 |

### Gold rank distribution

Counts of gold papers by one-based rank; `absent` means not in the stored ranking.

| Variant | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | 51-100 | absent |
|---|---|---|---|---|---|---|---|---|
| `hybrid` | 385 | 129 | 73 | 23 | 31 | 41 | 18 | 70 |
| `hybrid_pool200` | 385 | 129 | 74 | 25 | 31 | 38 | 24 | 64 |
| `hybrid_pool300` | 385 | 130 | 73 | 27 | 35 | 38 | 17 | 65 |

`hybrid` by slice:

| Slice | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | 51-100 | absent |
|---|---|---|---|---|---|---|---|---|
| query_set: `inline_acl` | 77 | 18 | 15 | 5 | 5 | 6 | 6 | 22 |
| query_set: `inline_nonacl` | 160 | 45 | 29 | 10 | 14 | 24 | 7 | 31 |
| query_set: `manual_acl` | 93 | 41 | 20 | 4 | 8 | 6 | 3 | 11 |
| query_set: `manual_iclr` | 55 | 25 | 9 | 4 | 4 | 5 | 2 | 6 |
| specificity: `broad` | 119 | 20 | 26 | 2 | 9 | 18 | 10 | 34 |
| specificity: `specific` | 266 | 109 | 47 | 21 | 22 | 23 | 8 | 36 |

### Queries whose gold paper `hybrid` lost

Query ids only; the text stays under `DATA_DIR`.

- **inline_acl / broad** (22 queries) — not in pool: litsearch-0002, litsearch-0009, litsearch-0023, litsearch-0059, litsearch-0070, litsearch-0075, litsearch-0092, litsearch-0095, litsearch-0096; in pool, beyond 100: litsearch-0000
- **inline_acl / specific** (37 queries) — not in pool: litsearch-0003, litsearch-0012, litsearch-0017, litsearch-0053, litsearch-0055, litsearch-0058, litsearch-0061; in pool, beyond 100: litsearch-0071, litsearch-0084
- **inline_nonacl / broad** (57 queries) — not in pool: litsearch-0123, litsearch-0137, litsearch-0142, litsearch-0149, litsearch-0177, litsearch-0179, litsearch-0194, litsearch-0205, litsearch-0213, litsearch-0249, litsearch-0257, litsearch-0260, litsearch-0269; in pool, beyond 100: litsearch-0226, litsearch-0257
- **inline_nonacl / specific** (95 queries) — not in pool: litsearch-0102, litsearch-0128, litsearch-0132, litsearch-0135, litsearch-0163, litsearch-0166, litsearch-0199, litsearch-0238, litsearch-0244, litsearch-0261, litsearch-0278, litsearch-0297, litsearch-0340; in pool, beyond 100: litsearch-0265, litsearch-0310, litsearch-0312
- **manual_acl / broad** (18 queries) — not in pool: litsearch-0404, litsearch-0426, litsearch-0467, litsearch-0475; in pool, beyond 100: none
- **manual_acl / specific** (75 queries) — not in pool: litsearch-0378, litsearch-0409, litsearch-0411, litsearch-0473; in pool, beyond 100: litsearch-0395, litsearch-0421, litsearch-0502
- **manual_iclr / broad** (6 queries) — not in pool: litsearch-0544, litsearch-0557; in pool, beyond 100: none
- **manual_iclr / specific** (49 queries) — not in pool: litsearch-0510, litsearch-0527, litsearch-0566; in pool, beyond 100: litsearch-0519

## development — 359 queries, beside the paper

| Variant | inline broad R@20 (n=79) | inline specific R@5 (n=132) | inline specific R@20 (n=132) | author broad R@20 (n=24) | author specific R@5 (n=124) | author specific R@20 (n=124) |
|---|---|---|---|---|---|---|
| `hybrid` | 45.5 [34.8, 55.7] | 50.8 [42.8, 58.7] | 70.8 [62.9, 78.0] | 66.7 [50.0, 87.5] | 66.9 [58.9, 75.0] | 79.8 [71.8, 87.1] |
| `hybrid_pool200` | 46.8 [36.0, 56.8] | 52.3 [43.9, 59.9] | 73.1 [65.5, 80.7] | 62.5 [41.7, 83.3] | 66.9 [58.9, 75.0] | 79.0 [71.8, 86.3] |
| `hybrid_pool300` | 51.2 [40.2, 61.6] | 52.3 [43.9, 59.9] | 73.1 [65.9, 80.3] | 66.7 [50.0, 87.5] | 66.9 [58.9, 75.0] | 79.8 [72.6, 87.1] |
| *published:* BM25 | 37.4 | 38.5 | 55.8 | 48.6 | 62.6 | 73.5 |
| *published:* GTR-T5-large | 45.7 | 38.5 | 51.5 | 37.1 | 40.8 | 55.9 |
| *published:* Instructor-XL | 56.3 | 48.9 | 60.0 | 57.1 | 55.9 | 70.1 |
| *published:* E5-large-v2 | 55.8 | 50.4 | 63.9 | 54.3 | 62.6 | 75.8 |
| *published:* GritLM-7B | 69.7 | 67.7 | 77.9 | 74.3 | 82.5 | 89.1 |
| *published:* GPT-4o reranking (w/ BM25) | 54.9 | 60.0 | 67.5 | 77.1 | 76.8 | 82.9 |
| *published:* GPT-4o one-hop (w/ BM25) | 62.0 | 64.1 | 71.6 | 74.3 | 73.5 | 77.7 |
| *published:* GPT-4o reranking (w/ GritLM) | 74.7 | 73.2 | 79.9 | 77.1 | 85.8 | 92.4 |
| *published:* GPT-4o one-hop (w/ GritLM) | 72.9 | 70.3 | 78.4 | 74.3 | 84.4 | 87.2 |
