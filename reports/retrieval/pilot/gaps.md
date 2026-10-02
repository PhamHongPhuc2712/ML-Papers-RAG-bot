# Retrieval gaps — m2-retrieval-pilot

Rendered by `eval gaps` from `reports/retrieval/pilot/<split>/per_query.parquet`; nothing here was rerun. Corpus release `m2-20260924T095724Z`, labels `paper_id`.

LitSearch's cutoffs follow its paper (arXiv 2407.18940v2, Table 3): R@20 for broad questions, R@5 and R@20 for specific ones, inline-citation and author-written sets apart. Recall is the share of a query's gold papers in the top k, as in the benchmark's `calculate_recall` and in ours. Specificity 0 is broad, 1 specific. Cells are percentages with 95% bootstrap intervals over queries.

This run is **not** over LitSearch's corpus, so the published rows are not shown: the same cutoffs over another corpus measure something else.

## Development — 150 queries

### Recall at LitSearch's published cutoffs

| Variant | inline broad R@20 (n=1) | inline specific R@5 (n=2) | inline specific R@20 (n=2) | author broad R@20 (n=24) | author specific R@5 (n=123) | author specific R@20 (n=123) |
|---|---|---|---|---|---|---|
| `bm25` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 50.0 [29.2, 70.8] | 61.0 [51.2, 69.1] | 70.7 [61.8, 78.0] |
| `dense` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 50.0 [29.2, 70.8] | 50.4 [42.3, 58.6] | 68.3 [60.1, 76.4] |
| `hybrid` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 54.2 [33.3, 75.0] | 63.4 [54.5, 71.5] | 78.0 [69.9, 84.6] |
| `hybrid_rerank` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 66.7 [45.8, 83.3] | 68.3 [60.2, 75.6] | 81.3 [74.0, 87.0] |
| `hybrid_rerank_pair512` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 66.7 [45.8, 83.3] | 68.3 [60.2, 75.6] | 81.3 [74.0, 87.0] |
| `dense_bge_small` | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 54.2 [33.3, 75.0] | 52.8 [43.9, 61.8] | 66.7 [57.7, 74.0] |
| `hybrid_rerank_bge_small` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 62.5 [41.7, 83.3] | 71.5 [63.4, 78.9] | 83.7 [76.4, 89.4] |

### Where the gold paper is lost (depth 50)

Share of gold papers, averaged per query. The pool is the union of each branch's top 100; for a single-branch mode it is that branch's list.

| Variant | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| `bm25` | 150 | 74.0 | 4.0 | 22.0 |
| `dense` | 150 | 76.7 | 4.7 | 18.7 |
| `hybrid` | 150 | 80.7 | 9.3 | 10.0 |
| `hybrid_rerank` | 150 | 80.7 | 9.3 | 10.0 |
| `hybrid_rerank_pair512` | 150 | 80.7 | 9.3 | 10.0 |
| `dense_bge_small` | 150 | 75.3 | 3.3 | 21.3 |
| `hybrid_rerank_bge_small` | 150 | 83.3 | 4.7 | 12.0 |

`hybrid_rerank` by slice:

| Slice | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| query_set: `inline_nonacl` | 3 | 100.0 | 0.0 | 0.0 |
| query_set: `manual_acl` | 93 | 88.2 | 3.2 | 8.6 |
| query_set: `manual_iclr` | 54 | 66.7 | 20.4 | 13.0 |
| specificity: `broad` | 25 | 68.0 | 12.0 | 20.0 |
| specificity: `specific` | 125 | 83.2 | 8.8 | 8.0 |

### Gold rank distribution

Counts of gold papers by one-based rank; `absent` means not in the stored ranking.

| Variant | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| `bm25` | 150 | 51 | 37 | 7 | 7 | 9 | 39 |
| `dense` | 150 | 51 | 21 | 17 | 10 | 16 | 35 |
| `hybrid` | 150 | 64 | 27 | 13 | 8 | 9 | 29 |
| `hybrid_rerank` | 150 | 66 | 34 | 11 | 8 | 2 | 29 |
| `hybrid_rerank_pair512` | 150 | 66 | 34 | 11 | 8 | 2 | 29 |
| `dense_bge_small` | 150 | 46 | 31 | 13 | 7 | 16 | 37 |
| `hybrid_rerank_bge_small` | 150 | 68 | 35 | 10 | 8 | 4 | 25 |

`hybrid_rerank` by slice:

| Slice | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| query_set: `inline_nonacl` | 3 | 1 | 1 | 1 | 0 | 0 | 0 |
| query_set: `manual_acl` | 93 | 47 | 21 | 8 | 5 | 1 | 11 |
| query_set: `manual_iclr` | 54 | 18 | 12 | 2 | 3 | 1 | 18 |
| specificity: `broad` | 25 | 8 | 6 | 1 | 2 | 0 | 8 |
| specificity: `specific` | 125 | 58 | 28 | 10 | 6 | 2 | 21 |

### Queries whose gold paper `hybrid_rerank` lost

Query ids only; the text stays under `DATA_DIR`.

- **inline_nonacl / broad** (1 queries) — not in pool: none; in pool, beyond 50: none
- **inline_nonacl / specific** (2 queries) — not in pool: none; in pool, beyond 50: none
- **manual_acl / broad** (18 queries) — not in pool: litsearch-0404, litsearch-0467; in pool, beyond 50: litsearch-0460, litsearch-0475
- **manual_acl / specific** (75 queries) — not in pool: litsearch-0378, litsearch-0395, litsearch-0403, litsearch-0409, litsearch-0411, litsearch-0473; in pool, beyond 50: litsearch-0421
- **manual_iclr / broad** (6 queries) — not in pool: litsearch-0544, litsearch-0557, litsearch-0595; in pool, beyond 50: litsearch-0521
- **manual_iclr / specific** (48 queries) — not in pool: litsearch-0527, litsearch-0554, litsearch-0560, litsearch-0590; in pool, beyond 50: litsearch-0506, litsearch-0507, litsearch-0509, litsearch-0510, litsearch-0519, litsearch-0532, litsearch-0540, litsearch-0547, litsearch-0565, litsearch-0575

## Validation — 52 queries

### Recall at LitSearch's published cutoffs

| Variant | inline broad R@20 (n=0) | inline specific R@5 (n=3) | inline specific R@20 (n=3) | author broad R@20 (n=4) | author specific R@5 (n=45) | author specific R@20 (n=45) |
|---|---|---|---|---|---|---|
| `bm25` | — | 33.3 [0.0, 100.0] | 33.3 [0.0, 100.0] | 25.0 [0.0, 75.0] | 53.3 [37.8, 66.7] | 66.7 [53.3, 80.0] |
| `dense` | — | 33.3 [0.0, 100.0] | 33.3 [0.0, 100.0] | 25.0 [0.0, 75.0] | 55.6 [42.2, 68.9] | 73.3 [60.0, 84.4] |
| `hybrid` | — | 33.3 [0.0, 100.0] | 33.3 [0.0, 100.0] | 25.0 [0.0, 75.0] | 57.8 [42.2, 71.1] | 73.3 [60.0, 86.7] |
| `hybrid_rerank` | — | 33.3 [0.0, 100.0] | 33.3 [0.0, 100.0] | 25.0 [0.0, 75.0] | 62.2 [46.7, 75.6] | 77.8 [64.4, 88.9] |
| `hybrid_rerank_pair512` | — | 33.3 [0.0, 100.0] | 33.3 [0.0, 100.0] | 25.0 [0.0, 75.0] | 62.2 [46.7, 75.6] | 77.8 [64.4, 88.9] |
| `dense_bge_small` | — | 0.0 [0.0, 0.0] | 33.3 [0.0, 100.0] | 25.0 [0.0, 75.0] | 51.1 [37.8, 64.4] | 64.4 [51.1, 77.8] |
| `hybrid_rerank_bge_small` | — | 33.3 [0.0, 100.0] | 33.3 [0.0, 100.0] | 25.0 [0.0, 75.0] | 62.2 [46.7, 75.6] | 77.8 [64.4, 88.9] |

### Where the gold paper is lost (depth 50)

Share of gold papers, averaged per query. The pool is the union of each branch's top 100; for a single-branch mode it is that branch's list.

| Variant | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| `bm25` | 52 | 69.2 | 5.8 | 25.0 |
| `dense` | 52 | 71.2 | 5.8 | 23.1 |
| `hybrid` | 52 | 76.9 | 7.7 | 15.4 |
| `hybrid_rerank` | 52 | 76.9 | 7.7 | 15.4 |
| `hybrid_rerank_pair512` | 52 | 76.9 | 7.7 | 15.4 |
| `dense_bge_small` | 52 | 71.2 | 1.9 | 26.9 |
| `hybrid_rerank_bge_small` | 52 | 75.0 | 9.6 | 15.4 |

`hybrid_rerank` by slice:

| Slice | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| query_set: `inline_nonacl` | 3 | 33.3 | 33.3 | 33.3 |
| query_set: `manual_acl` | 31 | 74.2 | 6.5 | 19.4 |
| query_set: `manual_iclr` | 18 | 88.9 | 5.6 | 5.6 |
| specificity: `broad` | 4 | 25.0 | 0.0 | 75.0 |
| specificity: `specific` | 48 | 81.2 | 8.3 | 10.4 |

### Gold rank distribution

Counts of gold papers by one-based rank; `absent` means not in the stored ranking.

| Variant | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| `bm25` | 52 | 18 | 8 | 3 | 3 | 4 | 16 |
| `dense` | 52 | 14 | 13 | 4 | 4 | 2 | 15 |
| `hybrid` | 52 | 20 | 8 | 5 | 2 | 5 | 12 |
| `hybrid_rerank` | 52 | 20 | 10 | 3 | 4 | 3 | 12 |
| `hybrid_rerank_pair512` | 52 | 20 | 10 | 3 | 4 | 3 | 12 |
| `dense_bge_small` | 52 | 13 | 11 | 3 | 4 | 6 | 15 |
| `hybrid_rerank_bge_small` | 52 | 22 | 8 | 3 | 4 | 2 | 13 |

`hybrid_rerank` by slice:

| Slice | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| query_set: `inline_nonacl` | 3 | 1 | 0 | 0 | 0 | 0 | 2 |
| query_set: `manual_acl` | 31 | 12 | 7 | 1 | 2 | 1 | 8 |
| query_set: `manual_iclr` | 18 | 7 | 3 | 2 | 2 | 2 | 2 |
| specificity: `broad` | 4 | 0 | 1 | 0 | 0 | 0 | 3 |
| specificity: `specific` | 48 | 20 | 9 | 3 | 4 | 3 | 9 |

### Queries whose gold paper `hybrid_rerank` lost

Query ids only; the text stays under `DATA_DIR`.

- **inline_nonacl / specific** (3 queries) — not in pool: litsearch-0221; in pool, beyond 50: litsearch-0349
- **manual_acl / broad** (3 queries) — not in pool: litsearch-0369, litsearch-0423; in pool, beyond 50: none
- **manual_acl / specific** (28 queries) — not in pool: litsearch-0389, litsearch-0417, litsearch-0433, litsearch-0504; in pool, beyond 50: litsearch-0352, litsearch-0380
- **manual_iclr / broad** (1 queries) — not in pool: litsearch-0543; in pool, beyond 50: none
- **manual_iclr / specific** (17 queries) — not in pool: none; in pool, beyond 50: litsearch-0583
