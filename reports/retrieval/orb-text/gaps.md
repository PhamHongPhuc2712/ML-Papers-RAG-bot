# Retrieval gaps — orb-text

Rendered by `eval gaps` from `reports/retrieval/orb-text/<split>/per_query.parquet`; nothing here was rerun. Corpus release `orb-v1`, labels `paper_id`.

LitSearch's cutoffs follow its paper (arXiv 2407.18940v2, Table 3): R@20 for broad questions, R@5 and R@20 for specific ones, inline-citation and author-written sets apart. Recall is the share of a query's gold papers in the top k, as in the benchmark's `calculate_recall` and in ours. Specificity 0 is broad, 1 specific. Cells are percentages with 95% bootstrap intervals over queries.

This run is **not** over LitSearch's corpus, so the published rows are not shown: the same cutoffs over another corpus measure something else.

## Retrieval — 1914 queries

### Recall at LitSearch's published cutoffs

| Variant | inline broad R@20 (n=0) | inline specific R@5 (n=0) | inline specific R@20 (n=0) | author broad R@20 (n=0) | author specific R@5 (n=0) | author specific R@20 (n=0) |
|---|---|---|---|---|---|---|
| `bm25_papers` | — | — | — | — | — | — |
| `dense_papers` | — | — | — | — | — | — |
| `hybrid_papers` | — | — | — | — | — | — |
| `hybrid_rerank_papers` | — | — | — | — | — | — |
| `bm25` | — | — | — | — | — | — |
| `dense` | — | — | — | — | — | — |
| `hybrid` | — | — | — | — | — | — |
| `hybrid_rerank` | — | — | — | — | — | — |

### Where the gold paper is lost (depth 50)

Share of gold papers, averaged per query. The pool is the union of each branch's top 100; for a single-branch mode it is that branch's list.

| Variant | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| `bm25_papers` | 1914 | 83.1 | 3.2 | 13.6 |
| `dense_papers` | 1914 | 83.3 | 3.9 | 12.8 |
| `hybrid_papers` | 1914 | 86.6 | 5.7 | 7.7 |
| `hybrid_rerank_papers` | 1914 | 86.6 | 5.7 | 7.7 |
| `bm25` | 1914 | 98.7 | 0.2 | 1.1 |
| `dense` | 1914 | 98.0 | 0.8 | 1.3 |
| `hybrid` | 1914 | 98.9 | 0.5 | 0.6 |
| `hybrid_rerank` | 1914 | 98.9 | 0.5 | 0.6 |

`hybrid_rerank` by slice:

| Slice | Queries | In top 50 | In pool, beyond 50 | Not in pool |
|---|---|---|---|---|
| query_set: `orb_abstractive` | 893 | 99.3 | 0.2 | 0.4 |
| query_set: `orb_extractive` | 1021 | 98.5 | 0.7 | 0.8 |
| specificity: `specific` | 1914 | 98.9 | 0.5 | 0.6 |

### Gold rank distribution

Counts of gold papers by one-based rank; `absent` means not in the stored ranking.

| Variant | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| `bm25_papers` | 1914 | 1110 | 226 | 98 | 63 | 94 | 323 |
| `dense_papers` | 1914 | 1061 | 248 | 89 | 90 | 106 | 320 |
| `hybrid_papers` | 1914 | 1171 | 250 | 80 | 57 | 100 | 256 |
| `hybrid_rerank_papers` | 1914 | 1245 | 237 | 76 | 53 | 47 | 256 |
| `bm25` | 1914 | 1730 | 113 | 20 | 13 | 13 | 25 |
| `dense` | 1914 | 1641 | 159 | 34 | 22 | 19 | 39 |
| `hybrid` | 1914 | 1741 | 109 | 17 | 15 | 11 | 21 |
| `hybrid_rerank` | 1914 | 1268 | 257 | 105 | 96 | 167 | 21 |

`hybrid_rerank` by slice:

| Slice | Golds | 1 | 2-5 | 6-10 | 11-20 | 21-50 | absent |
|---|---|---|---|---|---|---|---|
| query_set: `orb_abstractive` | 893 | 711 | 84 | 27 | 24 | 41 | 6 |
| query_set: `orb_extractive` | 1021 | 557 | 173 | 78 | 72 | 126 | 15 |
| specificity: `specific` | 1914 | 1268 | 257 | 105 | 96 | 167 | 21 |

### Queries whose gold paper `hybrid_rerank` lost

Query ids only; the text stays under `DATA_DIR`.

- **orb_abstractive / specific** (893 queries) — not in pool: 82aadc22-ff94-4eb7-b9c5-394dc2d18980, 8431252b-a908-42eb-ae19-75e1c1d68245, e4aa5613-8f28-44a6-ac2a-4e3186c624e0, fc4d4614-1926-4824-b9da-cfdcc2d0132e; in pool, beyond 50: d9653b7c-5ace-4aa4-ad81-39c3e739a67c, f6ed58d1-f346-4e32-b520-17dacd41dc42
- **orb_extractive / specific** (1021 queries) — not in pool: 0ee1e0b4-ba87-4b4b-9eb3-ccae367efaca, 2a20d5ac-d249-4bc8-8f83-4bdab8c800d1, 51ca04bc-2c6e-439d-a36d-ffbd993e684e, 579c73fa-0827-4d11-bca1-6b81e1da2c07, 781eab48-5ba2-40ba-99e6-b00a46817d77, ad569d84-fb7b-430f-94f7-21a7e923f5bd, cf5e78d1-9121-41ad-93be-30b28c2253ac, d6ad1c29-0565-474a-af9b-a17b9622b852; in pool, beyond 50: 01416eb6-2408-491e-a971-5647f51d7af8, 57974a20-83f8-4a30-b935-ca111cc76f08, 6e96f396-0181-4857-8121-7027e4d8baed, 7f25612d-9419-4e0a-a89a-9a5a6638e2a6, 8459f7cb-dc3a-4327-8195-98e269c487a6, 91ef1af9-518c-4588-bdc6-5b8ec42095c8, c83b2de2-c2b8-4d33-aaab-a843bd1aa46b
