# Gold sections — retrieval

Rendered by `eval orb-sections` from the recorded `per_query.parquet`; nothing was rerun. A gold section maps to our chunks when a chunk holds a run of its sentences (at least 3, or 40% of the section when that is fewer); 1022 of 1527 distinct (document, section) pairs mapped. `section_hit@1` is over mapped queries only: the rank-1 paper is the gold paper and its best chunk in some branch lies in the gold section. `paper@1` is over every query.

| Variant | Facet | Queries | Mapped | Coverage | paper@1 | section_hit@1 | section hit given paper@1 |
|---|---|---|---|---|---|---|---|
| `bm25` | all | 1914 | 1277 | 66.7 | 90.4 | 59.7 | 65.0 |
| `bm25` | query_set:orb_abstractive | 893 | 655 | 73.3 | 91.6 | 57.3 | 62.4 |
| `bm25` | query_set:orb_extractive | 1021 | 622 | 60.9 | 89.3 | 62.4 | 67.8 |
| `bm25_papers` | all | 1914 | 1277 | 66.7 | 58.0 | 0.0 | 0.0 |
| `bm25_papers` | query_set:orb_abstractive | 893 | 655 | 73.3 | 72.9 | 0.0 | 0.0 |
| `bm25_papers` | query_set:orb_extractive | 1021 | 622 | 60.9 | 45.0 | 0.0 | 0.0 |
| `dense` | all | 1914 | 1277 | 66.7 | 85.7 | 50.4 | 57.8 |
| `dense` | query_set:orb_abstractive | 893 | 655 | 73.3 | 89.5 | 51.9 | 57.1 |
| `dense` | query_set:orb_extractive | 1021 | 622 | 60.9 | 82.5 | 48.9 | 58.6 |
| `dense_papers` | all | 1914 | 1277 | 66.7 | 55.4 | 0.0 | 0.0 |
| `dense_papers` | query_set:orb_abstractive | 893 | 655 | 73.3 | 71.3 | 0.0 | 0.0 |
| `dense_papers` | query_set:orb_extractive | 1021 | 622 | 60.9 | 41.5 | 0.0 | 0.0 |
| `hybrid` | all | 1914 | 1277 | 66.7 | 91.0 | 66.8 | 72.8 |
| `hybrid` | query_set:orb_abstractive | 893 | 655 | 73.3 | 93.4 | 67.3 | 72.1 |
| `hybrid` | query_set:orb_extractive | 1021 | 622 | 60.9 | 88.8 | 66.2 | 73.6 |
| `hybrid_papers` | all | 1914 | 1277 | 66.7 | 61.2 | 0.0 | 0.0 |
| `hybrid_papers` | query_set:orb_abstractive | 893 | 655 | 73.3 | 74.8 | 0.0 | 0.0 |
| `hybrid_papers` | query_set:orb_extractive | 1021 | 622 | 60.9 | 49.3 | 0.0 | 0.0 |
| `hybrid_rerank` | all | 1914 | 1277 | 66.7 | 66.2 | 48.2 | 69.4 |
| `hybrid_rerank` | query_set:orb_abstractive | 893 | 655 | 73.3 | 79.6 | 56.9 | 70.2 |
| `hybrid_rerank` | query_set:orb_extractive | 1021 | 622 | 60.9 | 54.6 | 38.9 | 68.2 |
| `hybrid_rerank_papers` | all | 1914 | 1277 | 66.7 | 65.0 | 0.0 | 0.0 |
| `hybrid_rerank_papers` | query_set:orb_abstractive | 893 | 655 | 73.3 | 79.3 | 0.0 | 0.0 |
| `hybrid_rerank_papers` | query_set:orb_extractive | 1021 | 622 | 60.9 | 52.6 | 0.0 | 0.0 |
