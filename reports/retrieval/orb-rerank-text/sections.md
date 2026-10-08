# Gold sections — retrieval

Rendered by `eval orb-sections` from the recorded `per_query.parquet`; nothing was rerun. A gold section maps to our chunks when a chunk holds a run of its sentences (at least 3, or 40% of the section when that is fewer); 1022 of 1527 distinct (document, section) pairs mapped. `section_hit@1` is over mapped queries only: the rank-1 paper is the gold paper and its best chunk in some branch lies in the gold section. `paper@1` is over every query.

| Variant | Facet | Queries | Mapped | Coverage | paper@1 | section_hit@1 | section hit given paper@1 |
|---|---|---|---|---|---|---|---|
| `hybrid` | all | 1914 | 1277 | 66.7 | 90.9 | 66.8 | 72.8 |
| `hybrid` | query_set:orb_abstractive | 893 | 655 | 73.3 | 93.3 | 67.3 | 72.2 |
| `hybrid` | query_set:orb_extractive | 1021 | 622 | 60.9 | 88.8 | 66.2 | 73.6 |
| `hybrid_rerank` | all | 1914 | 1277 | 66.7 | 66.2 | 48.2 | 69.4 |
| `hybrid_rerank` | query_set:orb_abstractive | 893 | 655 | 73.3 | 79.6 | 56.9 | 70.2 |
| `hybrid_rerank` | query_set:orb_extractive | 1021 | 622 | 60.9 | 54.6 | 38.9 | 68.2 |
| `hybrid_rerank_chunk` | all | 1914 | 1277 | 66.7 | 94.1 | 69.5 | 73.1 |
| `hybrid_rerank_chunk` | query_set:orb_abstractive | 893 | 655 | 73.3 | 95.5 | 69.5 | 72.5 |
| `hybrid_rerank_chunk` | query_set:orb_extractive | 1021 | 622 | 60.9 | 92.9 | 69.6 | 73.9 |
| `hybrid_rerank_chunk_2500` | all | 1914 | 1277 | 66.7 | 94.1 | 69.5 | 73.1 |
| `hybrid_rerank_chunk_2500` | query_set:orb_abstractive | 893 | 655 | 73.3 | 95.5 | 69.5 | 72.5 |
| `hybrid_rerank_chunk_2500` | query_set:orb_extractive | 1021 | 622 | 60.9 | 92.9 | 69.6 | 73.9 |
| `hybrid_rerank_title_chunk` | all | 1914 | 1277 | 66.7 | 94.1 | 69.4 | 73.2 |
| `hybrid_rerank_title_chunk` | query_set:orb_abstractive | 893 | 655 | 73.3 | 96.1 | 69.6 | 72.4 |
| `hybrid_rerank_title_chunk` | query_set:orb_extractive | 1021 | 622 | 60.9 | 92.5 | 69.1 | 74.0 |
| `hybrid_rerank_title_chunk_2500` | all | 1914 | 1277 | 66.7 | 94.4 | 69.7 | 73.1 |
| `hybrid_rerank_title_chunk_2500` | query_set:orb_abstractive | 893 | 655 | 73.3 | 96.0 | 69.8 | 72.4 |
| `hybrid_rerank_title_chunk_2500` | query_set:orb_extractive | 1021 | 622 | 60.9 | 93.0 | 69.6 | 73.8 |
