# Gold sections — retrieval

Rendered by `eval orb-sections` from the recorded `per_query.parquet`; nothing was rerun. A gold section maps to our chunks when a chunk holds a run of its sentences (at least 3, or 40% of the section when that is fewer); 987 of 1527 distinct (document, section) pairs mapped. `section_hit@1` is over mapped queries only: the rank-1 paper is the gold paper and its best chunk in some branch lies in the gold section. `paper@1` is over every query.

| Variant | Facet | Queries | Mapped | Coverage | paper@1 | section_hit@1 | section hit given paper@1 |
|---|---|---|---|---|---|---|---|
| `hybrid` | all | 1914 | 1229 | 64.2 | 92.6 | 57.1 | 61.2 |
| `hybrid` | query_set:orb_abstractive | 893 | 635 | 71.1 | 94.3 | 58.0 | 61.4 |
| `hybrid` | query_set:orb_extractive | 1021 | 594 | 58.2 | 91.2 | 56.2 | 60.9 |
| `hybrid_rerank` | all | 1914 | 1229 | 64.2 | 66.4 | 40.1 | 57.5 |
| `hybrid_rerank` | query_set:orb_abstractive | 893 | 635 | 71.1 | 79.6 | 47.7 | 58.5 |
| `hybrid_rerank` | query_set:orb_extractive | 1021 | 594 | 58.2 | 54.8 | 32.0 | 55.9 |
| `hybrid_rerank_chunk` | all | 1914 | 1229 | 64.2 | 92.2 | 57.1 | 61.4 |
| `hybrid_rerank_chunk` | query_set:orb_abstractive | 893 | 635 | 71.1 | 93.4 | 57.3 | 61.5 |
| `hybrid_rerank_chunk` | query_set:orb_extractive | 1021 | 594 | 58.2 | 91.2 | 56.9 | 61.2 |
| `hybrid_rerank_chunk_2500` | all | 1914 | 1229 | 64.2 | 92.2 | 57.1 | 61.4 |
| `hybrid_rerank_chunk_2500` | query_set:orb_abstractive | 893 | 635 | 71.1 | 93.4 | 57.3 | 61.5 |
| `hybrid_rerank_chunk_2500` | query_set:orb_extractive | 1021 | 594 | 58.2 | 91.2 | 56.9 | 61.2 |
| `hybrid_rerank_title_chunk_2500` | all | 1914 | 1229 | 64.2 | 93.2 | 57.4 | 61.1 |
| `hybrid_rerank_title_chunk_2500` | query_set:orb_abstractive | 893 | 635 | 71.1 | 94.7 | 57.8 | 61.1 |
| `hybrid_rerank_title_chunk_2500` | query_set:orb_extractive | 1021 | 594 | 58.2 | 91.8 | 56.9 | 61.1 |
| `hybrid_rerank_v1` | all | 1914 | 1229 | 64.2 | 66.2 | 0.0 | 0.0 |
| `hybrid_rerank_v1` | query_set:orb_abstractive | 893 | 635 | 71.1 | 79.6 | 0.0 | 0.0 |
| `hybrid_rerank_v1` | query_set:orb_extractive | 1021 | 594 | 58.2 | 54.6 | 0.0 | 0.0 |
| `hybrid_v1` | all | 1914 | 1229 | 64.2 | 91.0 | 0.0 | 0.0 |
| `hybrid_v1` | query_set:orb_abstractive | 893 | 635 | 71.1 | 93.4 | 0.0 | 0.0 |
| `hybrid_v1` | query_set:orb_extractive | 1021 | 594 | 58.2 | 88.8 | 0.0 | 0.0 |
