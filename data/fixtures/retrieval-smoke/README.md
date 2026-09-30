# Retrieval smoke fixture

Synthetic and written for this repository — no benchmark or corpus text — so it can
live in git and run anywhere. `eval smoke` ranks it with the four baselines using the
exact BM25 oracle, the two-dimensional fixture embedding and the term-overlap fixture
reranker: no services, no network, no model download.

- `papers.jsonl` — 24 invented papers in six topics.
- `queries.jsonl` — 12 queries in 10 families; paraphrases share a family. Splits are
  recorded but the smoke set scores all of them.
- `qrels.jsonl` — graded 0–3. A 0 is a judged irrelevant paper, which counts toward
  judged coverage. Query `smoke-12` has only a 0 label, so it has nothing to recall and
  is left out.
- `expected.json` — the frozen outputs. `eval smoke` fails when Recall@10 or nDCG@10
  falls more than 0.03 below them (spec §11) and reports any ranking that moved.
  Re-freeze with `eval smoke --update` only for an intended change, and say why in the
  commit.
