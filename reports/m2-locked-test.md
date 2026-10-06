# M2 locked test — the release decision for G2

Date: 2026-10-06. Go-ahead given by the developer the same day, after P2.6 closed.
Scored **once**, with `--locked-test`, on an idle GPU (before-run p50 utilization 0%,
only the idle Windows-side process seen during both runs). No decision is computed from
the test split by the harness; this file records what the chosen configuration does on
queries nothing was ever tuned or chosen on. Host: WSL2 Linux, 12 cores / 23 GB RAM,
RTX 3080 Laptop 16 GB. Code `9124974`.

**The configuration under test** is the one validation chose: `hybrid_rerank` — exact
BM25 and BGE-M3 candidates, 100 per branch, fused by RRF (k = 60), the top 50 reranked by
`bge-reranker-v2-m3` at 1,024 pair tokens — over the **chunk collection** on our corpus
(P2.6, `candidates.source: chunks`; each paper scored by its best evidence chunk) and over
titles and abstracts on LitSearch's corpus, which has no chunks.

## LitSearch's corpus — the primary G2 evidence

Run `e3-litsearch-test-20261006T091138Z` (`reports/retrieval/e3-litsearch/test/`), the
118 test queries, 0 failed, 0 degraded. Means with 95% bootstrap intervals over query
families; judged coverage at 10 is 0.06–0.07 throughout (LitSearch labels only the paper a
query was written about, so every number is a lower bound).

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | p50 / p95 |
|---|---|---|---|---|---|
| `bm25` | 0.555 [0.470, 0.644] | 0.674 [0.593, 0.763] | 0.411 [0.339, 0.490] | 0.367 [0.294, 0.447] | 6 / 18 ms |
| `dense` | 0.551 [0.462, 0.644] | 0.746 [0.665, 0.822] | 0.406 [0.332, 0.484] | 0.367 [0.291, 0.448] | 28 / 61 ms |
| `hybrid` | 0.640 [0.559, 0.720] | 0.805 [0.737, 0.873] | 0.478 [0.405, 0.554] | 0.436 [0.362, 0.512] | 29 / 42 ms |
| **`hybrid_rerank`** | **0.682** [0.597, 0.763] | **0.805** [0.737, 0.873] | **0.550** [0.473, 0.621] | **0.516** [0.432, 0.589] | 521 / 810 ms |

Paired differences, candidate minus baseline:

| Comparison | nDCG@10 | Recall@10 | Recall@50 | MRR@10 |
|---|---|---|---|---|
| `hybrid` vs `bm25` | +0.067 [+0.006, +0.123], 34 / 14 | +0.085 [+0.017, +0.153] | +0.131 [+0.064, +0.208] | +0.068 [+0.004, +0.131] |
| `hybrid_rerank` vs `hybrid` | +0.072 [+0.011, +0.134], 34 / 21 | +0.042 [−0.017, +0.110] | identical head | +0.080 [+0.010, +0.149] |
| `hybrid_rerank` vs `bm25` | +0.139 [+0.081, +0.201], 38 / 12 | +0.127 [+0.068, +0.203] | +0.131 [+0.064, +0.208] | +0.148 [+0.081, +0.216] |

The two steps validation promoted hold on the locked test with intervals above zero:
fusion over BM25, and reranking over fusion. Validation had them at +0.055 and +0.054 on
nDCG@10; the test split gives +0.067 and +0.072. End-to-end p95 0.81 s against the 3 s
budget, rerank-stage p95 785 ms.

## Our corpus — the in-domain slice under the shipped first stage

Run `m2-release-test-20261006T091458Z` (`reports/retrieval/m2-release/test/`,
`configs/experiments/m2-release.yaml`), the 48 in-domain test queries: the four baselines
under `configs/search.yaml` as shipped, plus the first stage P2.5 had shipped (titles and
abstracts) for the record. The default mode, `hybrid_rerank`, had 0 failed and 0 degraded
queries.

| Variant | Recall@10 | Recall@50 | nDCG@10 | MRR@10 | Failed | p50 / p95 |
|---|---|---|---|---|---|---|
| `bm25` | 0.688 [0.562, 0.833] | 0.792 [0.667, 0.896] | 0.494 [0.381, 0.607] | 0.433 [0.317, 0.546] | 0 | 164 / 259 ms |
| `dense` | 0.458 [0.312, 0.604] | 0.604 [0.458, 0.750] | 0.343 [0.218, 0.466] | 0.307 [0.189, 0.428] | **12** | 475 / 1,001 ms |
| `hybrid` | 0.771 [0.646, 0.876] | 0.875 [0.771, 0.958] | 0.630 [0.515, 0.746] | 0.587 [0.463, 0.713] | 0 | 61 / 101 ms |
| **`hybrid_rerank`** (shipped) | **0.750** [0.625, 0.875] | **0.875** [0.771, 0.958] | **0.582** [0.467, 0.704] | **0.528** [0.409, 0.660] | 0 | 623 / 737 ms |
| `hybrid_rerank_papers` (before P2.6) | 0.708 [0.583, 0.833] | 0.792 [0.667, 0.896] | 0.558 [0.445, 0.683] | 0.510 [0.393, 0.641] | 1 | 592 / 696 ms |

Paired differences (computed from the recorded per-query rows with the harness's own
`paired`; the config has no decision block, so the harness wrote none):

| Comparison | nDCG@10 | Recall@10 | Recall@50 |
|---|---|---|---|
| `hybrid_rerank` vs `hybrid_rerank_papers` (P2.6's change) | +0.024 [−0.043, +0.088], 11 / 3 | +0.042 [−0.042, +0.146], 4 / 2 | +0.083 [−0.021, +0.208], 6 / 2 |
| `hybrid_rerank` vs `hybrid` | −0.048 [−0.159, +0.060], 11 / 13 | −0.021 [−0.146, +0.104] | identical head |
| `hybrid` vs `bm25` | +0.136 [+0.063, +0.208], 20 / 2 | +0.083 [+0.021, +0.167] | +0.083 [+0.000, +0.167] |

- **The chunk-level first stage points the same way as validation, on 48 queries that
  cannot confirm it.** Gold in the pool rises from 87.5% to 93.8%, Recall@50 from 0.792
  to 0.875, nDCG@10 by +0.024 with 11 wins to 3; every interval includes zero. Validation's
  +0.135 Recall@50 and +0.060 nDCG@10 (P2.6) remain the decision's evidence; the test
  neither contradicts nor strengthens it beyond what 48 queries can.
- **On this slice the reranker did not help the top 10**: `hybrid` scores above
  `hybrid_rerank` by 0.048 nDCG@10, 11 wins to 13, interval [−0.159, +0.060]. Development
  (150 queries, P2.5: reranking +0.082) and validation (52) both favoured reranking, and
  LitSearch's 118 test queries do above. A 48-query slice disagreeing within its interval
  is noise to record, not a reversal to act on; it is one more reason E4's graded
  in-domain queries matter.
- **`dense` alone failed 12 of 48 queries at its 1 s branch deadline.** They are the first
  queries the variant ran, right after another variant and another run had read other
  parts of the on-disk chunk collection: a cold page cache over the 26 GB collection puts
  the first dozen grouped dense queries past 1 s, after which they take 100 ms. Inside
  `hybrid` and `hybrid_rerank`, which ran afterwards, no dense branch timed out, and on
  validation and the development retry `dense_chunks` alone timed out 0 and 2 times. This
  is an operational property of the adopted first stage, recorded here and in the tracker
  as an open item: the service warms each branch once at startup, and a chunk-level dense
  branch needs a warmer cache than that, or its quantized vectors kept in RAM, before a
  demo on a cold host. It does not change the release decision: the default mode never
  degraded, and a cold dense branch degrades `hybrid` explicitly to BM25 with
  `dense_timeout`, which is the designed fallback.
- `hybrid_rerank_papers` lost one query to both branches timing out at once
  (`candidates_unavailable`, 1.01 s), a host hiccup with nothing else on the GPU.

## Release decision

**G2 is met and the M2 release is `hybrid_rerank` over chunk-level candidates**, as
validation chose and as configured. All four baselines were measured on one frozen
snapshot with Recall@50, MRR@10, nDCG@10, judged coverage and p50/p95; hybrid and
reranking were promoted on validation with intervals above zero and a p95 inside the
budget; the locked test confirms both steps on LitSearch's corpus. The in-domain locked
slice is too small to confirm anything on its own and is reported as such.

Recorded with the decision:
- the test split is spent for this release; a future configuration change needs its own
  release and the next locked run;
- open item: cold-cache latency of the chunk-level dense branch (above);
- the remaining first-stage gap on LitSearch's corpus (21% never pooled on validation,
  18% on test: `hybrid_rerank` candidate recall 0.898 at 100 per branch) is a model
  problem, with GritLM-7B as the measured gap, and belongs to M6;
- judged coverage of about 6% at 10 means every number here is a floor; E4's 30 graded
  in-domain queries are the mitigation.

## Commands

```text
uv run --env-file .env --project backend python -m copilot.cli eval retrieval \
  --config configs/experiments/e3-litsearch.yaml --split test --locked-test --out reports/retrieval/e3-litsearch
uv run --env-file .env --project backend python -m copilot.cli eval retrieval \
  --config configs/experiments/m2-release.yaml --split test --locked-test --out reports/retrieval/m2-release
uv run --project backend python -m copilot.cli eval report --out reports/retrieval/e3-litsearch
uv run --project backend python -m copilot.cli eval report --out reports/retrieval/m2-release
```

Before each run, four GPU samples 5 s apart at 8–53% utilization with 1,475 MiB held by
the idle Windows-side process and no Linux compute process; after each, 0% and the same
process only.
