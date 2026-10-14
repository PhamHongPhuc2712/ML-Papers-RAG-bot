# What the search page's reranker reads (S2, 2026-10-12)

**Question.** Should the search page's cross-encoder read the passage that found each paper
(title + chunk, haiku.rag's mechanism, +0.180 nDCG@10 on Open RAG Bench) instead of the title
and abstract (`rerank.text: abstract`, shipped)?

**Answer: no, by the registered rule. `abstract` stays.** On S1's section-level questions,
reading the chunk roughly doubles nDCG@10 (0.350 → 0.698). On the paper-level guard set
(LitSearch's in-domain queries), the candidate's interval could not rule out a loss larger than
0.03, and it degraded one of those queries. The rule needs both guards, so nothing is promoted
and the locked test split is not run. The other chunk-reading variant would also have failed
the guard.

*LitSearch was retired on 2026-10-14 (the developer's decision). The paper-level guard's
results, its config (`m5-rerank-text-papers.yaml`) and its run folder were deleted with it, and
git history keeps them. The decision above was made on them and stands as recorded; it can no
longer be re-measured.*

## The rule (registered before any number, commit `f7d2f2e`)

From `configs/experiments/m5-rerank-text.yaml`:

- **Candidate.** Of the two chunk-reading variants, the one with the higher section-level
  validation nDCG@10.
- **Promotion over `hybrid_rerank`.** Only if, on validation, all three hold:
  1. its section-level nDCG@10 gain has a paired 95% interval wholly above zero;
  2. on the paper-level set, its nDCG@10 interval rules out losing more than 0.03;
  3. its p95 is at most 3.0 s and it degraded no query, on both sets.
- **Locked test.** A promoted candidate is scored once on the section-level test split
  (`--locked-test`), and the promotion stands only if that interval is wholly above zero.

Variants, all over release `m2-20260924T095724Z` with chunk-level candidates:

| Variant | Reranker reads | Pair budget | Rerank deadline |
|---|---|---|---|
| `hybrid_rerank` | title + abstract (shipped) | 1,024 | 1.5 s |
| `title_chunk_1024` | title + the chunk that found the paper | 1,024 | 2.5 s |
| `title_chunk_512` | title + that chunk | 512 (`rerank.pair_max_tokens`, new in S2) | 1.5 s |

The harness's own verdict reads "hybrid_rerank" on both runs for a different reason. Its
`decision` block climbs in cost order and also refuses a step when the baseline degraded a
query, which `hybrid_rerank` did once on the section-level validation run. That verdict is not
the rule. The rule is applied below from the paired intervals against `hybrid_rerank`.

## Validation (the deciding read)

**Section level** (S1, 190 queries, `m5-rerank-text-validation-20261012T101403Z`):

| Variant | nDCG@10 | MRR@10 | Recall@10 | Recall@50 | p95 ms | rerank p95 ms | Degraded |
|---|---|---|---|---|---|---|---|
| `hybrid_rerank` | 0.350 [0.289, 0.404] | 0.295 | 0.526 | 0.842 | 725 | 558 | 1 (+1 failed) |
| `title_chunk_1024` | **0.698** [0.639, 0.756] | 0.673 | 0.774 | 0.842 | 1,285 | 1,190 | 0 |
| `title_chunk_512` | 0.664 [0.604, 0.722] | 0.631 | 0.768 | 0.842 | 711 | 652 | 0 |

Paired nDCG@10 against `hybrid_rerank`:
- `title_chunk_1024`: +0.348 [+0.292, +0.408].
- `title_chunk_512`: +0.314 [+0.253, +0.376], 101 wins / 17 losses.

**Paper level:** removed with LitSearch on 2026-10-14 (see the note at the top).

**The rule, applied to the candidate `title_chunk_1024`** (0.698 > 0.664):

| Condition | Value | Holds |
|---|---|---|
| (1) section-level gain wholly above zero | +0.348 [+0.292, +0.408] | yes |
| (2) paper-level loss ruled out beyond 0.03 | (removed with LitSearch) | **no** |
| (3) p95 ≤ 3.0 s on both sets | 1.28 s / (removed) | yes |
| (3) no degraded query on both sets | 0 / **1** (paper level, removed) | **no** |

**Not promoted.** `configs/search.yaml`, `configs/chat.yaml` and spec §7 are unchanged. The
chat keeps its own `search.overrides` (title + chunk at a 2.5 s rerank deadline, from O5),
which this decision does not touch.

## Development (read first, decided nothing)

190 queries, `m5-rerank-text-development-20261012T100315Z`:

| Variant | nDCG@10 | p95 ms | Degraded |
|---|---|---|---|
| `hybrid_rerank` | 0.374 | 1,182 | 4 (+1 failed) |
| `title_chunk_1024` | 0.682 | 1,250 | 0 |
| `title_chunk_512` | 0.654 | 735 | 0 |

No chunk-reading variant degraded a query, so the plan's early stop did not apply.
`hybrid_rerank` ran first against a cold Qdrant page cache: its lexical p95 was 267 ms against
64–70 ms for the variants after it. That is the cold-start gap already recorded in
`reports/m3-chat.md` (and G2's locked-test record, deleted with LitSearch).

## What this says

- **On passage questions the chunk is what matters,** reproducing ORB on our own corpus. The
  gain is in the head (MRR@10 0.30 → 0.67); Recall@50 is identical because the candidates are
  the same.
- **On paper-level queries the evidence was too thin to clear the guard.** The guard set was
  small, so ruling out a 0.03 loss needed a nearly loss-free candidate. It was underpowered, the
  way C7's rule was. The registered rule still decides, and this report does not reinterpret it.
- **Latency.** At 512 tokens under the shipped 1.5 s deadline, the chunk reader costs the same
  as today's reranker (p95 0.71 s vs 0.73 s). At 1,024 tokens it needs the longer deadline and
  still degraded a query.
- **Revisiting the decision.** That needs a new registration and a new paper-level set, since
  LitSearch's was retired.
  This validation split has now been read. S1's test split stays unread, so it is still
  available for a later registered decision.

## Commands, GPU and spend

```text
eval retrieval --config configs/experiments/m5-rerank-text.yaml --split development --out reports/retrieval/m5-rerank-text
eval retrieval --config configs/experiments/m5-rerank-text.yaml --split validation --out reports/retrieval/m5-rerank-text
```

- **GPU.** Every run started behind the idle gate: 30 one-second samples with a median of 0%
  (10:03, 10:13 and 10:24 UTC) and no Linux compute process. Each run's own sample read
  0–1% busy beforehand and saw no other GPU process during the run, so the GPU was used
  only by these runs.
- **Spend.** No LLM call: $0.
- **Tests.** `test_the_reranker_pair_budget_is_optional_and_validated`, plus the search config
  and service suites (117 passed).
