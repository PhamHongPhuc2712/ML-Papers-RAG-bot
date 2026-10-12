# A section-level query set from our corpus (S1, 2026-10-12)

**Why.** On Open RAG Bench the cross-encoder reading the passage that found each paper beat the
one reading title and abstract by +0.180 nDCG@10. ORB's questions are each answered by one
passage of one paper. Our only in-domain set (LitSearch's in-domain slice) asks for papers,
not passages. S2 decides the search page's reranker text on our own corpus, so it needs
ORB-style questions written over our papers. This set is that: one body passage per sampled
paper, one model-written question per passage, the passage's paper as the single gold.

**What is stored where.** The questions are derived from corpus text, so they live under
`${DATA_DIR}/benchmarks/sections-v1/queries.jsonl`. The repository keeps ids, labels and
splits only (`data/fixtures/retrieval-sections/`): `grep -c '"query"' queries.jsonl` is 0.
This report quotes no question.

## The set

| | |
|---|---|
| Source | `copilot_v2`, active release `m2-20260924T095724Z` |
| Sampled | 600 passages, one per paper, 15 per venue-year, seed 42 |
| Venue-years | 40 of 42 (see Deviations: the 600 cap leaves out WACV 2025 and 2026) |
| Kept | **571** questions (29 dropped after one retry) |
| Splits, by paper | development 190, validation 190, test 191 |
| Question length | median 19 words (9–33) |
| `eval validate-dataset` | 571 queries, 571 qrels, 571 families, no family across splits |

Sampling (`sample_chunks`): chunks of kind `body` with at least 200 pinned-tokenizer tokens,
from parsed, unmerged papers. A section heading that matches `references` or `bibliography` is
excluded, since a few bibliographies are parsed as body text. One chunk per paper and 15
papers per venue-year are chosen by md5 of the id and the seed. The first 600 in venue-year
order are kept.

Question writing: `openai-gpt-6-luna` (effort `none`, temperature 0, seed 42), prompt
`prompts/eval/section-question-v1.yaml`, response cache on. Each question is checked
mechanically (`acceptable`):

- `names_paper`: it contains a distinctive title term (an acronym, a hyphenated or
  digit-bearing token) or two consecutive title content words (C7's `leaks`).
- `copies_passage`: it shares eight or more consecutive words with the passage.
- `empty`, `unparsable`.

A rejected question is retried once with the reason, then the sample is dropped.

Rejections in the kept build (`section-queries-20261012T095056Z`, 714 calls):

| Reason | Count |
|---|---|
| names_paper | 141 |
| copies_passage | 2 |
| unparsable | 0 |

## Read: 30 development questions beside their passages

Drawn with `random.Random(7)` from the development split and read against the full passage:

| Class | Count | Notes |
|---|---|---|
| Answerable from the passage, specific to it | **25** | |
| Ambiguous | **4** | Answerable from the passage, but a generic definition (total variation distance, catastrophic forgetting, concept satisfiability) or a reproducibility statement. Many papers would answer it equally well, so the single gold label undercounts relevant papers. |
| Wrong | **1** | A NeurIPS checklist passage ("does the paper present theoretical results?"): boilerplate, not a research question. |

1 of 30 wrong, under the plan's threshold of 5, so `section-question-v1` stands and no v2 was
written. Two of the 30 passages were NeurIPS checklist text, which our parser keeps as body.
Generic and checklist questions make the set harder for every variant alike, since each has
one gold paper. They do not favour either reranker text.

## Deviations from the plan, all before any S2 run

1. **Parser.** The first build (`section-queries-20261012T090436Z`, 1,095 calls, $0.0837)
   rejected 577 replies as `unparsable`. Asked for JSON, the model answered with the bare
   question about half the time, and those questions were fine. `_question` now also accepts
   a single non-JSON line, or a JSON string, as the question. The prompt is unchanged. The
   rebuild reused the cached first replies and kept 571 instead of 437. The first build's
   output was overwritten and nothing was scored on it.
2. **Bibliography sections.** A dry sample found 11 of 600 `body` passages under a
   "References" heading. Those headings are now excluded (the plan asked to check this).
3. **Budget flag.** The plan's `--max-spend-usd 0.60` sits below the command's own worst case,
   $0.81: every sample asked twice at the full 1,024-token output budget. The flag passed was
   0.85, and the $3 daily cap bounded actual spend.
4. **Venue-years.** 42 venue-years × 15 = 630 samples, cut to the plan's 600 in venue-year
   order, which drops WACV 2025 and WACV 2026 entirely. Kept as specified.

## Commands and spend

```text
uv run --env-file .env --project backend python -m copilot.cli eval section-queries \
  --database-url <copilot_v2> --per-venue-year 15 --max-spend-usd 0.85
uv run --project backend python -m copilot.cli eval validate-dataset --path data/fixtures/retrieval-sections
grep -c '"query"' data/fixtures/retrieval-sections/queries.jsonl   # 0
```

Ledger, purpose `evaluation`: first build $0.0837, rebuild $0.0068, **total $0.0905**. No GPU
was used.

Tests: `test_section_queries.py` (5: copied runs, rejection reasons, deterministic split,
a scripted `ChatClient` over three samples with one accepted, one accepted on retry and one
dropped, and the bare-question parser) and `test_section_sampling.py` (one body passage per
paper, deterministic, capped per venue-year).
