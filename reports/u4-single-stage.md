# U4: the single first stage, measured, and the real-stack walk (2026-10-16)

RAG-only plan, task U4. Since U1 the one first stage is `configs/search.yaml`: chunk-level
candidates, RRF, and the cross-encoder reading **title + the chunk that found the paper** at a
2.5 s rerank deadline. This task measures it on our corpus's two model-written query sets and
walks the real stack once. **It is a measurement, not a decision**: the setting is the developer's
choice (U1), and nothing here changes it.

## Results in plain words

- On **questions about a passage** (S1) the shipped stage ranks the right paper far higher than
  the old search page's reranker (title + abstract): nDCG@10 0.698 against 0.350 on validation.
- On **"find papers about X" requests** (S3) it ranks the right paper lower than the old reranker:
  0.563 against 0.663, a difference of 0.100 [0.070, 0.130]. Both sets were written by a model,
  and each leans toward one reranker (below), so the size of both gaps is uncertain; the
  direction agreed on development and validation.
- **No reranker at all** is worse than the shipped stage on both sets.
- The **real-stack walk** passed: a paper-finding request, a question and a condensed follow-up
  each got a cited answer with its References. In the paper-finding answer, three unrelated
  3D-generation papers ranked 3rd to 5th and were among the 10 papers read.

## The two query sets, and what they can and cannot say

Both were written by `gpt-6-luna`; nobody wrote or checked a query by hand.

| Set | Built | How a query was made | Gold |
|---|---|---|---|
| **S1**, 571 passage questions (`data/fixtures/retrieval-sections/`) | 2026-10-12 | the model read one passage of one paper and wrote a question it answers (`prompts/eval/section-question-v1.yaml`) | that paper |
| **S3**, 1,048 topic queries (`data/fixtures/retrieval-topics/`) | 2026-10-15 | the model read one paper's **abstract** and wrote the 5–15-word query a researcher might type to find papers like it, without names or copied phrases (`prompts/eval/topic-question-v1.yaml`) | that paper |

Two limits apply to every number below:

1. **Each set favors the reranker that reads its source text.** S3's queries paraphrase
   abstracts, and the old reranker reads abstracts; S1's questions come from passages, and the
   shipped reranker reads passages. Recall@50 is identical across variants (the candidates are
   the same), so only the ordering differs, which is where that match would show.
2. **One gold paper per query.** A topic query has many good answers in 85,729 papers; every
   other relevant paper counts as a miss. This undercounts topic queries more than passage
   questions.

## Runs

`configs/experiments/u4-single-stage-{topics,sections}.yaml`, three variants each:
`shipped` (configs/search.yaml as it stands), `abstract_reranker` (the former search page:
title + abstract, 1.5 s rerank deadline) and `fusion_only` (no reranker). Development and
validation only; **the test splits of S1 and S3 stay unread**. The harness's `decision` field on
validation is computed mechanically and is not read. Release `m2-20260924T095724Z`, code
`112a98c`. Full tables: `reports/retrieval/u4-single-stage-topics/report.md` and
`reports/retrieval/u4-single-stage-sections/report.md`.

Every recorded run had an idle GPU before it (0% busy) and **degraded no query**. The first S3
development run was flagged contended (25% busy before it, from Windows-side desktop apps) and was
set aside and redone; its ranking metrics were identical, only its timings differed. A 30-query
warm-up after the morning's Qdrant restart ran first, outside `reports/`
(`${DATA_DIR}/eval/u4-warmup/`); 2 of its `shipped` queries degraded while cold.

### nDCG@10 (95% interval over query families)

| Set, split | queries | `shipped` | `abstract_reranker` | `fusion_only` |
|---|---|---|---|---|
| S1 development | 190 | **0.682** [0.619, 0.738] | 0.374 [0.317, 0.434] | 0.580 [0.524, 0.636] |
| S1 validation | 190 | **0.698** [0.639, 0.756] | 0.350 [0.289, 0.404] | 0.605 [0.542, 0.664] |
| S3 development | 349 | 0.526 [0.481, 0.571] | **0.617** [0.572, 0.661] | 0.491 [0.441, 0.535] |
| S3 validation | 349 | 0.563 [0.518, 0.610] | **0.663** [0.615, 0.706] | 0.512 [0.465, 0.557] |

### Paired against `shipped`, validation

| Set | Variant | nDCG@10 difference | wins / losses / ties | Recall@10 | Recall@50 |
|---|---|---|---|---|---|
| S1 | `abstract_reranker` | −0.348 [−0.408, −0.292] | 10 / 106 / 74 | 0.526 vs 0.774 | 0.842 = 0.842 |
| S1 | `fusion_only` | −0.093 [−0.142, −0.044] | 22 / 54 / 114 | 0.747 vs 0.774 | 0.842 = 0.842 |
| S3 | `abstract_reranker` | +0.100 [+0.070, +0.130] | 99 / 30 / 220 | 0.765 vs 0.705 | 0.811 vs 0.808 |
| S3 | `fusion_only` | −0.052 [−0.087, −0.017] | 63 / 95 / 191 | 0.656 vs 0.705 | 0.808 = 0.808 |

Recall@10 matters to the chat, which reads the top 10: on S3, the old reranker brings the gold
paper into those 10 for 6 more queries in 100.

### Latency of `rank()`, validation (ms)

| Set | `shipped` p50 / p95 | `abstract_reranker` p50 / p95 | `fusion_only` p50 / p95 |
|---|---|---|---|
| S1 | 1,114.9 / 1,283.5 | 550.7 / 600.6 | 39.9 / 49.1 |
| S3 | 1,078.4 / 1,273.8 | 555.1 / 602.8 | 50.2 / 60.4 |

All within the 3 s total. Reading chunks costs about twice the abstract reader's time.

## The real-stack walk

`frontend/e2e/real-stack.spec.ts`, rewritten for one page: sign in, ask "Find papers on
contrastive learning of sentence embeddings", open the References and the first paper's page,
then a new chat with "What loss does SimCSE use to train sentence embeddings?" and the follow-up
"How large was its batch size?". API (`uvicorn`, startup warm-up "ready" in 17.5 s over 7 queries
with 2 dense timeouts) and `next start` on this host, a throwaway account created for the walk
and deleted after it (`users create`, `users delete`).

| Attempt | UTC | Outcome | Stored turns |
|---|---|---|---|
| 1 | 05:46 | Passed (1.2 min), **set aside** | every turn degraded: `lexical_timeout` on the paper-finding turn (retrieve 33.7 s), `dense_timeout` on the other two (3.8 s, 3.4 s) |
| 2 | 05:48 | **Passed** (14.3 s), GPU idle (0%) | no warnings; retrieve 1.75 s, 2.34 s, 1.97 s; the follow-up condensed to "How large was SimCSE's batch size?" |

Screenshots of attempt 2: `reports/u4-web/real-papers.png` (the paper-finding answer with its 20
References open) and `reports/u4-web/real-chat.png` (the question, a cited passage, and the
follow-up). They show our corpus's own titles and passages.

What the screenshots show:
- The paper-finding answer has four cited claims on four sentence-embedding papers, and the
  header reads "Searched for … · read 10 papers". **References 3–5 are 3D-generation papers**
  (AnimatableDreamer, MinD-3D, VividDreamer), unrelated to the request, ranked above SimCSE++ and
  miCSE, and three of the ten papers read. With the search undegraded, this is the reranker's
  ordering: one concrete instance of what S3 measures.
- In attempt 1, degraded, the same three papers ranked 1st to 3rd, and the answer named SimCSE
  and DiffCSE (2021, 2022, older than the corpus) through other papers' related-work passages.
- One corpus title shows mis-decoded text ("CÂ²A-SLU"), a metadata encoding issue, not new here.
- The follow-up is answered (512 for supervised training, 64 as a default), each cited.
- Search degradation is not shown in the UI: the warnings recorded on attempt 1's turns have no
  user-facing label (`lib/chat-labels.ts` names only condensing and validation warnings).

**Spend.** Purpose `chat`, 2026-10-16: 4 calls, $0.00188 (attempt 2's answers to identical
evidence came from the response cache).

**Warm-up, again.** As in U2's hand read, startup reported ready and the first live turns still
degraded, here with a 33.7 s retrieve. R3's warm-up repeats a fixed query list and does not keep
new queries' index pages warm. Recorded, not fixed.

## What this leaves open

The developer's choice stands. Two follow-ups would turn S3's warning into evidence, each to be
registered before it runs:
- **A fairer paper-finding measure**: an LLM judge marks every top-10 paper of both rerankers as
  relevant or not for S3's validation queries, so a relevant paper that is not the source paper
  counts. Estimated under $1.
- If the old reranker still wins there, **a reranker reading title + abstract + the matching
  chunk** together, measured on S1 and S3 against a rule fixed in advance.
