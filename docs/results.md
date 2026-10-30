# Results

Every number here was measured on this project's own stack. Comparisons were registered before
their numbers were read: a rule was committed with its configuration (`configs/experiments/`),
development data tuned it, and a held-out test set was scored **once**. Intervals are 95%:
bootstrap over query families or conversations, Wilson for single accuracies. Accuracy is the
share of answers an LLM judge (`gpt-6-luna`, reasoning effort low, haiku.rag's answer-equivalence
rubric) marks equivalent to the dataset's reference answer.

- [Datasets](#datasets)
- [Corpus](#corpus)
- [Retrieval](#retrieval)
- [Answering](#answering)
- [The research agent](#the-research-agent)
- [Agent follow-ups: a stricter skip rule and a planning step](#agent-follow-ups)
- [Self-check (paused)](#self-check-paused)
- [Speed, cost and reliability](#speed-cost-and-reliability)
- [The judge](#the-judge)

## Datasets

| Dataset | What it is | Size used | Licence |
|---|---|---|---|
| [Open RAG Bench](https://huggingface.co/datasets/vectara/open_ragbench) (ORB) | Vectara's 1,000 arXiv papers and their questions, single-turn | 1,914 text questions; 100 development, 400 frozen, a second frozen 400 | CC-BY-NC-4.0 |
| [MTRAG](https://github.com/IBM/mt-rag-benchmark) ClapNQ | IBM's human-written multi-turn conversations over Wikipedia | 29 conversations, 224 turns | CC-BY-SA-4.0 |
| MTRAG Govt | The same, over government web pages | 28 conversations, 214 turns | CC-BY-SA-4.0 |
| MTRAG FiQA | The same, over finance forum answers | 27 conversations, 199 turns | CC-BY-SA-4.0 |
| [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) | News questions whose evidence spans 2–4 articles | comparison and temporal: 30 development, 150 test | ODC-BY |

Each is fetched at a pinned revision into `DATA_DIR`; the repository keeps ids, labels and splits
(`data/fixtures/`), never question, answer or passage text. Each corpus is loaded into its own
database and index release, never the served one.

## Corpus

The served corpus is **85,729 accepted main-conference papers from 42 venue-years** (AAAI, ACL,
CVPR, ECCV, EMNLP, ICCV, ICLR, ICML, IJCAI, Interspeech, JMLR, NAACL, NeurIPS, WACV; 2023–2026),
parsed from source PDFs into **3,426,221 chunks** under `paragraph-pack-v1`.

- No chunk over the 1,200-token ceiling; ordinals contiguous in all 81,929 documents with text.
- Body prose ending mid-sentence: 15.2% (the fixed-window chunker it replaced: 47.2%).
- 3,802 papers (4.4%) have no text; 3,759 of them are zero-byte PDFs at the source (NeurIPS 2025,
  ICLR 2026). See the [data card](data-card.md).
- Bibliography chunks that had been stored as body text were relabelled, and 13 whole proceedings
  volumes were excluded from search.

## Retrieval

**ORB, all 1,914 text questions** (the paper that holds the answer is the target):

| Mode | nDCG@10 |
|---|---|
| `hybrid` (BM25 + dense, RRF) over chunk-level candidates | **0.948** [0.936, 0.959], MRR@5 0.937 |
| Chunk-level candidates against title-and-abstract candidates | +0.22 to +0.25 in every mode |

haiku.rag reports MRR@5 0.977–0.990 on the same questions with 4B embedding models.

**What the cross-encoder reads.** Reranking on titles and abstracts lost 0.163 nDCG@10 to plain
fusion on ORB's section-level questions. Reading the passage that found each paper fixed that
(+0.180 over the abstract reader). The served reranker reads **title + abstract + passage**
(+0.015 to +0.056 nDCG@10 over title + passage on development and validation reads, −0.002 on ORB).

**Chunk size (ORB).** 256-token chunks gain +0.011 nDCG@10 for plain fusion over 800-token ones
but cost answer accuracy (0.79–0.84 against 0.87), so the served corpus keeps ~800-token chunks.

**MTRAG retrieval** (judged turns, the served first stage, no answers generated):

| Domain | Last turn as written | IBM's rewritten turn |
|---|---|---|
| Govt (201 turns) | nDCG@10 0.418, recall@10 0.50 | 0.488, 0.62 |
| FiQA (180 turns) | 0.327, 0.41 | 0.396, 0.49 |

## Answering

**ORB, frozen 400**, the condense → search → answer pipeline:

| Configuration | Accuracy |
|---|---|
| `flat`: one search with the question as written | **0.880** [0.844, 0.908] |
| `section`: passages widened to their section | 0.865 (McNemar p 0.26; `flat` kept) |
| `flat` with structured output (`ChatOpenAI` JSON schema) | **0.884** [0.849, 0.912], p 0.82, adopted |

Of `flat`'s 48 errors, 35 had the gold paper in hand. On the 100 development questions the
reranker reading abstracts had lost the gold paper for 16; reading title + passage, it was in the
top 10 for 99, and accuracy rose from 0.77 to 0.87 (McNemar p 0.021).

**Follow-up questions** (43 frozen ORB question pairs written into conversations): condensing a
follow-up into a standalone query 0.791 [0.648, 0.886], searching it as typed 0.674 [0.525,
0.795], an oracle standalone question 0.930. McNemar p 0.227, short of the registered rule;
condensing stayed on as a decision over that underpowered result.

## The research agent

A LangGraph tool-calling agent: `search_papers` (the served first stage, up to 3 times),
`read_section` (up to 4), `answer`; at most 8 model calls. It may answer a follow-up from passages
earlier answers cited, and hands the passages it chose to the same grounded generator. It
**serves the chat since 2026-10-30** (`configs/chat.yaml`), a decision taken over the results below.

| Test set (scored once) | Agent | Pipeline | Difference, McNemar |
|---|---|---|---|
| MTRAG ClapNQ, 24 test conversations, live history | 0.636 | 0.588 | +0.048 [−0.005, +0.103], p 0.163 |
| MTRAG ClapNQ, reference history | 0.684 | 0.583 | +0.102 [+0.057, +0.149], p 0.0005 |
| MTRAG Govt, 28 conversations, live history | 0.556 | 0.528 | +0.028 |
| ORB frozen 400 | 0.890 | 0.890 | 7 against 7, p 1.0; cited MAP 0.959 / 0.956 |

The registered rule (live history, p < 0.05 and interval above zero) was not met on ClapNQ, the
primary test. With reference history (each turn after the people's own conversation) the agent's
lead is clear; with its own earlier answers as history it is not. On ClapNQ's live turns the agent
skipped searching on 21% of follow-ups and was right 0.51 of the time then, against 0.64 when it
searched.

## Agent follow-ups

**A stricter skip rule** (`research-v3`: answer without searching only when a carried passage
states the answer outright). Development, all of ClapNQ: v3 0.621 against v2 0.571 (live), skip
rate 15% against 19%, but skipped turns stayed ~10 points worse than searched ones.

| Govt test, scored once | Pipeline | Agent v2 | Agent v3 |
|---|---|---|---|
| Live history | 0.528 | 0.556 | 0.561 |
| Reference history | 0.523 | 0.593 | 0.533 |
| Correct when skipping / when searching (live) | — | 0.59 / 0.51 | 0.61 / 0.53 |

v3 against the pipeline, live: +0.033 [−0.011, +0.098], p 0.36; against v2: +0.005 live, −0.061
[−0.103, −0.001] with reference history (p 0.03). **Not adopted.** The ORB guard held (0.895
against 0.890). On Govt, skipping was not the weak point it was on ClapNQ.

**A planning step** (one call splits a question into at most 3 sub-questions before the loop):

| MultiHop-RAG test, 150 questions | Pipeline | Agent | Agent + plan |
|---|---|---|---|
| Accuracy | 0.720 | 0.767 | 0.787 |
| Comparison / temporal (of 75) | 57 / 51 | 62 / 53 | 63 / 55 |

Plan against no plan: 4 against 1, p 0.375. **No evidence planning helps**; the agent already found
the evidence articles (coverage 0.87 without a plan, 0.89 with), and its misses are reasoning about
dates. Planning stays off.

## Self-check (paused)

After the answer is written, one call checks each claim against the passages it cites; unsupported
claims are rewritten once, and any still unsupported are dropped. A separate judge scores
faithfulness (the share of claims their cited passages support). Development reads only; the
registered test sets (a second ORB frozen 400 and MTRAG FiQA) were not scored, and the check is off.

| Development | Agent | Agent + self-check |
|---|---|---|
| ORB, 100 questions: accuracy / supported claims | 0.939 / 0.942 | 0.930 / 0.978 |
| MultiHop-RAG, 30 questions: accuracy / supported claims / abstained | 0.633 / 0.81 / 3% | 0.667 / 0.94 / 17% |

The checker and the faithfulness judge are the same model family, so part of the gain may be the
two agreeing.

## Speed, cost and reliability

| Govt test, live history | Pipeline | Agent |
|---|---|---|
| Seconds per turn, p50 / p95 | 5.0 / 7.4 | 7.3 / 12.5 |
| Cost per turn, answer and judge | $0.0007 | $0.0015 |

On ORB the agent cost about 4× the pipeline per answer. Every priced call reserves its worst case
against a daily cap before it is sent.

**Failure drills** on the local stack: 7 of 8 pass, 1 was not possible; one hidden failure (a
database outage on the chat page shown as an empty history) was found and fixed. After a Qdrant
restart the search warm-up took 138 s, otherwise about 38 s; the first live searches after a cold
start can still time out a branch.

## The judge

In place of hand labels, a second model (`gpt-6-sol`) re-judged the judge of record's verdicts:
95.5% agreement, Cohen's κ 0.807 on 89 cases, above the registered 0.6 floor. Model-model
agreement is weaker evidence than human labels. The judge shares a provider with the generator,
a stated limitation. Rubric: [judge-rubric.md](judge-rubric.md).
