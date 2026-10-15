# U2: an answer for paper-finding requests (2026-10-15)

RAG-only plan, task U2. The prompt `prompts/generate/grounded-overview-v1.yaml` answers a
question directly and a request for papers with an overview grouped by approach, under the
unchanged claims contract. Rule registered in `configs/experiments/orb-text-qa-u2-overview.yaml`
by commit `df8f88f`, before any run.

**Outcome: not adopted.** Part (a) of the rule passes; part (b), the hand read, finds one claim
whose cited evidence does not support it. By the rule, short cited claims stay
(`grounded-claims-v1`, `configs/chat.yaml` unchanged).

## (a) ORB development, both arms fresh the same day

| Arm | Run | Accuracy | Cited MAP | Provenance valid | Abstained | Cost |
|---|---|---|---|---|---|---|
| baseline `grounded-claims-v1` (v3-structured) | `orb-text-qa-v3-structured-development-20261015T094115Z` | 0.93 (93/100) [0.863, 0.966] | 0.97 | 0.99 | 0.03 | $0.000 (all cached) |
| treated `grounded-overview-v1` | `orb-text-qa-u2-overview-development-20261015T094556Z` | 0.94 (94/100) [0.875, 0.972] | 0.98 | 0.99 | 0.02 | $0.069 |

Intervals are Wilson. `eval pair`: 92 both right, 2 treated only, 1 baseline only, 5 neither;
McNemar p = 1.0. The rule needs treated ≥ baseline − 0.03 = 0.90: **passes**. Both arms read the
same papers and the same evidence ids on all 100 cases (checked case by case), so the prompt is
the only difference; the GPU was idle before each (`gpu_before_run`: busy false, 0% p50). The
baseline cost nothing because every call hit the response cache: its retrieval reproduced the
2026-10-11 run's evidence exactly, and its accuracy matches that run (0.93). Each arm ran with
`--max-spend-usd 0.75`; the config's suggested 0.60 is below the runner's worst case ($0.712) and
was refused before anything ran.

## (b) Hand read of 20 paper-finding requests

The first 20 of S3's development topic queries by query id, each through the chat turn as built
by `build_default_chat_graph` over the live corpus (`m2-20260924T095724Z`, the U1 first stage) with
the overview prompt, ledger purpose `evaluation`; run `u2-overview-20261015T095847Z`, $0.0129.
Queries, answers and evidence are under `${DATA_DIR}/eval/u2-overview/` (S3's text never enters
git). Read by Claude, claim by claim, against the full text of every passage each claim cites;
names were also checked mechanically against each answer's evidence.

- 20 answers, 71 claims, no abstention, no repair. All 20 read as overviews (2 to 5 claims, grouped
  by approach where the evidence held several papers; answers 2, 3, 5, 6, 14 and 19 rest on one paper).
- **No invented name**: every method, model and dataset name in a claim appears in its answer's
  evidence.
- **70 of 71 claims are supported** by the passages they cite.
- **One is not**: answer 14, query `t-090d6a00-127b-45f6-a88d-e939c918c361`, claim 2. It credits the proposed graph-factorization model with separating homophilous and
  heterophilous structure into different factors, citing a Related Work passage that begins
  mid-discussion ("The authors discuss how, interestingly, this model separates…") and whose next
  sentence contrasts "these decompositions" with "our exact factorization results". The passage
  describes a prior model. What the claim says may be true of the paper, but its citation does
  not support it: the rule's exclusion.
- Weaknesses below the rule's bar: answer 4 splits a compound request ("protect training data"
  with "few, unevenly distributed examples") into two unrelated groups without connecting them;
  answer 10's second claim comes from an image-to-audio paper, correct but off the VQA topic.

The failure's mechanism is a chunk whose antecedent ("this model") lies in the previous chunk. It
is not specific to the overview prompt, but an overview that groups claims across passages leans
on such sentences more than a direct answer does. No measurement here says how often
`grounded-claims-v1` makes the same error.

## Seen on the way

The first attempt at the hand read stopped on its first query with `candidates_unavailable`
(both branches timed out); the second ran after startup reported "search ready … 3 queries,
3.7 s", and its first three searches were still degraded (`lexical_timeout` and `rerank_timeout`,
then `dense_timeout` twice). The live corpus's collections had not been queried that day (the ORB
runs read `orb-v1`). R3's warm-up repeats a fixed list of queries, which can pass on pages
already cached while other queries still hit cold ones. Recorded, not fixed here. The degraded
searches still found papers; the answers to those three were read like the rest.
