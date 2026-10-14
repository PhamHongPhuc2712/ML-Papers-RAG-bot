# Follow-up answering on ORB conversations (C7, 2026-10-11)

**Question.** A chat follow-up such as "What is it and why is it significant?" only makes sense
after the previous turn. Should the chat condense a follow-up into one standalone query before
it searches (`configs/chat.yaml condense`)? Pre-registered rule
(`configs/experiments/orb-chat-followups.yaml`, frozen split only): condensing stays on if
`condensed` is more accurate than `raw` with exact McNemar p < 0.05; otherwise it is turned
off.

**Answer: by the rule as written, condensing is turned off** (`condense: false`). On 43 frozen
conversations `condensed` read 0.791 against `raw`'s 0.674 (8 cases only condensed got right,
3 only raw, p = 0.227). The run is underpowered (see Limitations), and every secondary
measure favours condensing. This is a rule outcome, not evidence that condensing does not
help. A later recorded decision, or a larger conversation set, can revisit it.

**Revisited 2026-10-14: condensing is on (`condense: true`), the developer's recorded decision**
over the rule's outcome, since every measure here points the same way. On the live corpus the
follow-up from the W4 walkthrough, "How large was its batch size?", was then searched as "How
large was SimCSE's batch size?" and answered with a cited claim, where with condensing off it had
come back insufficient. The numbers below are unchanged.

## Design

- **Conversation.** Two `retrieval_only` questions about the same ORB paper, A then B: no
  answering split has used them. Turn 2 is B rewritten as a follow-up B′ by `gpt-6-luna`.
  The repository keeps ids and each rewrite's sha256 (`data/fixtures/chat/conversations.jsonl`,
  58 lines). Texts live under `${DATA_DIR}/eval/chat/`. ORB is CC-BY-NC-4.0, so no text is
  quoted here.
- **Conditions,** all over the same turn 1 (A, answered once per conversation through the
  chat's graph):
  - `standalone`: B asked with no history (the ceiling).
  - `raw`: B′ with history, not condensed (the follow-up is searched as typed).
  - `condensed`: B′ with history, condensed first.
- **Stack.** The chat's own graph over release `orb-v1`: title-and-chunk rerank at 2.5 s, 10
  papers, `configs/evidence.yaml`, `StructuredGenerator` (C4's decision), and
  `prompts/chat/condense-v1.yaml`. Purpose is `evaluation`. Each turn-2 answer is judged
  against B's reference by `openai-gpt-6-luna-low` under haiku.rag's rubric
  (`prompts/judge/answer-equivalence-haiku-v1.yaml`). An abstention counts as wrong.

## How the conversations were built, and why that changed before frozen scoring

The plan's first build (`followup-rewrite-v1`, commit `1e5f5f3`) told the model to replace
every name. It kept 99 development and 197 frozen conversations ($0.008).

**The development read** (`orb-chat-followups-development-20261011T171618Z`) gave a result
that made no sense:

| Condition | Accuracy [Wilson] | Gold in top 10 |
|---|---|---|
| standalone | 0.939 [0.874, 0.972] | 0.980 |
| raw | 0.677 [0.580, 0.761] | 0.798 |
| condensed | 0.586 [0.487, 0.678] | 0.980 |

Condensing found the paper as often as `standalone` yet answered worse than `raw`. Reading
the development cases (allowed by the plan) showed why. v1 had also removed B's *own* subject:
the method, acronym or quantity B asks about, which A never mentions. At least 30 of 99
development and 40 of 197 frozen follow-ups lost such a term (distinctive tokens only, so a
lower bound). Those B′ no longer say what they ask about. The condenser then correctly
resolved "this technology" to turn 1's subject, and the judge scored the answer against B's
reference. That contradicts the builder's own definition: only names B shares with A are
the ones a follow-up replaces.

Two amendments, committed before any frozen scoring:

1. `b1d2c41`: `followup-rewrite-v2` replaces only names A also uses, and the builder rejects
   (one retry, then drops) a rewrite that loses a distinctive term of B that A never names.
   The rebuild kept 74 / 175 ($0.011), but 53 of 74 development and 137 of 175 frozen B′ came
   back unchanged. B shared nothing with A, so those conversations did not test follow-ups.
2. `14272b2`: pairs are drawn, per paper, among questions where B shares a distinctive name
   with A. Only 78 of the 315 papers with two or more unused questions have such a pair.
   Development takes 25 pairs and frozen the remaining 53. After rejections, **15
   development and 43 frozen conversations** were kept ($0.003). Every B′ differs from B, and
   none drops B's subject.

## Results

Development (15, `orb-chat-followups-development-20261011T181013Z`, read first, decided
nothing):

| Condition | Accuracy [Wilson] | Gold in top 10 | Cited MAP |
|---|---|---|---|
| standalone | 1.000 [0.796, 1.000] | 1.000 | 1.000 |
| raw | 0.667 [0.417, 0.848] | 0.733 | 0.733 |
| condensed | 0.867 [0.621, 0.963] | 1.000 | 0.933 |

**Frozen (43, `orb-chat-followups-frozen-20261011T181643Z`, scored once):**

| Condition | Accuracy [Wilson] | Gold in top 10 | Cited MAP | Abstained |
|---|---|---|---|---|
| standalone | 0.930 [0.814, 0.976] | 1.000 | 1.000 | 0.000 |
| raw | 0.674 [0.525, 0.795] | 0.814 | 0.733 | 0.116 |
| condensed | 0.791 [0.648, 0.886] | 1.000 | 1.000 | 0.000 |

Pairings (`eval pair`, all 43 judged in each):

- **condensed vs raw (the rule):** both right 26, neither 6, condensed only 8, raw only 3,
  exact McNemar **p = 0.227**. The smallest discordant split that would have rejected is 10 / 1.
  Cited MAP sign test: 12 up, 0 down, 31 ties, p = 0.0005 (reported, deciding nothing).
- **condensed vs standalone (the price of a follow-up):** both 32, neither 1, condensed only 2,
  standalone only 8, p = 0.109.

Condense failures: 0 in every run. Cost: frozen $0.105 for 43 conversations (four turns and
three judgements each), about $0.0006 a chat turn including the judge.

## Decision

`configs/chat.yaml condense: false`, applying the registered rule. Condensing raised accuracy
by 0.116, retrieval of the gold paper from 0.81 to 1.00, and abstention fell from 0.12 to 0.
None of these reached the registered bar on 43 conversations. With `condense: false` the
graph searches a follow-up as typed. Turn 1 and standalone questions are unaffected.

## Limitations

- **Power.** 43 frozen conversations, against the plan's 200. ORB's questions rarely name
  what they share, which caps genuine follow-ups at 78 papers. A difference of the size
  observed would need roughly 10 / 1 discordant cases to reject.
- **Synthetic follow-ups.** Each B′ is written by a model, with one follow-up per
  conversation, and every follow-up replaces a name. Real follow-ups ("and on CIFAR?") are
  more varied.
- **The judge.** It is a model sharing the generator's provider, calibrated by model-model
  agreement for `orb-text-qa-v2`, not by human labels.
- **Re-registration.** The registration was amended twice after a development read, both times
  for the conversation set and never for the condense prompt or the rule. The frozen
  conversations were never scored under v1 or the first v2.

## Commands, GPU and spend

```text
eval chat-conversations --database-url .../copilot_orb --max-spend-usd 0.40      # three builds: v1, v2, v2 + shared-name pairs
eval chat --config configs/experiments/orb-chat-followups.yaml --split development --database-url .../copilot_orb --out reports/chat --max-spend-usd 1.95   # v1 conversations
eval chat --config configs/experiments/orb-chat-followups.yaml --split development --database-url .../copilot_orb --out reports/chat --max-spend-usd 0.50
eval chat --config configs/experiments/orb-chat-followups.yaml --split frozen --frozen --database-url .../copilot_orb --out reports/chat --max-spend-usd 1.00
eval pair --treated reports/chat/frozen/<run>.condensed/cases.parquet --baseline reports/chat/frozen/<run>.raw/cases.parquet
eval pair --treated reports/chat/frozen/<run>.condensed/cases.parquet --baseline reports/chat/frozen/<run>.standalone/cases.parquet
```

- **Budget flags.** The plan's 0.20 / 1.00 / 2.00 sit below the runners' own worst case
  (`(1 condense + 4 generations) × 2 + 3 judge calls` a conversation at the full output budget,
  $0.0191). The flags passed are the computed ceilings above, and the $3 daily cap bounded
  actual spend.
- **GPU.** Every run passed the idle gate (30 one-second samples, median ≤ 20%, no Linux
  compute process). The frozen run's own sample read a 0% median, `busy: false`.
- **Spend.** Ledger for C7, purpose `evaluation`: builds $0.0080, $0.0106 and $0.0033;
  development $0.2388 (v1) and $0.0349; frozen $0.1055. Total $0.401.
