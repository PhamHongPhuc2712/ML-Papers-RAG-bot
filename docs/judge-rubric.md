# The answer-equivalence judge

How a generated answer is scored on Open RAG Bench's text slice (ORB plan O5), and how a
human labels the calibration slice the same way.

## The rubric

The prompt is `prompts/judge/answer-equivalence-v1.yaml`, pinned by digest; its rules follow
haiku.rag's answer-equivalence rubric (ggozad/haiku.rag, MIT) in our own wording.

A candidate answer is **equivalent** to the reference answer when it carries the
reference's core facts and conclusions, in any wording. Extra correct detail is neutral.
Numbers and names must agree. It is **not equivalent** when it contradicts the reference,
omits something that changes its meaning, or does not address the question. Length and
confidence are not evidence either way: a shorter answer stating the same facts is
equivalent; a longer, surer answer with a different fact is not.

The judge writes its reason first and its verdict second, as JSON:
`{"reason": "...", "equivalent": true}`. The order is deliberate: the reason is composed
before the verdict is committed, and a wrong verdict is legible.

## How the judge runs

`gpt-6-luna` at reasoning effort `low` (`openai-gpt-6-luna-low`), the same provider that
generates the answers at effort `none`. A malformed verdict is retried once with a
reminder (which changes the request, so a cached malformed answer is not served again);
a second failure is `judge_error`, excluded from the accuracy denominator and counted. An
abstention (no claims, insufficient evidence) is scored wrong without a judge call.

## Labelling the calibration slice

Fifty development answers are hand-labelled **blind**: the labeller sees the question,
the reference answer and the generated answer, and not the judge's verdict or reason.
Label with the rubric above, one line per case in JSONL:

```text
{"query_id": "<id>", "equivalent": true}
```

`eval calibration --labels <file> --cases <run>/cases.parquet` reports raw agreement,
Cohen's κ, the confusion matrix and the disagreeing ids. Below κ = 0.6 the accuracy numbers
are reported as uncalibrated and the rubric is fixed before the frozen run; the labels are
never adjusted to the judge. Every disagreement is read.

The labelling sheet for a run is written by `eval qa-labels` under `${DATA_DIR}/eval/qa/`
(the text never enters the repository); the labels file goes beside it.
