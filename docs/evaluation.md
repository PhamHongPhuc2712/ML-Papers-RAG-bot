# Evaluation

How the numbers in [results](results.md) are produced, and how to reproduce them.

## Method

- **Registered before read.** Each comparison is an experiment file in `configs/experiments/` whose
  header states its decision rule. It is committed before any test answer exists.
- **Development tunes, test decides once.** Development splits may be read and re-run while prompts
  settle. A test split (ORB's frozen samples, MTRAG's test conversations, MultiHop-RAG's test
  questions) runs only with `--frozen`, once per configuration; each scoring is recorded in a
  `frozen-scorings.json` beside the outputs.
- **Paired statistics.** Two systems answer the same questions: exact McNemar on the answers only
  one got right, and a bootstrap interval of the accuracy difference over query families or
  conversations.
- **An LLM judge**, haiku.rag's answer-equivalence rubric ([judge-rubric](judge-rubric.md)),
  calibrated against a second judge model.
- **Spend is bounded.** Every paid run takes `--max-spend-usd`, is refused when its worst case
  exceeds it, and is stopped by the ledger once its real spend does.
- **Idle GPU.** Every run samples the GPU before it starts and marks itself contended when
  another process kept it busy; timing evidence is taken only on an idle GPU.
- **No third-party text in git.** Questions, answers and passages stay under `DATA_DIR`; the
  repository keeps ids, labels, splits and metrics.

## Datasets

Each is fetched once, at a pinned revision, into `$DATA_DIR/benchmarks/`, and loaded into its own
database and never-activated index release.

```bash
CLI="uv run --env-file .env --project backend python -m copilot.cli"

# Open RAG Bench: 1,000 arXiv papers through our own parser and chunker
$CLI eval orb-fetch
$CLI eval orb-load --database-url .../copilot_orb
$CLI eval orb-dataset --out data/fixtures/orb --database-url .../copilot_orb

# MTRAG: one domain at a time (clapnq, govt, fiqa); each passage becomes one paper with one chunk
$CLI eval mtrag-fetch --domain govt
$CLI eval mtrag-load --domain govt --database-url .../copilot_mtrag_govt

# MultiHop-RAG: 609 news articles, comparison and temporal questions
$CLI eval multihop-fetch
$CLI eval multihop-load --database-url .../copilot_multihop
```

Then export and index each corpus as in [setup](setup.md), with `--release` set to the release
the dataset names (`orb-v1`, `mtrag-govt-v1`, `multihop-rag-v1`, …) and `--database-url` to its
database.

## Running

```bash
# Retrieval only, no LLM spend
$CLI eval retrieval --config configs/experiments/orb-text-retrieval.yaml --split retrieval \
  --database-url .../copilot_orb --out reports/retrieval/orb-text
$CLI eval mtrag-retrieval --domain govt --queries rewrite --database-url .../copilot_mtrag_govt \
  --out reports/retrieval/mtrag-govt/rewrite --max-spend-usd 0

# Single-turn answering (ORB, MultiHop-RAG): development, then the frozen split once
$CLI eval qa --config configs/experiments/orb-text-qa-agent.yaml --split development \
  --database-url .../copilot_orb --out reports/qa/orb-text-agent --max-spend-usd 5
$CLI eval qa --config configs/experiments/orb-text-qa-agent.yaml --split frozen --frozen \
  --database-url .../copilot_orb --out reports/qa/orb-text-agent --max-spend-usd 10

# Multi-turn answering (MTRAG): live and reference-history modes
$CLI eval mtrag --config configs/experiments/mtrag-govt-agent.yaml --split test --frozen \
  --database-url .../copilot_mtrag_govt --out reports/mtrag --max-spend-usd 40

# Compare two variants of one run
$CLI eval pair --treated RUN.agent/cases.parquet --baseline RUN.flat/cases.parquet

# Re-judge with a second model, and the agreement
$CLI eval qa-rejudge --run RUN.flat --judge openai-gpt-6-sol --out $DATA_DIR/eval/qa/labels/x.jsonl --max-spend-usd 3
```

Outputs go to `--out` (metrics, a manifest with every digest and the GPU sample, and per-case
tables without text) and to `$DATA_DIR/eval/` (the same cases with text). `reports/` is ignored by
git.

Offline, frozen-output checks run in CI and need no services or spend:

```bash
uv run --project backend python -m copilot.cli eval smoke       # retrieval metrics
uv run --project backend python -m copilot.cli eval qa-smoke    # answering, judging, pairing
```

## Experiment files

| File | Compares |
|---|---|
| `orb-text-retrieval.yaml` | Search modes over all ORB questions |
| `orb-text-qa-v2.yaml`, `orb-text-qa-v3-structured.yaml` | The answering pipeline's variants on ORB's frozen 400 |
| `orb-chat-followups.yaml` | Condensing follow-ups, on ORB question pairs |
| `orb-text-qa-agent.yaml`, `mtrag-agent.yaml` | The research agent against the pipeline |
| `mtrag-govt-agent.yaml`, `orb-text-qa-agent-v3.yaml` | The stricter skip rule, and its ORB guard |
| `multihop-plan.yaml` | The planning step |
| `orb-text-qa-agent-check.yaml`, `mtrag-fiqa-agent-check.yaml` | The self-check (registered, not scored) |
