# Answering runs

Each directory here is one answering experiment (`configs/experiments/*-qa.yaml`) run over
one split. Everything is written by `eval qa` (`backend/src/copilot/evaluation/qa.py`).

```text
reports/qa/<experiment>/
  frozen-scorings.json                 every scoring of the frozen split, per configuration
  <split>/<run_id>.<variant>/
    manifest.json                      what produced the run: experiment, dataset and corpus
                                       digests, search and evidence settings, generator and
                                       judge identities with prompt digests, budget, code,
                                       hardware, the GPU before the run, where the text went
    metrics.json                       accuracy (Wilson and family bootstrap), cited MAP,
                                       provenance validity, abstention rate, repairs, judge
                                       errors, failures, per-type facets, cost
    cases.parquet                      one row per case: ids, top papers, evidence and cited
                                       papers, cited MAP, provenance, verdict, costs, seconds
```

**No question, reference answer, generated answer or judge reason is under `reports/`.**
Open RAG Bench is CC-BY-NC-4.0: that text is written per case to
`${DATA_DIR}/eval/qa/<run_id>/<variant>/cases.jsonl`, which `manifest.json` names.

## Producing a run

```bash
uv run --env-file .env --project backend python -m copilot.cli eval qa \
  --config configs/experiments/orb-text-qa.yaml --split development --limit 20 --variant flat \
  --database-url postgresql+psycopg://...copilot_orb --out reports/qa/orb-text --max-spend-usd 0.50
uv run --env-file .env --project backend python -m copilot.cli eval qa \
  --config configs/experiments/orb-text-qa.yaml --split frozen --frozen \
  --database-url ...copilot_orb --out reports/qa/orb-text --max-spend-usd 3.00
uv run --project backend python -m copilot.cli eval pair \
  --treated reports/qa/orb-text/frozen/<run>.section/cases.parquet \
  --baseline reports/qa/orb-text/frozen/<run>.flat/cases.parquet
uv run --project backend python -m copilot.cli eval calibration \
  --labels <blind labels>.jsonl --cases reports/qa/orb-text/development/<run>.flat/cases.parquet
uv run --project backend python -m copilot.cli eval qa-smoke        # offline; --update to re-freeze
```

## Rules the runner enforces

- **A case is one pass:** the shipped search's top papers, evidence inside them
  (`configs/evidence.yaml`, plus the variant's overrides), the generator's claims with
  evidence ids validated structurally with one repair, then the judge. An abstention
  (no claims, `insufficient_evidence`) is scored wrong without a judge call: every ORB
  question has an answer. A refused or timed-out generation is a typed failure, kept in
  every denominator. A judge error (two malformed verdicts) is excluded from the accuracy
  denominator and counted.
- **Accuracy** is judged-correct over judged, with a Wilson interval and a bootstrap over
  families (a family is a gold paper). **Cited MAP** is 1/rank of the gold paper among the
  papers the answer cites, 0 when it is not cited. **Provenance validity** is the share of
  cases whose every evidence id passed with no repair.
- **The budget is the run's.** The worst case (two generation and two judge calls per case
  at the models' output caps) is refused before anything loads; the ledger's spend for the
  run stops it mid-way, keeping finished cases, and `manifest.json` says so (`stopped`).
  The ledger of record is the main database's, so one daily cap covers every run.
- **Development tunes, frozen is scored once.** `--split frozen` needs `--frozen`, and
  `frozen-scorings.json` records every scoring per configuration. The development split
  is the only one read while prompts settle.
- **The judge is the same provider as the generator** (`gpt-6-luna`, effort `low` against
  `none`): a correlated-bias limitation, stated in every report and measured by the
  calibration against 50 blind hand labels (`eval calibration`, Cohen's κ, floor 0.6).
- **Two runs are compared on identical cases** with exact McNemar on verdicts and a sign
  test on cited MAP (`eval pair`); the pre-registered rule is in the experiment file.
