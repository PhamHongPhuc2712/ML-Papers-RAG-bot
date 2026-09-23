# Evaluation Benchmarks Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Implement only the
> task asked for; do not pull later-task infrastructure forward without a demonstrated need.

**Goal:** Measure retrieval with the LitSearch benchmark and measure generation with a
reference-free LLM judge, so both halves of the system have numbers that a reader can
reproduce and disbelieve.

**Architecture:** Two independent harnesses over one shared run manifest. The retrieval
harness is deterministic and free — it replays frozen queries against frozen indexes and
computes rank metrics. The generation harness costs money — it calls a pinned judge model
over a frozen question set and writes per-item verdicts, never only aggregates.

**Tech Stack:** Python 3.12, pyarrow, `huggingface_hub`, the `anthropic` SDK, pytest.

**Spec:** [Technical specification](../specs/2026-09-05-ml-research-copilot-design.md) §11
(evaluation and progress gates), §7 (retrieval), §9 (evidence and chat), §12 (cost).

**Status:** Draft for review. No implementation tasks completed.

---

## What this plan changes about §11

Spec §11 assumed every label would be produced by hand. Two of its instructions are
superseded here; everything else in §11 still binds.

| §11 as written | This plan | Why |
|---|---|---|
| Expand to 150 hand-judged query families | Use LitSearch's 597 queries as the primary retrieval set; keep a 30-query hand-authored in-domain pilot | 597 expert-checked queries already exist. Hand-labeling 150 is weeks of work for a smaller, less comparable set. |
| RAG cases record reference claims and a gold answer | Reference-free judging: keep the questions, discard the answers | Gold answers rot the moment the corpus or prompt changes, and writing them is the most expensive part. Faithfulness is judged against retrieved chunks, which are the thing we actually care about. |

Both need a decision row in `docs/superpowers/progress.md` before E2 and E5 start.

Three things §11 requires that reference-free evaluation does **not** excuse:

- **Human calibration is still mandatory.** G4 passes on human-audited supported-claim rate,
  not on the judge's own number. A judge nobody checked is an unvalidated instrument.
- **Judged coverage must be reported.** LitSearch labels only the cited paper. Unlabeled
  results are unjudged, not irrelevant, and the report says so every time.
- **The test split stays locked.** It is scored, never tuned against.

---

## Measured facts this plan is built on

Taken 2026-09-21 against `princeton-nlp/LitSearch` at revision
`9573fb284a1026c998df47024b888a163f0f0e25`, and against `copilot_v2` while the corpus
rebuild was 47% complete (39,957 of 85,732 papers).

| Fact | Value |
|---|---|
| Queries | 597 — `inline_nonacl` 253, `manual_acl` 155, `inline_acl` 98, `manual_iclr` 91 |
| Gold papers per query | 1 for 563 queries; 2 for 29; 3 for 3; 4 for 1; 5 for 1 |
| Annotations | `specificity` 0 for 155 / 1 for 442; `quality` 1 for 294 / 2 for 303 |
| Corpus | 64,183 documents; `corpus_clean` carries `corpusid`, `title`, `abstract`, `citations`, `full_paper` |
| Full text present | 10,696 of 10,698 rows in shard 0 (99.98%) |
| Corpus-wide title overlap with ours | 1,284 of 10,698 shard-0 titles (12.0%) |
| **Gold-paper coverage in our corpus** | **102 of the 150 queries whose gold falls in shard 0 (68%)** |

The last row is the one that makes this plan viable. Corpus-wide overlap is low because
LitSearch's corpus is mostly the older work that recent papers cite, which our 2023–2026
window excludes. But the *gold* papers are the recent ML/NLP papers the queries are asking
for, and those are largely inside our window. A 68% in-domain slice is a real benchmark;
12% would not have been.

Both figures are samples — one of six shards, against a corpus that was 47% built. E1
re-measures both on the finished corpus and neither number is quoted anywhere until it does.

**License is unresolved.** The dataset repo declares no license tag. Until E1 establishes
the terms, LitSearch data is local-use-only: it never enters a corpus export, and
`configs/artifacts.yaml` marks every LitSearch-derived artifact non-redistributable.

---

## Entry checkpoint

Entry: G1 evidence exists; the `paragraph-pack-v1` rebuild has finished and its coverage
report is written. E3 additionally needs P2.2 (vector indexing with release switching);
E4 needs P2.3 (fusion and reranking). E1, E2 and E5 can start immediately.

E6 needs something that generates answers — P4.2 produces them and P4.1 attaches the
citations. **This is our own generator, not reference answers.** Reference-free judging
removes the hand-written ideal answer; it still judges the answer our system produced, at
run time, against the chunks our system retrieved. Both sides come from the pipeline, so the
pipeline has to exist first. A minimal generator — retrieve top-k chunks, one call that
answers and cites chunk IDs — is enough to start E6 against, and can be replaced by P4.2
later without changing the judge.

Run commands from the repository root. Every run records the manifest in §"Run manifest"
below; a run without one is not evidence.

---

## E1: Pin LitSearch and measure what it can actually tell us

**Depends on:** the corpus rebuild completing.

**Files:**

- Create `backend/src/copilot/evaluation/__init__.py`, `backend/src/copilot/evaluation/litsearch.py`.
- Create `backend/tests/unit/test_litsearch_fetch.py`, `backend/tests/fixtures/litsearch/mini.parquet`.
- Create `configs/evaluation.yaml`, `reports/e1-litsearch-coverage.md`.
- Modify `CLAUDE.md` (the "Hugging Face is only contacted by `corpus mirror`" constraint).

**Interfaces:**

```python
LITSEARCH_REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"

def fetch_litsearch(dest: Path, revision: str = LITSEARCH_REVISION) -> LitSearchPaths: ...
def load_queries(paths: LitSearchPaths) -> list[LitSearchQuery]: ...
def load_corpus(paths: LitSearchPaths, with_full_text: bool = False) -> Iterator[LitSearchDoc]: ...
def match_to_corpus(docs: Iterable[LitSearchDoc], session: Session) -> MatchReport: ...
```

`MatchReport` carries `matched_by_strong_id`, `matched_by_title`, `unmatched`, and
`gold_coverage` — the fraction of the 597 queries with at least one gold paper in our corpus.

- [ ] Write the regression test below and confirm it fails for the stated reason.

```python
def test_fetch_pins_the_revision(tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(litsearch, "hf_fetch", lambda *a, **kw: seen.append(kw["revision"]) or tmp_path)
    litsearch.fetch_litsearch(tmp_path)
    assert seen and set(seen) == {litsearch.LITSEARCH_REVISION}


def test_unpinned_fetch_is_refused(tmp_path):
    with pytest.raises(litsearch.UnpinnedRevisionError):
        litsearch.fetch_litsearch(tmp_path, revision="main")
```

- [ ] Download the three configurations into `${DATA_DIR}/benchmarks/litsearch/` — never into
      the repository; `corpus_clean` carries full text and is gigabytes. Record each file's
      sha256 in `configs/evaluation.yaml` alongside the revision.
- [ ] Reuse `copilot.corpus.mirror.hf_fetch` rather than writing a second Hugging Face client,
      and amend the CLAUDE.md constraint to name both callers. The constraint that matters —
      Hugging Face is never contacted during a user request — is unchanged.
- [ ] Resolve the license. Check the dataset card, the LitSearch paper and the project's
      GitHub repository. Record the finding verbatim in `reports/e1-litsearch-coverage.md`.
      If it stays unresolved, that is the finding, and local-use-only stands.
- [ ] Confirm `corpus_s2orc` exposes `externalids`. If it does, match on DOI, arXiv ID and ACL
      Anthology ID against `paper_identifiers` **first**, falling back to normalized title only
      for rows with no strong ID. Strong-ID matching is what makes the coverage number
      trustworthy; title normalization alone silently misses papers whose titles were
      reformatted between preprint and proceedings.
- [ ] Re-measure both overlap figures against the finished corpus and write
      `reports/e1-litsearch-coverage.md`: corpus-wide overlap, gold coverage overall and per
      `query_set`, the matching method's false-positive rate on a hand-checked sample of 50,
      and the count of queries whose golds are *partially* present.
- [ ] Record the documented semantics of `specificity` and `quality` from the LitSearch paper.
      If either is undocumented, say so and stratify on `query_set` alone.

**Acceptance cases:** an unpinned revision raises; a checksum mismatch raises rather than
silently using a changed file; `match_to_corpus` over the bundled mini fixture returns exact
counts; a query whose golds are split between matched and unmatched is reported as partial,
never as covered.

**Suites:** `uv run --project backend pytest backend/tests/unit/test_litsearch_fetch.py -q`,
then Ruff and `mypy --config-file backend/pyproject.toml backend/src`.

**Commit:** `feat: pin the LitSearch benchmark and measure its overlap with the corpus`

---

## E2: Convert LitSearch into the evaluation dataset and freeze splits

**Depends on:** E1.

**Files:**

- Create `backend/src/copilot/evaluation/datasets.py`, `backend/src/copilot/evaluation/metrics.py`.
- Create `backend/tests/unit/test_metrics.py`, `backend/tests/unit/test_datasets.py`.
- Create `data/fixtures/retrieval/{queries,qrels,splits}.jsonl`, `docs/labeling-guide.md`.

These are the files P2.1 owns. This task implements P2.1's stated interfaces
(`ndcg_at_k`, `recall_at_k`, `mrr_at_k`, split records carrying `query_id`, `family_id`,
`workflow`, `split`) and only replaces where the labels come from. If P2.1 has already run,
this task rewrites the fixtures and leaves the interfaces untouched.

- [ ] Write the regression test below and confirm it fails.

```python
def test_binary_grades_still_rank_correctly():
    grades = {"gold": 1}
    assert metrics.ndcg_at_k(["gold", "x"], grades, 10) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["x", "gold"], grades, 10) == pytest.approx(0.6309, abs=1e-4)
    assert metrics.ndcg_at_k(["x", "y"], grades, 10) == 0.0


def test_judged_coverage_is_reported_not_assumed():
    result = metrics.evaluate(ranked=["gold", "unknown"], grades={"gold": 1}, k=10)
    assert result.recall_at_k == 1.0
    assert result.judged_coverage == pytest.approx(0.5)  # "unknown" is unjudged, not irrelevant
```

- [ ] Map each LitSearch query to a split with seed 42, stratified by `query_set` and
      `specificity`: 60% development (≈358), 20% validation (≈119), 20% test (≈120). Write the
      assignment to `splits.jsonl` and never recompute it — a split that moves is not a split.
- [ ] Set `family_id` to the query's own ID. LitSearch has no rewrite families; inventing them
      would fake a grouping the data does not have. Say this in `docs/labeling-guide.md`.
- [ ] Carry `specificity`, `quality` and `query_set` through to the records so results can be
      sliced by them without re-reading the source.
- [ ] Make `judged_coverage` a required field of every metric result, not an optional extra.
      A caller cannot report Recall@50 from this module without also holding the coverage it
      was computed at.
- [ ] Document in `docs/labeling-guide.md` that LitSearch grades are binary, that nDCG over
      binary gains is reported for §11 comparability but carries less information than the
      graded 0–3 scale §11 describes, and that the hand-authored pilot set is where graded
      judgments live.

**Acceptance cases:** splits are disjoint and stable across two runs; a query with 5 golds
computes recall against all 5; `ndcg_at_k` deduplicates repeated IDs; an empty ranking scores
0.0 rather than raising.

**Suites:** `uv run --project backend pytest backend/tests/unit/test_metrics.py backend/tests/unit/test_datasets.py -q`.

**Commit:** `feat: build the retrieval evaluation set from LitSearch with frozen splits`

---

## E3: Run the baselines on LitSearch's own corpus

**Depends on:** E2, P2.2.

This is the number that is comparable to published work: same corpus, same queries, our
retrieval stack. It measures the *stack*, not our corpus.

**Files:**

- Create `backend/src/copilot/evaluation/runner.py`, `backend/src/copilot/cli/eval.py`.
- Create `backend/tests/integration/test_eval_runner.py`, `reports/e3-litsearch-baselines.md`.

**Interfaces:** `run_retrieval_eval(config: EvalConfig) -> EvalRun`; CLI
`eval retrieval --benchmark litsearch --corpus litsearch --split dev --baseline bm25`.

- [ ] Index `corpus_clean` as its own release (`litsearch-v1`) with its own Qdrant collection
      pair, using the existing release machinery. It is a separate corpus and must never share
      a collection with the paper corpus.
- [ ] Index **title + abstract only** for the headline number — that is the shape the corpus is
      packaged in. If a full-text configuration is also run, it is reported as a separate,
      labeled row and never merged into the headline.
- [ ] Run all four baselines — BM25, dense, hybrid RRF, hybrid + rerank — on the same frozen
      snapshot, as G2 requires.
- [ ] Report Recall@50, MRR@10, nDCG@10, judged coverage and p50/p95 latency per baseline,
      sliced by `query_set` and `specificity`.
- [ ] Report query-level failures explicitly. A query that errored is not a query that scored
      zero, and it does not silently leave the denominator.
- [ ] Bootstrap the confidence intervals with 1,000 resamples over queries, per §11.

**Acceptance cases:** a run against the mini fixture produces a manifest with every required
field; rerunning with the same seed and revisions reproduces the metrics exactly; a
deliberately broken baseline surfaces as a counted failure, not as a zero.

**Suites:** the integration test with the `test` compose profile up; then the dev-split run.

**Commit:** `feat: run the four retrieval baselines against LitSearch`

---

## E4: Run the in-domain slice against our corpus

**Depends on:** E3, P2.3.

**Files:** modify `runner.py`; create `reports/e4-indomain-retrieval.md`.

- [ ] Restrict to queries whose gold papers are in our corpus, per E1's strong-ID matching.
      Report the slice size and the excluded count on every table.
- [ ] Run the same four baselines over our corpus, both abstract-level and chunk-level.
- [ ] Report judged coverage prominently. Our corpus holds many papers that would answer these
      queries and that LitSearch never labeled; recall here is recall against known golds and
      the report must not imply otherwise.
- [ ] Add the 30 hand-authored in-domain queries with graded 0–3 judgments, covering topic
      discovery, related work and keeping up — the workflows LitSearch's citation-derived
      queries do not exercise. This is the only hand-labeling this plan asks for.
- [ ] Compare the two slices and write down where they disagree. A stack that wins on
      LitSearch's corpus and loses on ours is telling you something about the corpus.

**Acceptance cases:** slice membership is derived from E1's report, not recomputed ad hoc; a
query whose golds are partially present is excluded from recall and counted separately.

**Commit:** `feat: evaluate retrieval in-domain on the paper corpus`

---

## E5: Build the reference-free question set

**Depends on:** the corpus rebuild. No model calls.

**Files:**

- Create `backend/src/copilot/evaluation/questions.py`, `data/fixtures/rag/questions.jsonl`,
  `backend/tests/unit/test_questions.py`, `docs/rag-question-guide.md`.

Target 60 questions in three groups, grouped by family into 36 development / 12 validation /
12 test, as §11 specifies:

| Group | Count | Source |
|---|---|---|
| Answerable, single paper | 30 | Generated from a sampled chunk, then hand-edited; the answer is discarded |
| Answerable, multiple papers | 10 | Written against 2–4 papers on one topic |
| Unanswerable near-miss | 10 | Plausible, corpus-shaped, and not answerable |
| Real user queries | 10 | **Deferred — there are no logs yet.** Recorded as a gap, backfilled after the pilot |

- [ ] Write the generator: sample a chunk, ask for a question that chunk answers, keep the
      question and the source chunk ID, discard the answer. The chunk ID is provenance for
      later inspection, never an input to the judge.
- [ ] Hand-edit every generated question. Generated questions leak the source's phrasing and
      turn retrieval into string matching; this step is not optional.
- [ ] Build the unanswerable slice deliberately, with the reason recorded per item:
      out-of-window (a 2019 paper's result), out-of-scope (a venue we exclude), and
      plausible-but-nonexistent (a method that reads real and is not). Near-misses only — a
      question about cooking tests nothing.
- [ ] Assign splits by family with seed 42 and freeze them.
- [ ] Record, for each item: `question_id`, `family_id`, `group`, `split`, `answerable`,
      `source_chunk_ids`, `unanswerable_reason`, `author`, `created_at`.

**Acceptance cases:** no question text appears in two splits; every unanswerable item carries
a reason; `source_chunk_ids` resolve to live chunks.

**Commit:** `feat: assemble the reference-free RAG question set`

---

## E6: The judge harness

**Depends on:** E5, and a generator that answers with chunk citations (P4.2 + P4.1, or the minimal stand-in described in the entry checkpoint). **This task spends money.**

**Files:**

- Create `backend/src/copilot/evaluation/judge.py`, `backend/src/copilot/evaluation/schemas.py`.
- Create `prompts/judge/{claims,classify,relevance,precision,citation}-v1.md`.
- Create `backend/tests/unit/test_judge.py`, `backend/tests/integration/test_judge_live.py`.

### Model and cost

Judge model `claude-opus-5`, pinned in `configs/evaluation.yaml` with the prompt version
hashes. §11 asks for a judge separate from the generator; the project funds one provider, so
the separation is not available and **the correlated-bias limitation is stated in every
report**, mitigated two ways: the human calibration slice in E7, and a judge-swap run with
`claude-sonnet-5` over the same calibration slice to measure how much the verdicts move.

Estimated cost per 60-case run, from the call shapes below: ≈19.5K input and ≈6K output tokens
per case → ≈$0.25/case → **≈$15 per run, ≈$7.50 through the Batch API**, plus ≈$2.50 to
generate the answers. These are arithmetic from list prices ($5/$25 per MTok), not
measurements. The first run replaces them with recorded `usage`, and no cost figure is quoted
in a report until it does.

- [ ] Use the **Batch API** — this is offline evaluation with no latency requirement, and it
      halves the bill.
- [ ] Put the rubric and scale anchors in the system prompt behind a `cache_control`
      breakpoint, with the per-item content after it. Verify `cache_read_input_tokens` is
      non-zero across items; if it is zero, something in the prefix is varying.
- [ ] Enforce the P5.2 daily spend cap before the first call, not after the run. A harness that
      discovers the cap by exceeding it has not implemented the cap.

### The five metrics

- [ ] **Faithfulness**, two calls. First extract atomic claims from the answer. Then classify
      each claim against the retrieved chunks as `supported`, `unsupported` or `contradicted`.
      Classify all of one answer's claims in a single call so the chunks are sent once.
      **Report supported-claim rate and contradiction rate as two separate numbers** — an answer
      that invents a citation and one that misreads a result are different failures.
- [ ] **Answer relevance**, anchored 0–3 against the question: `0` does not address it; `1`
      addresses the topic but not the question; `2` answers it partially or with padding; `3`
      answers exactly what was asked. Anchors live in the prompt file, not in code.
- [ ] **Context precision**, per retrieved chunk, judged **against the question, never against
      the answer**. Judging against the answer rewards a generator for ignoring good chunks.
      Report precision@k, and position-weighted precision when ranking order matters.
- [ ] **Abstention correctness**, binary both ways: did it decline on the unanswerable slice,
      and did it wrongly decline on an answerable one. Cheap, and the one metric that catches a
      system that has learned to say nothing.
- [ ] **Citation accuracy**: for each cited chunk, does it support the sentence it is attached
      to. Pairs with P4.1's structural validation — that check proves the ID exists, this one
      proves it is the right ID.

### Output contract

- [ ] Every judge call returns structured JSON via `output_config.format`, with **`reason`
      before `verdict` in the schema**. The ordering is the point: the reason is written first
      and the verdict follows from it, which both improves accuracy and makes a wrong verdict
      legible.
- [ ] Log per item: claims, per-claim verdicts and reasons, chunk IDs, per-chunk precision
      verdicts, token usage, latency, prompt version hash. Write one JSONL row per item to
      `${DATA_DIR}/eval/runs/<run_id>/items.jsonl`. **Aggregates alone are not an acceptable
      output** — when a number moves, the failing items are what you open.
- [ ] A malformed judge response is retried once, then recorded as `judge_error` and excluded
      from the denominator with a count. It is never coerced into a verdict.

**Acceptance cases:** a recorded-fixture run produces byte-identical aggregates on replay; a
claim contradicted by a chunk is classified `contradicted`, not `unsupported`; a chunk
relevant to the answer but not the question scores 0 on context precision; abstention on an
answerable question is counted as a wrongful refusal.

**Suites:** unit tests against recorded fixtures in normal CI; the live test opt-in and
excluded from the default run.

**Commit:** `feat: add the reference-free LLM judge harness`

---

## E7: Calibrate the judge, then report

**Depends on:** E6. G4 depends on this task, not on E6.

**Files:** create `backend/src/copilot/evaluation/calibration.py`,
`reports/e7-judge-calibration.md`, `docs/judge-calibration-guide.md`.

- [ ] Hand-label 50 answers from the development and validation splits with the same rubric the
      judge uses — §11 says start at 30 and grow to 50–100 before stronger claims; 50 is the
      floor for a G4 claim.
- [ ] Label blind. Labeling with the judge's verdict visible measures agreement with your own
      anchoring.
- [ ] Report Cohen's κ per metric, plus raw agreement, plus the confusion matrix for the
      three-way claim classification.
- [ ] **Read every disagreement.** Systematic disagreement means the prompt is wrong, and the
      fix is the prompt, not the labels.
- [ ] Run the `claude-sonnet-5` judge swap over the same 50 and report how far the aggregate
      moves. A metric that changes materially with the judge model is not reportable as a
      system property.
- [ ] Gate the claim: below κ = 0.6 on faithfulness, the judge's number is reported as
      *uncalibrated* and G4 does not pass on it.

**Commit:** `docs: calibrate the judge against human labels`

---

## Run manifest

Every run of E3, E4 or E6 writes this, per §11. A result without one is not evidence.

```yaml
run_id: ...
git_sha: ...
corpus_release: ...            # release ID and checksum
benchmark: litsearch
benchmark_revision: 9573fb284a1026c998df47024b888a163f0f0e25
split: dev | val | test
parser_version: ...
chunker_version: paragraph-pack-v1
embedding_model: ...           # ID and revision
reranker_model: ...
generation_model: ...
judge_model: claude-opus-5
prompt_versions: {claims: <sha256>, classify: <sha256>, ...}
seed: 42
parameters: {k: 10, ...}
hardware: WSL2, 12 cores, 23 GB, RTX 3080 16 GB
timing_method: ...
cost: {input_tokens: ..., output_tokens: ..., usd: ...}
failures: {query_errors: ..., judge_errors: ...}
```

Outputs: metrics JSON, per-query and per-item Parquet, and a Markdown report under `reports/`.

---

## Exit checkpoint

- **G2 Retrieval** passes on E3 and E4: four baselines, one frozen snapshot, Recall@50, MRR@10,
  nDCG@10, judged coverage, p50/p95, bootstrap intervals, and a recorded promotion decision for
  hybrid and reranking.
- **G4 Evidence** passes on E6 and E7: zero fabricated IDs, ≥95% provenance validity, ≥90%
  supported claims on the **human-audited** slice, and a calibrated judge. Below the
  calibration floor, release behavior narrows rather than the gate loosening.
- CI keeps the deterministic smoke set: fixture corpus, recorded judge responses, no network.
  Fail on Recall@10 or nDCG@10 dropping more than 0.03 absolute, or any citation provenance
  failure.

---

## Risks

**Coverage is measured on a partial sample.** The 68% gold coverage comes from one of six
shards against a 47%-built corpus. If the finished number lands below ~40%, the in-domain
slice is too small to carry G2 and E3's own-corpus numbers become the headline, with E4
demoted to a secondary signal. E1 decides this, and it decides it before E4 is built.

**LitSearch queries are not our users' queries.** They are citation-derived questions about
2016–2023 work. They test whether the stack can find a known paper from a description. They do
not test "what came out this month that I should read" — which is the workflow this project
exists for. That is exactly why the 30 hand-authored queries survive as a required task.

**The judge shares a family with the generator.** Stated in every report, measured two ways in
E7, and not solvable within the budget. If a second provider is ever funded, rerun E7 first.

**Reference-free does not mean free of human work.** It removes gold answers, not judgment. If
the calibration slice in E7 gets skipped, the entire generation half of this plan produces
numbers with nothing behind them.
