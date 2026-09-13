# Research Expansion and Advanced Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the research product through measured graph, learned-ranking and synthesis experiments.

**Architecture:** Reuse existing corpus, retrieval, recommendation and evidence contracts. New adapters and models must earn promotion through frozen evaluation and operational budgets.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL, Qdrant, Next.js/TypeScript, pytest, Playwright; subsystem additions are specified below.

**Spec:** [Technical specification](../specs/2026-09-05-ml-research-copilot-design.md). Read the relevant sections and global contracts before implementing.

**Status:** Draft for review. No implementation tasks completed.

## Global Constraints

- Python 3.12; Node.js 22; TypeScript strict mode; UTF-8 files; UTC timestamps.
- Lock Python and JavaScript dependencies; pin container images and model revisions before benchmark or deployment runs.
- Hugging Face is an offline corpus/artifact destination and is never queried during user requests.
- Public corpus data and private user documents use separate Qdrant collections and storage namespaces.
- Derive user identity from a verified token; never trust a client-supplied user_id.
- All benchmark runs record corpus, split, model, configuration, code, hardware, and seed versions.
- No paid cloud resources, public publishing, or application implementation occur while preparing these documents.
- Postgres and Qdrant are self-hosted via Docker on the developer's own machine by default; no managed database, vector, or storage subscription is used unless a future, explicitly-recorded decision changes this.
- All persistent local data lives under one configurable root directory, default `C:\ml-copilot-data\` on the developer's machine, so the entire dataset can be deleted by removing that directory.
- The hosted LLM/multimodal generation API is the one paid resource in the project, funded personally by the developer; its cost tracking and daily spend cap from P5.2 remain required because real personal money is involved.

## Entry checkpoint and working conventions

Entry: M1–M5 form a working beta; this plan is optional expansion, not an MVP prerequisite. Read spec §13 and the evaluation protocol. Tasks requiring real labels/users stay pending until that evidence exists. Do not treat future research work as a promised model improvement.

Run commands from the repository root. Paths name future files. Each task contains a concrete regression example and critical implementation logic; finish the stated contracts and acceptance cases in the same task. These snippets are design artifacts, not tested application code. A task is typically several focused work sessions; each checkbox may be split into 2–5 minute edit/run actions while executing. No task is complete merely because its example test passes.

For each task: capture the expected behavioral failure, implement one behavior at a time, run the listed suite, inspect the diff, and record command/exit status, report path and commit in the progress tracker. Commit only that task's files after review. Keep external API tests opt-in and deterministic fixtures in normal CI. Missing dependencies are setup failures, not the intended red test.

## P6.1: Expand coverage through independently verified venue adapters

**Depends on:** G5; capacity measurements from P5.4

**Files:**

- Create `backend/src/copilot/corpus/sources/{acl_anthology,cvf,aaai,kdd}.py`.
- Create `configs/corpus/{nlp,vision,general_ai}.yaml`, `backend/tests/integration/test_expansion.py`.
- Update `docs/data-card.md` and coverage/report generation.

**Interfaces:** Each adapter implements P1.4 `fetch_page` and produces identical normalized records. `coverage_fraction(ingested: int, expected: int|None) -> float|None` reports known coverage; source-specific count/acceptance rules are explicit manifests.

- [ ] Write the regression test below in `backend/tests/integration/test_expansion.py` and add the additional acceptance cases listed for this task.

```python
from copilot.corpus.releases import coverage_fraction

def test_unknown_denominator_is_not_complete():
    assert coverage_fraction(100, None) is None
    assert coverage_fraction(90, 100) == 0.9
    assert coverage_fraction(0, 0) is None
```

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_expansion.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def coverage_fraction(ingested: int, expected: int | None) -> float | None:
    if ingested < 0 or (expected is not None and expected < 0):
        raise ValueError("negative_coverage_count")
    if expected in (None, 0):
        return None
    if ingested > expected:
        raise ValueError("coverage_count_conflict")
    return ingested / expected
```

Expand one venue-year pilot at a time: ACL/EMNLP/NAACL, then CVPR/ICCV/ECCV, then AAAI/KDD. Confirm authoritative main-track membership, year, source URLs and allowed redistribution before ingestion. Reuse normalization/parser/worker/export interfaces without branching core logic per provider. Workshops and Findings remain excluded until an explicit manifest revision enables them with separate labels/counts. Deduplicate across arXiv and existing works. Record 2026 as-of availability; avoid claiming future proceedings completeness. Capacity gate measures bytes/point and rebuild time before multiplying corpus scale. Support a separately labeled curated pre-2022 context subset for foundational queries, with independent coverage and benchmark revision.

- [ ] Verify the additional acceptance cases: adapter captured-response tests; count mismatch detected; duplicate cross-provider work resolves; missing full text doesn't delete metadata; rights filter maintained; venue expansion doesn't change old IDs; index switch atomic; search quality slices compared by venue/year; new corpus never compared directly to old as if only ranker changed.

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_expansion.py -q` again, then replay a 100-paper expansion pilot and run retrieval/evidence smoke plus per-venue evaluation slices. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m6-coverage.md` with manifest revisions, actual counts and capacity decision; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: expand verified publication sources and coverage reporting`.

## P6.2: Add evidence-based comparison and a bounded research graph

**Depends on:** P4.5, P5.2

**Files:**

- Create `backend/src/copilot/agents/{graph,tools,comparison}.py`, `configs/agent.yaml`.
- Create `backend/tests/unit/test_agent_budget.py`, `backend/tests/integration/test_comparison.py`.
- Create `frontend/src/features/comparison/`, comparison page and `frontend/tests/comparison.spec.ts`.

**Interfaces:** `can_call_tool(calls: int, search_rounds: int, elapsed_seconds: float, is_search: bool=False) -> bool`; `compare_papers(principal: Principal, paper_ids: list[UUID], question: str) -> dict`; tools wrap existing search/evidence/metadata/recommendation services, never bypassing their authorization or limits.

- [ ] Write the regression test below in `backend/tests/unit/test_agent_budget.py` and add the additional acceptance cases listed for this task.

```python
from copilot.agents.graph import can_call_tool

def test_orchestration_budget_is_hard():
    assert can_call_tool(5, 1, 29.0)
    assert not can_call_tool(6, 1, 1.0)
    assert not can_call_tool(1, 2, 1.0, is_search=True)
    assert can_call_tool(1, 2, 1.0, is_search=False)
    assert not can_call_tool(1, 1, 30.0)
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_agent_budget.py backend/tests/integration/test_comparison.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def can_call_tool(calls: int, search_rounds: int, elapsed_seconds: float,
                  is_search: bool = False) -> bool:
    return (calls < 6 and elapsed_seconds < 30.0
            and (not is_search or search_rounds < 2))
```

Use LangGraph with explicit state: question, principal, selected paper IDs, evidence, tool_calls, search_rounds, deadline, answer and status. Graph nodes route→plan→tools→collect→generate→validate→end; repair is bounded. The dispatcher passes is_search=True for search tools. After two search rounds, evidence/metadata tools may still run within the six-call/30-second budget; a third search is rejected. Record both counters and test actual dispatch. Comparison accepts 2–5 papers and generates typed rows for method/dataset/metric/result/compute/limitations with evidence per populated cell. Missing source data renders unavailable; inconsistent dataset/metric cells are marked not directly comparable. Add opt-in external web tool with separate provenance and two-search budget; adapter contract `web_search(query: str, limit: int=5) -> list[Evidence]`. It obeys host/network safety and marks source_kind=web via an explicit extension to the Evidence union, not as a paper source. LangGraph supplies the orchestration abstraction; these limits are project requirements. [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview).

- [ ] Verify the additional acceptance cases: tool call/search/deadline cap; prompt-injected tool requests cannot bypass dispatch; every comparison cell cites authorized evidence or says unavailable; unknown paper ID rejected; unlike metrics not silently compared; graph failure returns collected evidence; web disabled by default; no private context in public web query; routing baseline compared on fixed intent labels.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_agent_budget.py backend/tests/integration/test_comparison.py -q` again, then run comparison browser journey and fixed multi-step questions through both direct and graph workflows, comparing answer quality/tool cost. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m6-agent.md` with graph traces, comparison audit and measured value versus direct routing; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add bounded research tools and cited paper comparison`.

## P6.3: Evaluate citation-graph recommendation candidates

**Depends on:** P3.5, P6.1

**Files:**

- Create `backend/src/copilot/recommendations/graph.py`, `configs/experiments/graph-recommendations.yaml`.
- Create `backend/tests/unit/test_graph_candidates.py`, `backend/tests/integration/test_graph_eval.py`.

**Interfaces:** `graph_candidates(seeds: set[str], edges: list[dict], cutoff: datetime) -> set[str]`; edge fields are source,target,observed_at. `GraphCandidateSource.candidates(profile, release_id, cutoff, limit: int=100) -> list[UUID]`; candidates union into the existing recommendation service with versioned scoring.

- [ ] Write the regression test below in `backend/tests/unit/test_graph_candidates.py` and add the additional acceptance cases listed for this task.

```python
from datetime import datetime, timezone
from copilot.recommendations.graph import graph_candidates

def test_graph_uses_available_edges_and_excludes_seed():
    cutoff = datetime(2026, 9, 1, tzinfo=timezone.utc)
    edges = [
        {"source": "a", "target": "b", "observed_at": datetime(2026, 8, 1, tzinfo=timezone.utc)},
        {"source": "a", "target": "c", "observed_at": datetime(2026, 9, 2, tzinfo=timezone.utc)}
    ]
    assert graph_candidates({"a"}, edges, cutoff) == {"b"}
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_graph_candidates.py backend/tests/integration/test_graph_eval.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def graph_candidates(seeds, edges, cutoff):
    candidates = set()
    for edge in edges:
        if edge["observed_at"] >= cutoff:
            continue
        if edge["source"] in seeds:
            candidates.add(edge["target"])
        if edge["target"] in seeds:
            candidates.add(edge["source"])
    return candidates - seeds
```

Begin with references and citing neighbors; add bibliographic coupling/shared references as a separately versioned feature. Cap graph candidate addition at 100, preserving source-attribution counts. Require paper availability before cutoff in addition to edge availability. Normalize graph affinity by degree and test popularity bias; do not just count citations twice. Ablate graph-only, embedding-only, profile+graph and existing heuristic. Optional external Semantic Scholar recommendation comparison is clearly a service baseline with its own corpus and availability, not a controlled internal ablation. If historical graph snapshots don't exist, evaluate prospective data only and report the limitation.

- [ ] Verify the additional acceptance cases: future edge/paper leakage; dangling edge IDs; self loops; seed exclusion; duplicate edges; high-degree paper dominance; stale graph fallback; unavailable external provider doesn't block internal benchmark; output reasons cite actual relationship rather than semantic guess.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_graph_candidates.py backend/tests/integration/test_graph_eval.py -q` again, then run recommendation graph ablations on the unchanged preference split plus prospective temporal windows. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/recommendations/graph/report.md` with relevance/diversity/coverage and degree-bias analysis; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: evaluate citation-graph recommendation candidates`.

## P6.4: Train and compare a learned ranker only after label readiness

**Depends on:** P2.5, P3.5; data readiness audit

**Files:**

- Create `backend/src/copilot/evaluation/{training_data,train_ranker}.py`, `backend/src/copilot/search/learned_rank.py`.
- Create `configs/experiments/learned-rank.yaml`, `backend/tests/unit/test_rank_groups.py`, `backend/tests/integration/test_ranker_artifact.py`.

**Interfaces:** `group_sizes(query_ids: list[str]) -> list[int]` requires contiguous groups; `train_ranker(config: Path, out: Path) -> dict`; ranker artifact contains feature schema, model parameters, training manifest and held-out metrics. `LearnedRanker.score_features(rows: list[dict]) -> list[float]` rejects missing/incompatible feature schema.

- [ ] Write the regression test below in `backend/tests/unit/test_rank_groups.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.evaluation.training_data import group_sizes

def test_query_groups_are_not_mixed():
    assert group_sizes(["a", "a", "b", "b", "b"]) == [2, 3]
    with pytest.raises(ValueError, match="noncontiguous_query"):
        group_sizes(["a", "b", "a"])
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_rank_groups.py backend/tests/integration/test_ranker_artifact.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
from itertools import groupby

def group_sizes(query_ids: list[str]) -> list[int]:
    seen = set()
    sizes = []
    for query_id, rows in groupby(query_ids):
        if query_id in seen:
            raise ValueError("noncontiguous_query")
        seen.add(query_id)
        sizes.append(sum(1 for _ in rows))
    return sizes
```

Readiness audit requires the spec's 200 development query families/2,000 human-audited labels and separate test set; if absent, complete the audit report and leave training pending. Do not fabricate training data. Start LightGBM LGBMRanker objective=lambdarank, n_estimators=100, num_leaves=15, learning_rate=0.05, random_state=42; features: lexical, dense, reranker, freshness, abstract_missing, token_length. Sort training rows by query then paper, pass `group=group_sizes(query_ids)`; validation uses its own eval_group. Group counts must sum to row count. [LightGBM ranker API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html). Fit preprocessing only on training. Tune a small development/validation grid (leaves 7/15, estimators 50/100), compare to RRF/reranker on frozen test once. Add ranker-specific feature attribution and learning curves. Optional reranker fine-tuning is a subsequent bounded experiment: training-only hard negatives, labeled synthetic augmentation, fixed seeds, adapter artifact, unchanged human test set. No main chat-model fine-tuning.

- [ ] Verify the additional acceptance cases: query family/work-version split leakage rejected; group sizes align with rows; absent features fail closed; shuffled feature order normalized by schema; no test labels in fit; artifact round-trip preserves scores; CPU inference measured; temporal recommendation features available before cutoff; rollback retains existing reranker if learned ranker underperforms.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_rank_groups.py backend/tests/integration/test_ranker_artifact.py -q` again, then run fixture training round-trip and, only when data readiness is met, `uv run --project backend python -m copilot.cli eval train-ranker --config configs/experiments/learned-rank.yaml --out reports/learned-rank`. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/learned-rank/readiness.md`, then model card, training manifest, learning curves and promote/retain decision when trained; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add reproducible learned-ranking experiment pipeline`.

## P6.5: Present research landscapes and experiment evidence

**Depends on:** P6.2; P6.3/P6.4 reports as available

**Files:**

- Create `backend/src/copilot/agents/landscape.py`, `backend/tests/unit/test_landscape_provenance.py`.
- Create `frontend/src/features/landscape/`, `frontend/src/app/evaluation/page.tsx`, `frontend/tests/landscape.spec.ts`.
- Update `docs/portfolio-case-study.md` and experiment report schema.

**Interfaces:** `validate_landscape(sections: list[dict]) -> list[str]` identifies unsupported factual entries; sections include approaches, representative papers, limitations and suggested_directions. Dashboard reads aggregate versioned reports only, never private interactions or unreviewed raw logs.

- [ ] Write the regression test below in `backend/tests/unit/test_landscape_provenance.py` and add the additional acceptance cases listed for this task.

```python
from copilot.agents.landscape import validate_landscape

def test_factual_landscape_entries_need_evidence():
    sections = [
        {"kind": "approach", "text": "A method exists", "evidence_ids": []},
        {"kind": "suggested_direction", "text": "A hypothesis to explore", "evidence_ids": []},
    ]
    assert validate_landscape(sections) == ["0:missing_evidence"]
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_landscape_provenance.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def validate_landscape(sections: list[dict]) -> list[str]:
    errors = []
    for index, section in enumerate(sections):
        if section["kind"] != "suggested_direction" and not section["evidence_ids"]:
            errors.append(f"{index}:missing_evidence")
    return errors
```

Generate a bounded landscape from the existing agent/evidence pipeline, with labeled hypotheses and linked source claims. Apply full P4 citation validation after the structural helper. Display coverage limitations and the actual corpus date; don't promise exhaustive related work or validated novel research gaps. Dashboard compares only matching dataset/split/corpus runs by default, with explicit warnings for incompatible reports. Present retrieval quality versus p95 latency, recommendation quality/diversity, RAG citation/support and experiment history. Include failed hypotheses and non-promoted models. Final portfolio report can cite external pilot observations only after they exist, separately from personal data. Collaborative filtering remains a future conditional project requiring multiple real users and its own approved data/evaluation design.

- [ ] Verify the additional acceptance cases: unsupported factual entries blocked; speculative directions labeled; no invented citations; report comparison refuses incompatible manifests; hidden private metrics remain private; graph budget applies; keyboard navigation; evaluation charts show denominators and uncertainty, not just a winning number.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_landscape_provenance.py -q` again, then run landscape browser journey, report compatibility tests and the release smoke suite. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m6-showcase.md`, updated case study and an explicit list of evaluated versus deferred experiments; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: present cited research landscapes and experiment evidence`.

## P6.6: Compare question-answering architectures against the single-shot baseline

**Depends on:** P4.5 (the baseline and its judged dataset); P6.2 for the agentic arm

**Recorded intent, 2026-09-14.** The answering path specified in spec section 9 is
single-shot retrieve-then-read: one retrieval, one generation, then citation
validation. Its retrieval is strong (hybrid, reranked, papers-then-chunks) and
its output is provenance-checked, but the control flow itself is the vanilla RAG
skeleton. There is no query decomposition, no re-retrieval, and no reflection on
whether the retrieved evidence was adequate. The owner wants the alternatives
measured rather than assumed.

**Arms to compare**, all on the same frozen corpus release and the same 60-case
RAG dataset from spec section 11:

1. **Baseline** - the P4.1-P4.2 single-shot path, unchanged. The control.
2. **Knowledge / graph RAG** - build a retrievable graph over corpus content
   (entities, methods, datasets, metrics and their relations) and traverse it for
   multi-paper questions, instead of retrieving flat chunks. Distinct from P6.3,
   which uses the citation graph as a *recommendation* candidate source and does
   not touch the answering path.
3. **Agentic RAG** - iterative retrieval under P6.2's existing budget (six tool
   calls, two search rounds, 30-second deadline, no open-ended recursion), so the
   model may re-retrieve when the first evidence set is inadequate.

**Files:** create `backend/src/copilot/evidence/{graph_retrieve,agentic_retrieve}.py`,
`configs/qa_architectures.yaml`, `backend/tests/unit/test_qa_arms.py`,
`backend/tests/integration/test_qa_comparison.py`, `reports/m6-qa-architectures.md`.

**What decides it.** The 10 multi-paper and 10 unanswerable cases in the RAG
dataset are the discriminating subsets: graph retrieval should help the former,
and an agentic arm must not degrade the latter by retrieving until it finds
something to say. Report supported-claim rate, provenance validity, judged
answer quality, p50/p95 latency and per-question cost for every arm, with
paired bootstrap intervals over the 1,000 resamples spec section 11 requires.

**Promotion rule.** An arm replaces the baseline only on measured benefit that
survives the latency and cost gates; a negative result is a valid, publishable
outcome. Graph construction cost and its storage are counted against the arm,
not treated as free infrastructure. No arm may weaken the citation validation in
spec section 9 - every arm answers through the same validator.

## Exit checkpoint

G6 completes only for experiments actually run and documented. A retain-baseline decision is a valid completed experiment; absent training data is a pending experiment. Update corpus/model cards, reports and portfolio claims to match observed results.
