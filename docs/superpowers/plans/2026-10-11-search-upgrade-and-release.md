# Search Upgrade and Portfolio Release (M5, revised): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, on our own corpus and by a rule fixed in advance, whether the search page's
reranker should read the passage that found each paper instead of its abstract; then make the
local product dependable and presentable: request logs and cost reports, account deletion,
backup and restore of user data, recorded failure drills, and a case study.

**Architecture:** S1 builds a section-level query set from our corpus the way Open RAG Bench
built its own (one question per sampled passage, written by a model, checked mechanically),
stored like LitSearch's: ids and labels in git, text under `DATA_DIR`. S2 runs the retrieval
harness over it and over the existing in-domain paper-level set under a pre-registered rule,
then the new set's locked test once. R1–R4 harden and document what M3 and M4 built. Nothing
here adds a service; deployment stays deferred.

**What this plan replaces.** Old M5 (P5.1 CI, P5.2 tracing/limits/costs, P5.4 isolation and
latency, P5.5 case study and release gate). P5.3's staged deployment, P5.2's public rate
limiting and per-user quotas stay **deferred, not deleted**, until a deployment is decided.

**Tech Stack:** Python 3.12, the existing harness (`eval retrieval`, `eval compare`), the
chat backend and web app from the two 2026-10-11 plans. No new dependency.

**Spec:** [Technical specification](../specs/2026-09-05-ml-research-copilot-design.md), revised
2026-10-11: §7 (a change to `rerank.text` is a release decision), §11 (G5), §12 (operations).
Depends on the [chat backend plan](2026-10-11-chat-backend.md) (C1–C7) and the
[web app plan](2026-10-11-web-app.md) (W1–W4); S1–S2 depend only on M2 and may run earlier.

**Status:** Draft for review, 2026-10-11. No task started.

## Global Constraints

- Development tunes, validation chooses, and a test split is scored once with
  `--locked-test` for a release decision; the rule is committed before the validation run.
- Evaluate and time only on an idle GPU (30 s median ≤ 20%, no Linux compute process); a run
  that was contended or degraded a query is set aside and redone.
- Every priced call is under the ledger and `--max-spend-usd`; the daily cap is $3.
- Text derived from corpus papers (generated questions) lives under `DATA_DIR`; the repository
  keeps ids, labels, splits and digests. The live corpus `copilot_v2` is read, never written.
- Backups contain user data: they are written only under `${DATA_DIR}/backups/`, never in git,
  never in a report.
- Deleting `DATA_DIR` still deletes everything; nothing in this plan writes elsewhere.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A generated question gives its paper away or copies the passage.** Expected: rejected
   mechanically (distinctive title terms, title bigrams, eight or more consecutive words from
   the passage), retried once, then dropped and counted. Pinned in S1.
2. **The chunk-reading reranker wins on passages but loses on paper-level queries.** Expected:
   not promoted; the paper-level guard is part of the rule, not an afterthought. Pinned in S2.
3. **A user deletes their account.** Expected: their chats, messages and evidence snapshots
   are gone at once, their tokens stop working, and the spend ledger keeps its rows (it holds
   no text and no user id). Pinned in R2.
4. **Restoring a backup over live data.** Expected: refused unless the target tables are empty
   or `--replace` is given; a backup from another schema revision is refused. Pinned in R2.
5. **Logs leak content.** Expected: a request log line has method, route template, status,
   latency, request id and nothing a user typed. Pinned in R1.

---

### Task S1: A section-level query set from our corpus

**Files:**
- Create: `prompts/eval/section-question-v1.yaml`
- Create: `backend/src/copilot/evaluation/section_queries.py`
- Modify: `backend/src/copilot/cli.py` (`eval section-queries`)
- Create: `data/fixtures/retrieval-sections/{queries,qrels,splits}.jsonl` (no query text)
- Test: `backend/tests/unit/test_section_queries.py`, `backend/tests/integration/test_section_sampling.py`
- Create: `reports/s1-section-queries.md`

**Interfaces:**
- Consumes: `distinctive_terms`, `leaks` (C7, `evaluation/conversations.py`); `ChatClient`,
  `build_client`, `SpendLedger`, `ResponseCache`; `QueryRecord`, `QrelRecord`, `SplitRecord`,
  `Dataset`, `write_dataset`, `validate_dataset` (`evaluation/datasets.py`).
- Produces:
  - `section_queries.ChunkSample(chunk_id, paper_id, title, venue, year, section, text)`;
    `sample_chunks(engine, *, per_venue_year: int, seed: int, min_tokens: int = 200) -> list[ChunkSample]`
    (at most one chunk per paper); `longest_copied_run(question: str, passage: str) -> int`;
    `acceptable(question, sample) -> str | None` (the rejection reason, or None);
    `assign(samples, *, development: int, validation: int, test: int, seed: int) -> dict[str, str]`.
  - A dataset readable by `eval retrieval` with `dataset.path: data/fixtures/retrieval-sections`,
    `query_text: benchmarks/sections-v1/queries.jsonl`, `slice: all`, `labels: paper_id`.

- [ ] **Step 1: Write the failing unit tests**

`backend/tests/unit/test_section_queries.py`:
```python
"""Section-level questions: answerable from one passage, not by the paper's name or its words."""

from __future__ import annotations

from copilot.evaluation.section_queries import (
    ChunkSample,
    acceptable,
    assign,
    longest_copied_run,
)

SAMPLE = ChunkSample(
    chunk_id="c1",
    paper_id="p1",
    title="SimCSE: Simple Contrastive Learning of Sentence Embeddings",
    venue="EMNLP",
    year=2021,
    section="Training details",
    text="We train with a batch size of 64 and a learning rate of 3e-5 for one epoch on Wikipedia sentences.",
)


def test_a_copied_run_is_counted_in_words():
    assert longest_copied_run("batch size of 64 and a learning rate of 3e-5", SAMPLE.text) == 10
    assert longest_copied_run("What batch size was used?", SAMPLE.text) == 2


def test_questions_naming_the_paper_or_copying_it_are_refused():
    assert acceptable("What batch size does SimCSE train with?", SAMPLE) == "names_paper"
    assert acceptable("What batch size does simple contrastive learning use?", SAMPLE) == "names_paper"
    copied = "Do they train with a batch size of 64 and a learning rate of 3e-5?"
    assert acceptable(copied, SAMPLE) == "copies_passage"
    assert acceptable("", SAMPLE) == "empty"
    assert acceptable("What batch size and learning rate were used to train the sentence encoder?", SAMPLE) is None


def test_assignment_is_deterministic_and_by_paper():
    samples = [
        ChunkSample(f"c{i}", f"p{i}", "t", "ICLR", 2024, "Method", "x") for i in range(30)
    ]
    first = assign(samples, development=10, validation=10, test=10, seed=42)
    assert first == assign(samples, development=10, validation=10, test=10, seed=42)
    assert sorted(first.values()).count("test") == 10 and len(first) == 30
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run --project backend pytest backend/tests/unit/test_section_queries.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'copilot.evaluation.section_queries'`.

- [ ] **Step 3: Implement the pure parts and the sampler**

`backend/src/copilot/evaluation/section_queries.py`:
```python
"""A section-level query set from our own corpus (release plan S1).

Open RAG Bench asks questions a single passage answers; that is what the chunk-reading
reranker was better at there. This builds the same kind of question over our corpus so the
search page's reranker can be decided on our papers: one body passage per sampled paper,
stratified by venue-year, one model-written question per passage, the passage's paper as
the single gold. A question that names the paper (its distinctive title terms or two
consecutive title words) or copies eight or more consecutive words of the passage is
rejected: the first would be a lookup by name, the second a lookup by string.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Engine, text

from .conversations import distinctive_terms, leaks

COPY_LIMIT = 8
_WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


@dataclass(frozen=True)
class ChunkSample:
    chunk_id: str
    paper_id: str
    title: str
    venue: str
    year: int
    section: str
    text: str


def _words(value: str) -> list[str]:
    return _WORD.findall(value.lower())


def longest_copied_run(question: str, passage: str) -> int:
    """The longest run of consecutive words the question shares with the passage."""

    left, right = _words(question), _words(passage)
    best = 0
    previous = [0] * (len(right) + 1)
    for word in left:
        current = [0] * (len(right) + 1)
        for j, other in enumerate(right, start=1):
            if word == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def acceptable(question: str, sample: ChunkSample) -> str | None:
    """None when the question may be used; otherwise why not."""

    if not question.strip():
        return "empty"
    if leaks(question, distinctive_terms(sample.title, "", ""), sample.title):
        return "names_paper"
    if longest_copied_run(question, sample.text) >= COPY_LIMIT:
        return "copies_passage"
    return None


def assign(
    samples: Sequence[ChunkSample], *, development: int, validation: int, test: int, seed: int
) -> dict[str, str]:
    """query id (the chunk id) → split, papers shuffled once by seed."""

    ordered = sorted(samples, key=lambda sample: sample.paper_id)
    random.Random(seed).shuffle(ordered)
    sizes = (("development", development), ("validation", validation), ("test", test))
    splits: dict[str, str] = {}
    start = 0
    for name, size in sizes:
        for sample in ordered[start : start + size]:
            splits[sample.chunk_id] = name
        start += size
    return splits


_SAMPLE_SQL = """
with candidates as (
    select c.id::text as chunk_id, p.id::text as paper_id, p.title, v.name as venue,
           p.publication_year as year, c.section_path as section, c.text,
           row_number() over (partition by p.id order by md5(c.id::text || :seed)) as in_paper
    from chunks c
    join paper_versions pv on pv.id = c.paper_version_id and pv.parse_status = 'parsed'
    join papers p on p.id = pv.paper_id and p.merged_into is null
    join venues v on v.id = p.venue_id
    where c.kind = 'body' and c.token_count >= :min_tokens
), one_per_paper as (
    select *, row_number() over (
        partition by venue, year order by md5(paper_id || :seed)
    ) as in_venue_year
    from candidates where in_paper = 1
)
select chunk_id, paper_id, title, venue, year, section, text
from one_per_paper where in_venue_year <= :per_venue_year
order by venue, year, chunk_id
"""


def sample_chunks(
    engine: Engine, *, per_venue_year: int, seed: int, min_tokens: int = 200
) -> list[ChunkSample]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(_SAMPLE_SQL),
            {"seed": str(seed), "min_tokens": min_tokens, "per_venue_year": per_venue_year},
        ).mappings()
        return [ChunkSample(**{key: row[key] for key in ChunkSample.__dataclass_fields__}) for row in rows]
```
Before running, read the `chunks` columns in `backend/migrations/versions/0002_chunks.py` and
confirm `kind = 'body'` is how body text is marked and `section_path` is the section column
(the integration fixture inserts `kind 'body'` and `section_path 'Method'`, so both exist);
also confirm references are a different `kind` and adjust the `where` clause to exclude them if
they are not.

- [ ] **Step 4: Run the unit tests, then write and run the sampler's integration test**

Run: `uv run --project backend pytest backend/tests/unit/test_section_queries.py -q`
Expected: 3 passed.

`backend/tests/integration/test_section_sampling.py`:
```python
"""Sampling is one passage per paper, body text only, deterministic."""

from __future__ import annotations

import pytest

from copilot.evaluation.section_queries import sample_chunks

pytestmark = pytest.mark.integration


def test_one_body_passage_per_paper_and_the_same_twice(search_corpus):
    first = sample_chunks(search_corpus.engine, per_venue_year=10, seed=42, min_tokens=1)
    assert first == sample_chunks(search_corpus.engine, per_venue_year=10, seed=42, min_tokens=1)
    assert len({sample.paper_id for sample in first}) == len(first) == len(search_corpus.ids)
    capped = sample_chunks(search_corpus.engine, per_venue_year=1, seed=42, min_tokens=1)
    assert len(capped) == len({(s.venue, s.year) for s in first})
```
Run: `uv run --env-file .env.test --project backend pytest backend/tests/integration/test_section_sampling.py -q`
Expected: 1 passed.

- [ ] **Step 5: The question prompt and the build command**

`prompts/eval/section-question-v1.yaml`:
```yaml
# Section-level question writing, version 1 (release plan S1). Output is stored under DATA_DIR.
name: section-question-v1
system: >-
  You write one search question a researcher might ask that this passage from an ML paper
  answers. The question must be answerable from the passage, must not name the paper, its
  method, model or dataset names, and must not copy phrases from the passage: use your own
  words. Ask about a specific finding, setting, design choice or result, not a summary.
  Reply with JSON only: {"question": "..."}.
user: |-
  Section: {section}

  Passage:
  {passage}
```
Add `eval section-queries --database-url <live corpus> --per-venue-year 15 --max-spend-usd 0.60`
to `cli.py`. It samples with `seed=42` and `min_tokens=200`, keeps the first 600 samples in
venue-year order when more exist, writes one question per sample with `openai-gpt-6-luna`
(`ChatClient.complete`, purpose `evaluation`, run id `section-queries-<UTC stamp>`, the
response cache on), parses `{"question"}` with `loads_lenient`, applies `acceptable`, retries a
rejected one once with the user message "That question was rejected ({reason}). Write a
different one that follows the rules.", drops it if rejected again, then:
`assign(kept, development=200, validation=200, test=200, seed=42)` (smaller splits pro rata if
fewer than 600 were kept), writes `QueryRecord(query_id=chunk_id, query=<text>,
query_set="sections_v1", workflow="grounded_qa", specificity=1, quality=2, in_domain=True,
source="copilot_v2@<active release id>", author="generated by openai-gpt-6-luna,
section-question-v1")`, `QrelRecord(query_id=chunk_id, corpusid=paper_id, grade=1,
paper_id=paper_id)` and `SplitRecord(query_id=chunk_id, family_id=paper_id, split=...)` with
`write_dataset(..., include_query_text=False)` to `data/fixtures/retrieval-sections/`, and the
texts as `{"query_id", "query"}` lines to `${DATA_DIR}/benchmarks/sections-v1/queries.jsonl`.
It prints kept and rejected counts by reason. Unit-test the command with a scripted
`ChatClient` transport (the pattern in `test_generation.py`) over three fixture samples:
one accepted, one accepted on retry, one dropped.

- [ ] **Step 6: Build it (spends about $0.30) and validate**

```bash
uv run --env-file .env --project backend python -m copilot.cli eval section-queries \
  --database-url postgresql+psycopg://copilot:copilot@localhost:5432/copilot_v2 \
  --per-venue-year 15 --max-spend-usd 0.60
uv run --project backend python -m copilot.cli eval validate-dataset --path data/fixtures/retrieval-sections
grep -c '"query"' data/fixtures/retrieval-sections/queries.jsonl   # expect 0: no text in git
```
Read 30 development questions beside their passages (under `DATA_DIR`) and classify each as
answerable-from-passage / ambiguous / wrong; record the counts in
`reports/s1-section-queries.md` with the build's counts, cost and commands. If more than 5 of
30 are wrong, revise the prompt as `section-question-v2.yaml` and rebuild before any S2 run.

- [ ] **Step 7: Commit**

```bash
git add prompts/eval/section-question-v1.yaml backend/src/copilot/evaluation/section_queries.py \
  backend/src/copilot/cli.py backend/tests/unit/test_section_queries.py \
  backend/tests/integration/test_section_sampling.py data/fixtures/retrieval-sections \
  reports/s1-section-queries.md
git commit -m "feat: build a section-level query set from our corpus"
```

---

### Task S2: Decide what the search page's reranker reads

**Files:**
- Modify: `backend/src/copilot/search/service.py` (`SearchConfig.rerank_pair_max_tokens`, loaded from `rerank.pair_max_tokens`)
- Modify: `backend/src/copilot/search/api.py` (`default_search_service` applies the pair budget)
- Test: modify `backend/tests/unit/test_regression.py` or the search config unit test that loads `configs/search.yaml`
- Create: `configs/experiments/m5-rerank-text.yaml`
- Create: `reports/retrieval/m5-rerank-text/`, `reports/m5-rerank-text.md`
- Modify (only if promoted): `configs/search.yaml`, `configs/chat.yaml`, spec §7, `CLAUDE.md`

**Interfaces:**
- Consumes: S1's dataset; the existing in-domain dataset (`data/fixtures/retrieval`, `slice: in_domain`);
  the harness's variant `search:` overrides and `pair_max_tokens`.
- Produces: `SearchConfig.rerank_pair_max_tokens: int | None = None`; the recorded decision.

- [ ] **Step 1: The pair-budget knob, test first**

Append to the unit test module that already loads `configs/search.yaml` (find it with
`grep -ln "load_search_config" backend/tests/unit`):
```python
def test_the_reranker_pair_budget_is_optional_and_validated(tmp_path):
    from copilot.search.service import load_search_config

    shipped = load_search_config("configs/search.yaml")
    assert shipped.rerank_pair_max_tokens is None
    path = tmp_path / "search.yaml"
    path.write_text(Path("configs/search.yaml").read_text().replace("  text: abstract", "  text: abstract\n  pair_max_tokens: 512"))
    assert load_search_config(path).rerank_pair_max_tokens == 512
    path.write_text(Path("configs/search.yaml").read_text().replace("  text: abstract", "  text: abstract\n  pair_max_tokens: 0"))
    with pytest.raises(ValueError):
        load_search_config(path)
```
Run it: FAIL with `AttributeError: ... rerank_pair_max_tokens`. Then in `SearchConfig` add
`rerank_pair_max_tokens: int | None = None` with the `__post_init__` check
`if self.rerank_pair_max_tokens is not None and self.rerank_pair_max_tokens < 1: raise ValueError("rerank_pair_max_tokens must be at least 1")`,
read it in `load_search_config` as
`pair = (raw.get("rerank") or {}).get("pair_max_tokens")` → `rerank_pair_max_tokens=None if pair is None else int(pair)`,
add it to `SearchService.identity()` when set (beside `rerank_text`), and in
`default_search_service` replace `reranker=models.reranker` with:
```python
        reranker=(
            models.reranker.with_pair_budget(config.rerank_pair_max_tokens)
            if config.rerank_pair_max_tokens is not None
            and hasattr(models.reranker, "with_pair_budget")
            else models.reranker
        ),
```
Run the test module and `backend/tests/integration/test_search_service.py`: pass.

- [ ] **Step 2: Register the experiment**

`configs/experiments/m5-rerank-text.yaml`:
```yaml
# Release plan S2 (registered by the commit that adds this file, before any validation number):
# should the search page's cross-encoder read the passage that found each paper (title +
# chunk, haiku.rag's mechanism, ORB +0.180 nDCG@10) instead of the title and abstract?
#
# Two datasets, two runs of this file's variants: S1's section-level set (primary) and the
# in-domain paper-level set (guard), each on its validation split:
#   eval retrieval --config configs/experiments/m5-rerank-text.yaml --split validation --out reports/retrieval/m5-rerank-text
#   eval retrieval --config configs/experiments/m5-rerank-text-papers.yaml --split validation --out reports/retrieval/m5-rerank-text-papers
#
# Pre-registered rule. Between the two chunk-reading variants, the candidate is the one with
# the higher section-level validation nDCG@10. It is promoted over `hybrid_rerank` only if,
# on validation: (1) its section-level nDCG@10 gain has a paired 95% interval wholly above
# zero; (2) on the paper-level set its nDCG@10 interval rules out losing more than 0.03;
# (3) its end-to-end p95 is at most 3.0 s and it degraded no query, on both sets. A promoted
# candidate is then scored once on the section-level test split with --locked-test; the
# promotion stands only if the test's paired nDCG@10 interval is wholly above zero.
schema_version: 1
name: m5-rerank-text

dataset:
  path: data/fixtures/retrieval-sections
  query_text: benchmarks/sections-v1/queries.jsonl
  slice: all
  labels: paper_id

corpus:
  release: m2-20260924T095724Z
  papers: database

search: configs/search.yaml
models: configs/models.yaml

metrics:
  recall_at: [10, 50]
  ndcg_at: [10]
  mrr_at: [10]
  depth: 50

bootstrap:
  resamples: 1000
  seed: 42
  confidence: 0.95

variants:
  - name: hybrid_rerank
    mode: hybrid_rerank
    candidates: chunks
  - name: title_chunk_1024
    mode: hybrid_rerank
    candidates: chunks
    changes: the reranker reads title + the chunk that found the paper, 1,024-token pairs, 2.5 s rerank deadline
    search:
      rerank: {text: title_chunk}
      deadlines_seconds: {rerank: 2.5}
  - name: title_chunk_512
    mode: hybrid_rerank
    candidates: chunks
    changes: as title_chunk_1024 with 512-token pairs under the shipped 1.5 s deadline
    pair_max_tokens: 512
    search:
      rerank: {text: title_chunk}

decision:
  primary: ndcg@10
  order: [hybrid_rerank, title_chunk_512, title_chunk_1024]
  latency_p95_seconds: 3.0
  max_drop: 0.03
  require_undegraded: true
```
`configs/experiments/m5-rerank-text-papers.yaml` is identical except `name:
m5-rerank-text-papers` and the `dataset` block of `configs/experiments/m2-first-stage.yaml`
(`path: data/fixtures/retrieval`, its `query_text`, `slice: in_domain`, no `labels` line). It
keeps the same `decision` block so the harness computes the same paired intervals, latency and
degradation checks; its own "promote" verdict is not the rule — only the candidate's nDCG@10
lower bound (guard 2) and its latency and degradation (guard 3) are read from it. Check both load with
`uv run --project backend python -c "from copilot.evaluation.retrieval import load_experiment as l; l('configs/experiments/m5-rerank-text.yaml'); l('configs/experiments/m5-rerank-text-papers.yaml')"`
(use the loader's real name from `retrieval.py`). Commit both before any run:
```bash
git add backend/src/copilot/search configs/experiments/m5-rerank-text.yaml \
  configs/experiments/m5-rerank-text-papers.yaml backend/tests/unit
git commit -m "feat: register the search reranker-text decision on our corpus"
```

- [ ] **Step 3: Development read (idle GPU, no spend)**

```bash
uv run --env-file .env --project backend python -m copilot.cli eval retrieval \
  --config configs/experiments/m5-rerank-text.yaml --split development --out reports/retrieval/m5-rerank-text
```
Read the three rows, rerank p95 and degraded counts. Development decides nothing; if every
chunk-reading variant degrades queries even on an idle GPU, record it and stop: the rule
cannot promote, and S2 ends with "retain abstract".

- [ ] **Step 4: Validation (idle GPU)**

```bash
uv run --env-file .env --project backend python -m copilot.cli eval retrieval \
  --config configs/experiments/m5-rerank-text.yaml --split validation --out reports/retrieval/m5-rerank-text
uv run --env-file .env --project backend python -m copilot.cli eval retrieval \
  --config configs/experiments/m5-rerank-text-papers.yaml --split validation --out reports/retrieval/m5-rerank-text-papers
```
Read the paired differences against `hybrid_rerank` from each run's `report.md`. Apply the rule
exactly; write the outcome down before Step 5.

- [ ] **Step 5: Locked test, only if promoted (idle GPU)**

```bash
uv run --env-file .env --project backend python -m copilot.cli eval retrieval \
  --config configs/experiments/m5-rerank-text.yaml --split test --locked-test --out reports/retrieval/m5-rerank-text
```
If the test confirms: set `rerank.text: title_chunk` in `configs/search.yaml` (and
`rerank.pair_max_tokens: 512` with the shipped deadline, or `deadlines_seconds.rerank: 2.5`,
per the chosen variant), amend spec §7 and its comment block, and remove `search.overrides`
from `configs/chat.yaml` when the shipped values now equal them (the chat then follows the
search page). If it does not, nothing ships and the report says so.

- [ ] **Step 6: Report, tracker, commit**

`reports/m5-rerank-text.md`: the question, the rule, commands, GPU samples, development,
validation (both sets), test (if run), latency and degradation, the decision.
```bash
git add reports/retrieval/m5-rerank-text reports/retrieval/m5-rerank-text-papers reports/m5-rerank-text.md \
  docs/superpowers/progress.md configs/search.yaml configs/chat.yaml docs/superpowers/specs CLAUDE.md
git commit -m "feat: decide the search reranker's text on our corpus"
```

---

### Task R1: Request logs and cost reports

**Files:**
- Create: `backend/src/copilot/observability.py`
- Modify: `backend/src/copilot/app.py` (install the request log middleware)
- Modify: `backend/src/copilot/models/spend.py` (`spend_report` groups by purpose too)
- Modify: `backend/src/copilot/cli.py` (`chat stats`)
- Test: `backend/tests/unit/test_observability.py`, `backend/tests/integration/test_chat_stats.py`

**Interfaces:**
- Consumes: `request_id(request)`; `chat_messages.trace`, `.cost_usd`, `.status`, `.error_code`.
- Produces: one JSON log line per request on logger `copilot.requests`:
  `{"request_id", "method", "route", "status", "ms"}`; `spend_report(...)["purposes"]`;
  `chat_stats(engine, *, day) -> dict` (turns, by status, error codes, p50/p95 per stage, cost).

- [ ] **Step 1: Write the failing tests**

`backend/tests/unit/test_observability.py`:
```python
"""A request log line says what happened and nothing a user typed."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot.observability import install_request_log


def test_one_line_per_request_with_the_route_template_not_the_path(caplog):
    app = FastAPI()
    install_request_log(app)

    @app.get("/v1/chats/{chat_id}")
    def chat(chat_id: str) -> dict[str, str]:
        return {"ok": chat_id}

    with caplog.at_level(logging.INFO, logger="copilot.requests"):
        TestClient(app).get("/v1/chats/secret-looking-id?q=private+question")
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "copilot.requests"]
    assert len(lines) == 1
    line = lines[0]
    assert line["route"] == "/v1/chats/{chat_id}" and line["status"] == 200
    assert set(line) == {"request_id", "method", "route", "status", "ms"}
    assert "secret-looking-id" not in caplog.text and "private" not in caplog.text
```
`backend/tests/integration/test_chat_stats.py` seeds two completed turns and one failed turn
through `ChatStore` (C2's test helpers) with traces `{"condense": 0.1, "retrieve": 0.5,
"generate": 2.0}` and asserts `chat_stats(engine, day=<today>)` reports 3 turns,
`{"complete": 2, "failed": 1}`, `{"llm_timeout": 1}`, and `retrieve` p50 0.5.

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run --project backend pytest backend/tests/unit/test_observability.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'copilot.observability'`.

- [ ] **Step 3: Implement**

`backend/src/copilot/observability.py`:
```python
"""Request logs (spec §12): one JSON line per request, never anything a user typed.

The route is the matched template (``/v1/chats/{chat_id}``), not the path, so ids and query
strings stay out of the log; bodies are never read. A plain ASGI middleware, not
``@app.middleware("http")``, so a streamed chat turn passes through untouched and the line is
written when the stream ends: ``ms`` is the whole turn.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("copilot.requests")


class RequestLog:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        status = 500

        async def capture(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            state: dict[str, Any] = scope.get("state") or {}
            request = state.get("request_id")
            logger.info(
                json.dumps(
                    {
                        "request_id": None if request is None else str(request),
                        "method": scope["method"],
                        "route": getattr(scope.get("route"), "path", "unmatched"),
                        "status": status,
                        "ms": round((time.monotonic() - started) * 1000, 1),
                    },
                    separators=(",", ":"),
                )
            )


def install_request_log(app: FastAPI) -> None:
    app.add_middleware(RequestLog)
```
`request_id` is whatever `search.api.request_id(request)` stored on the request state, so the
log line and the response body carry the same id; a route that never asks for one logs
`null`. Call `install_request_log(app)` in `create_app` after the routers. In `spend_report`, run the same
aggregate grouped by `purpose` and return it as `"purposes": {purpose: {"calls", "cost_usd"}}`
beside `"models"`. Add `chat_stats(engine, *, day)` to `chat/store.py` (SQL over
`chat_messages` where `role = 'assistant'` and `created_at` within the UTC day: counts by
`status`, by `error_code`, `percentile_cont(0.5|0.95)` of each `trace` key, sum of `cost_usd`)
and the CLI `chat stats [--day YYYY-MM-DD] [--database-url]`.

- [ ] **Step 4: Run the tests and checks; commit**

```bash
uv run --project backend pytest backend/tests/unit/test_observability.py -q
uv run --env-file .env.test --project backend pytest backend/tests/integration/test_chat_stats.py -q
uv run --project backend ruff check backend && uv run --project backend mypy --config-file backend/pyproject.toml backend/src
git add backend/src/copilot/observability.py backend/src/copilot/app.py backend/src/copilot/models/spend.py \
  backend/src/copilot/chat/store.py backend/src/copilot/cli.py backend/tests/unit/test_observability.py \
  backend/tests/integration/test_chat_stats.py
git commit -m "feat: log requests without content and report chat cost and timings"
```

---

### Task R2: Account deletion, and backup and restore of user data

**Files:**
- Modify: `backend/src/copilot/identity/users.py` (`delete`), `identity/api.py` (`DELETE /v1/me`)
- Modify: `backend/src/copilot/cli.py` (`users delete`, `backup create`, `backup restore`)
- Create: `backend/src/copilot/backup.py`
- Create: `frontend/app/account/page.tsx`, `frontend/components/DeleteAccount.tsx`; modify `frontend/components/Header.tsx` (an Account link)
- Test: `backend/tests/integration/test_account_deletion.py`, `backend/tests/integration/test_backup.py`, `frontend/e2e/account.spec.ts`
- Modify: `frontend/openapi.json`, `frontend/lib/api-types.ts`, `frontend/e2e/mock-backend.mjs`

**Interfaces:**
- Consumes: C1's `UserStore`, `CurrentPrincipal`, `verify_password`; C2's tables.
- Produces: `DELETE /v1/me` with body `{"password": str}` → 204 (401 `invalid_credentials` on
  a wrong password); `UserStore.delete(user_id) -> bool`;
  `backup.create_backup(engine, out: Path) -> dict` and
  `backup.restore_backup(engine, source: Path, *, replace: bool) -> dict`;
  `USER_TABLES = ("users", "chats", "chat_messages", "message_evidence")`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/integration/test_account_deletion.py`:
```python
"""Deleting an account deletes its chats at once and ends its sessions."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from copilot.chat.store import ChatStore
from copilot.contracts import ChatRequest, Principal

pytestmark = pytest.mark.integration
PASSWORD = "correct horse battery staple"


def test_delete_me_removes_chats_and_ends_the_session(api_client, make_user, auth_headers, migrated_database):
    user_id, _ = make_user()
    store = ChatStore(migrated_database)
    store.begin_turn(Principal(user_id=user_id), ChatRequest(request_id=uuid4(), message="hi"), stale_after=timedelta(minutes=4))
    headers = auth_headers(user_id)
    wrong = api_client.request("DELETE", "/v1/me", json={"password": "not the password!"}, headers=headers)
    assert wrong.status_code == 401
    assert api_client.request("DELETE", "/v1/me", json={"password": PASSWORD}, headers=headers).status_code == 204
    assert api_client.get("/v1/me", headers=headers).status_code == 401
    with migrated_database.connect() as connection:
        left = connection.execute(text("select count(*) from chats where user_id = :id"), {"id": user_id}).scalar_one()
    assert left == 0
```
`backend/tests/integration/test_backup.py`:
```python
"""User data backs up to files under DATA_DIR and restores only where it is safe to."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from copilot.backup import BackupError, create_backup, restore_backup
from copilot.chat.store import ChatStore
from copilot.contracts import ChatRequest, Principal

pytestmark = pytest.mark.integration


def test_a_backup_restores_into_empty_tables_and_refuses_live_ones(
    migrated_database, make_user, test_settings, tmp_path
):
    user_id, _ = make_user()
    ChatStore(migrated_database).begin_turn(
        Principal(user_id=user_id), ChatRequest(request_id=uuid4(), message="kept"), stale_after=timedelta(minutes=4)
    )
    manifest = create_backup(migrated_database, tmp_path / "b1")
    assert manifest["tables"]["users"]["rows"] >= 1 and manifest["revision"]
    with pytest.raises(BackupError, match="target_not_empty"):
        restore_backup(migrated_database, tmp_path / "b1", replace=False)
    restore_backup(migrated_database, tmp_path / "b1", replace=True)
    with migrated_database.connect() as connection:
        assert connection.execute(text("select count(*) from chat_messages where text = 'kept'")).scalar_one() == 1
    (tmp_path / "b1" / "manifest.json").write_text(
        (tmp_path / "b1" / "manifest.json").read_text().replace(manifest["revision"], "0000_other")
    )
    with pytest.raises(BackupError, match="revision_mismatch"):
        restore_backup(migrated_database, tmp_path / "b1", replace=True)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run --env-file .env.test --project backend pytest backend/tests/integration/test_account_deletion.py backend/tests/integration/test_backup.py -q`
Expected: FAIL — `DELETE /v1/me` is 405 and `copilot.backup` does not exist.

- [ ] **Step 3: Implement deletion**

In `users.py`:
```python
    def delete(self, user_id: UUID) -> bool:
        """Delete the account; chats, messages and evidence go with it (ON DELETE CASCADE).
        The spend ledger keeps its rows: they hold no text and no user id."""

        with self._engine.begin() as connection:
            return connection.execute(text("delete from users where id = :id"), {"id": user_id}).rowcount > 0

    def check_password(self, user_id: UUID, password: str) -> bool:
        with self._engine.connect() as connection:
            stored = connection.execute(
                text("select password_hash from users where id = :id and disabled_at is null"),
                {"id": user_id},
            ).scalar_one_or_none()
        return verify_password(None if stored is None else str(stored), password)
```
In `identity/api.py`:
```python
class DeleteAccountRequest(ContractModel):
    password: str = Field(min_length=1, max_length=1024)

    @api.delete("/v1/me", status_code=204, response_class=Response)
    def delete_me(body: DeleteAccountRequest, principal: CurrentPrincipal, request: Request) -> Response:
        users = _identity(request).users
        if not users.check_password(principal.user_id, body.password):
            raise ApiError(401, "invalid_credentials", "the password is incorrect")
        users.delete(principal.user_id)
        return Response(status_code=204)
```
(the route goes inside `router()`; the class at module level; import `Response` from
`fastapi`). Add `users delete --email` to the CLI calling `UserStore.delete` after looking the
id up by email, printing `{"email", "deleted"}`.

- [ ] **Step 4: Implement backup and restore**

`backend/src/copilot/backup.py`:
```python
"""Backups of user data (spec §12): accounts, chats, messages and evidence snapshots.

The corpus is rebuilt from its own snapshots (corpus export/restore); this covers only what
users created. Each table is copied out as CSV with PostgreSQL's COPY, beside a manifest of
row counts, sha256 digests and the schema revision. A restore refuses another revision and,
unless asked to replace, any target table that already has rows.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import psycopg
from sqlalchemy import Engine, text

# Table names are this constant, never input: the f-strings below interpolate only these.
USER_TABLES = ("users", "chats", "chat_messages", "message_evidence")


class BackupError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def _revision(engine: Engine) -> str:
    with engine.connect() as connection:
        return str(connection.execute(text("select version_num from alembic_version")).scalar_one())


def _driver(engine: Engine) -> tuple[Any, psycopg.Connection[Any]]:
    raw = engine.raw_connection()
    return raw, cast(psycopg.Connection[Any], raw.driver_connection)


def create_backup(engine: Engine, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    tables: dict[str, Any] = {}
    raw, driver = _driver(engine)
    try:
        with driver.cursor() as cursor:
            for table in USER_TABLES:
                target = out / f"{table}.csv"
                with target.open("wb") as handle, cursor.copy(
                    f"COPY {table} TO STDOUT (FORMAT csv, HEADER)"
                ) as copy:
                    for chunk in copy:
                        handle.write(bytes(chunk))
                data = target.read_bytes()
                tables[table] = {
                    "rows": max(0, data.count(b"\n") - 1),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
        driver.commit()
    finally:
        raw.close()
    manifest = {"created_at": datetime.now(UTC).isoformat(), "revision": _revision(engine), "tables": tables}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def restore_backup(engine: Engine, source: Path, *, replace: bool) -> dict[str, Any]:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["revision"] != _revision(engine):
        raise BackupError("revision_mismatch", f"{manifest['revision']} != {_revision(engine)}")
    for table, facts in manifest["tables"].items():
        if hashlib.sha256((source / f"{table}.csv").read_bytes()).hexdigest() != facts["sha256"]:
            raise BackupError("checksum_mismatch", table)
    raw, driver = _driver(engine)
    try:
        with driver.cursor() as cursor:
            for table in USER_TABLES:
                cursor.execute(f"select exists (select 1 from {table})")
                row = cursor.fetchone()
                if row is not None and row[0] and not replace:
                    raise BackupError("target_not_empty", table)
            if replace:
                cursor.execute(f"truncate {', '.join(USER_TABLES)}")
            for table in USER_TABLES:
                with (source / f"{table}.csv").open("rb") as handle, cursor.copy(
                    f"COPY {table} FROM STDIN (FORMAT csv, HEADER)"
                ) as copy:
                    while chunk := handle.read(1 << 20):
                        copy.write(chunk)
        driver.commit()
    except Exception:
        driver.rollback()
        raise
    finally:
        raw.close()
    return {"restored": {table: manifest["tables"][table]["rows"] for table in USER_TABLES}}
```
(`USER_TABLES` is in foreign-key order, so a table is loaded after the ones it references.
The row count from newlines over-counts a value containing a newline; it is a sanity figure,
the digest is the check.) Add the CLI `backup create [--out]` (default
`${DATA_DIR}/backups/<UTC stamp>`, refusing any path outside `DATA_DIR`) and
`backup restore --from DIR [--replace] --database-url URL`.

- [ ] **Step 5: The account page**

`frontend/app/account/page.tsx` (server component: redirect to `/login?next=%2Faccount` without
the session cookie, as `app/chat/layout.tsx` does) renders a heading "Account", the sentence
"Deleting your account removes all of your chats at once. This cannot be undone." and
`<DeleteAccount />`: a client form with a labelled password input and a "Delete my account"
button that calls `fetch("/api/v1/me", {method: "DELETE", body: JSON.stringify({password})})`,
shows "The password is incorrect." on 401, and on 204 calls `DELETE /api/session` and
navigates to `/`. Add `<Link href="/account">Account</Link>` beside "Sign out" in `Header`.
Teach the scripted backend `DELETE /v1/me` (204 for the fixture password, 401 otherwise,
clearing its chats), regenerate `lib/api-types.ts`, and add `frontend/e2e/account.spec.ts`
covering the wrong password and the successful deletion (signed out, `/chat` redirects to
login), plus an axe check.

- [ ] **Step 6: Run everything; commit**

```bash
uv run --project backend python -m copilot.cli api export-schema --out frontend/openapi.json
uv run --env-file .env.test --project backend pytest backend/tests -q
uv run --project backend ruff check backend && uv run --project backend mypy --config-file backend/pyproject.toml backend/src
cd frontend && npm run api-types && npm run lint && npm run typecheck && npm test && npm run e2e && cd ..
git add backend/src/copilot/identity backend/src/copilot/backup.py backend/src/copilot/cli.py \
  backend/tests/integration/test_account_deletion.py backend/tests/integration/test_backup.py \
  frontend/app/account frontend/components/DeleteAccount.tsx frontend/components/Header.tsx \
  frontend/e2e frontend/openapi.json frontend/lib/api-types.ts
git commit -m "feat: let users delete their account and back up and restore user data"
```

---

### Task R3: Failure drills on the real stack

**Files:**
- Create: `reports/m5-drills.md`
- Modify: code only where a drill fails its expectation (each fix with its own regression test
  in the module that owns the behaviour, committed with this task)

**Interfaces:**
- Consumes: the whole local stack. Produces: the drill record for G5.

- [ ] **Step 1: Run each drill and record expected against observed**

On the local stack (API with `.env`, web app via `npm run start`, an account, an idle GPU):

| # | Do | Expected |
|---|---|---|
| 1 | `docker compose stop qdrant`, search in the web app, then send a chat message | Search shows a retryable error banner; the chat turn ends with an `error` frame (`search_unavailable` or `vector_unavailable`), stored `failed`; `/health/ready` is 503 `vector_unavailable` |
| 2 | `docker compose stop postgres`, load `/chat` | The page shows a retryable error, no stack trace; `/health/ready` 503 `database_unavailable` |
| 3 | Restart the API with `LLM_DAILY_SPEND_CAP_USD=0.0001`, send a chat message | `llm_spend_cap` message with the UTC reset; the turn `failed`; no ledger row over the cap |
| 4 | Restart the API with `OPENAI_API_KEY=` (empty) | Startup logs `chat unavailable: llm_unconfigured`; `POST /v1/chats/stream` 503 `chat_unavailable`; search and history still work |
| 5 | Start a chat turn, kill the API process (`kill -9`) mid-turn, restart it | The turn reads `interrupted` after restart (startup interrupt of stale turns needs `stale_after` to pass: shorten `timeouts_seconds.turn` to 5 for this drill and restore it) |
| 6 | Run a GPU-heavy process from another project, or, if none is running, record that the drill was not possible | Search results carry the degraded banner rather than failing |
| 7 | Sign in, wait past `ACCESS_TOKEN_TTL_SECONDS=300` (set for this drill), click a chat | Redirect to `/login?expired=1&next=…`; sign in returns to the chat |
| 8 | `uv run ... python -m copilot.cli backup create`, `users delete` a test account, `backup restore --replace` into a scratch database at head | The account and its chats are back in the scratch database; the live database is untouched |

Record each drill's commands, the observed responses (status codes, frame names, log lines —
never user text), and pass/fail in `reports/m5-drills.md`. A failed drill gets a fix and a
regression test in the owning module, then the drill is re-run and both runs are recorded.

- [ ] **Step 2: Commit**

```bash
git add reports/m5-drills.md
git commit -m "docs: record failure drills on the local stack"
```
(add any fix and its test to the same commit, staged by path).

---

### Task R4: Case study and the release gate

**Files:**
- Modify: `README.md` (or create it at the repository root if absent)
- Create: `docs/case-study.md`
- Modify: `docs/data-card.md`, `docs/superpowers/progress.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: every report named in the tracker. Produces: G5's evidence.

- [ ] **Step 1: Write the README**

The README says, in this order: what the product is (paper search and a cited-answer chatbot
over 85,729 accepted ML papers from 42 venue-years, run locally); a screenshot from
`reports/m4-web/`; how to run it (Docker services, `uv sync`, `alembic upgrade head`, `users
create`, `uvicorn`, `npm run start`), with the real commands from `CLAUDE.md`; the
architecture in five sentences and the §3 diagram; the measured results table with links:
G2 retrieval (LitSearch test nDCG@10 0.550, in-domain), ORB retrieval (hybrid 0.948 nDCG@10),
ORB answering (frozen 400: 0.880 [0.844, 0.908] under haiku.rag's rubric, plus C4's result),
follow-up answering (C7's frozen numbers and decision), the search reranker decision (S2); and
the limitations: model-model judge calibration, synthetic follow-ups and section questions,
3,802 papers without text, a shared GPU, local only. Every number is copied from its report,
never retyped from memory; run `grep` for each figure in its report before committing.

- [ ] **Step 2: Write the case study**

`docs/case-study.md`: the problem; the decisions and the evidence for each (chunk-level
candidates, the reranker's text, the rubric, structured generation, condensing, LangGraph for
the turn); what failed and was set aside (paper-level reranking on ORB, 256-token chunks for
answering, the abstract reranker losing gold papers); costs (`llm spend` totals by purpose);
what a deployment would add (the deferred P5.3, rate limits, per-user quotas).

- [ ] **Step 3: Close the gate**

G5 is met when: CI is green on the branch (backend and frontend jobs); S2's decision is
recorded; R1–R3 are done with their reports; the README and case study exist and every number
in them traces to a report. Update the tracker's gate board and session log; update the data
card's "what the product does with the corpus" section (search and grounded answers; no
redistribution of text beyond the passages shown to a signed-in user).
```bash
git add README.md docs/case-study.md docs/data-card.md docs/superpowers/progress.md CLAUDE.md
git commit -m "docs: write the case study and record the release gate"
```
