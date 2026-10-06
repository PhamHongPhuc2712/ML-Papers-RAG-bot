# Open RAG Bench, Text Slice: Retrieval and Grounded Answering Benchmark

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the shipped M2 retrieval stack and a first grounded generator on Vectara's
Open RAG Bench (ORB), text-only queries first. Retrieval is scored on all 1,914 text queries at
no cost. Answering is scored on a frozen, stratified 400-case sample through a hosted LLM under
the daily spend cap, with an LLM judge calibrated against hand labels. The result is the first
end-to-end answer-and-citation evidence for G4, on a public benchmark with 3,045 reference
answers rather than the 60 hand-written questions E5 plans.

**Architecture:** ORB's 1,000 arXiv PDFs are parsed and chunked by **our own pipeline** into a
separate database and a separate, never-activated release (`orb-v1`), the same shape as
`litsearch-v1` but with both collections. The existing retrieval harness scores it unchanged,
with one new per-query fact: whether the top paper's best evidence chunk falls in the gold
section. A minimal grounded generator, the P4.1 evidence retrieval plus a structured-claims
generation adapter, answers each sampled question with evidence ids that are validated before
the answer counts. An answer-equivalence judge scores correctness; a citation MAP scores
grounding; exact McNemar compares two configurations on identical cases.

**What is borrowed from haiku.rag (`ggozad/haiku.rag` 0.92.0, MIT), and what is not.**
Borrowed: the rule that an answer must declare its citations before it counts, with a single
repair when it does not; section-aware context expansion of a hit chunk within a character
budget; document-level MAP for retrieval and for cited documents; a pinned judge with a measured
agreement figure; per-case JSONL results paired by exact McNemar. Not borrowed: LanceDB, Docling,
the Pydantic AI agent loop with a Python sandbox, and its model stack (4B embedder, 4B reranker,
26B generator, 27B judge served by vLLM), none of which fit this project's storage, parser
boundary or a 16 GB laptop GPU.

**Tech Stack:** Python 3.12, PostgreSQL + Qdrant (existing), `httpx` chat client and spend
ledger (L1–L2, existing), the retrieval harness (P2.5/P2.6, existing). No new runtime
dependency.

**Spec:** [Technical specification](../specs/2026-09-05-ml-research-copilot-design.md):
§7 retrieval, §9 evidence and claims, §11 evaluation and G4, §12 spend cap. Also read the
[evidence assistant plan](2026-09-05-04-evidence-assistant.md) P4.1, P4.2 and P4.5, the
[benchmarks plan](2026-09-21-evaluation-benchmarks.md) E5–E7, and the
[LLM reranking plan](2026-10-01-llm-reranking.md) L1–L2 and L5.

**Status:** Draft for review, 2026-10-06. No task started. Requested by the developer after
exploring haiku.rag and ORB on 2026-10-06. The provider data-terms decision this plan needs was
recorded the same day (progress tracker, decisions table).

## Global Constraints

- Python 3.12; UTC timestamps; locked dependencies; pinned model and dataset revisions before
  any run.
- The hosted LLM API is the one paid resource. Every priced call reserves its worst case in the
  ledger under `LLM_DAILY_SPEND_CAP_USD` ($3 a day) before it reaches the network. An
  evaluation that calls an LLM needs `--max-spend-usd` and stops when the ledger passes it.
- **ORB is CC-BY-NC-4.0.** Its questions, answers, section text and PDFs live under `DATA_DIR`,
  never in git, never in a report, never in an export. The repository holds query ids, labels,
  split assignment and checksums, the LitSearch rule.
- Hugging Face and arxiv.org are contacted only by the offline fetch command in O1, at a pinned
  revision. Nothing in the serving path contacts either.
- `orb-v1` is never activated. The API's namespace guard already refuses to serve it; O2 asserts
  that.
- The retrieval harness never decides on a benchmark that is not the decision benchmark: ORB
  retrieval runs carry **no `decision` block**. The shipped configuration was chosen on LitSearch
  validation and confirmed on the locked test; ORB reads it, it does not tune it.
- Development cases tune prompts; the frozen 400 are scored once per configuration and never
  read for debugging. A configuration change after a frozen run is a new run, recorded beside
  the old one.
- GPU policy (2026-10-05): timing and deadline-bound runs only on an idle GPU; a run that shared
  it is set aside and redone.
- `mypy` always runs with `--config-file backend/pyproject.toml`. Never pass `.env` to pytest.

## Measured facts this plan is built on (checked 2026-10-06)

| Fact | Value | Source |
|---|---|---|
| Dataset | `vectara/open_ragbench`, revision `63f6b052ff83508b08e242db42263ee708815c26`, ungated, licence tag `cc-by-nc-4.0` | Hugging Face API |
| Layout | `pdf/arxiv/{queries,qrels,answers,pdf_urls}.json` (1.6 MB together) and `pdf/arxiv/corpus/<arxiv_id>.json`, 1,000 files, 742 MB, sections with markdown tables and base64 images | Hugging Face API; files read |
| Corpus | 1,000 arXiv papers, 2401 to 2503 ids, 396 of them gold for some query, 604 never gold | `qrels.json`, `pdf_urls.json` |
| Queries | 3,045; `type` abstractive 1,793 / extractive 1,252; `source` text 1,914, text-image 763, text-table 148, text-table-image 220 | `queries.json` |
| Labels | One `doc_id` and one integer `section_id` per query; queries per gold document min 1, median 9, max 10 | `qrels.json` |
| **Text slice** | 1,914 queries: extractive 1,021, abstractive 893; 387 gold documents, median 5 queries each, 350 documents with at least 2; query median 11 words; reference answer median 18 words, p90 42 | computed 2026-10-06 |
| haiku.rag's handling | Re-downloads the PDFs from arxiv.org and parses with Docling; ignores `section_id`; retrieval metric is MAP over the deduplicated document ids of the top 5 chunks, which with one gold document is reciprocal rank at depth 5 | `evaluations/datasets/open_rag_bench.py`, `evaluators/map.py` |
| haiku.rag's published ORB rows | Retrieval MAP 0.977–0.990 on all 3,045 queries; QA accuracy 0.92–0.97; cited MAP 0.80–0.99 depending on stack. Judge agreement with its previous judge: 95%, κ 0.90 on 120 cases | ggozad.github.io/haiku.rag/benchmarks |
| Our download policy | `arxiv.org` and `export.arxiv.org` are already allowed hosts; host, address, redirect, size and magic-byte checks apply | `corpus/download.py`, `configs/corpus.yaml` |
| Our parser on arXiv-style PDFs | 95% usable text on a 20-paper ICLR audit; 30 parse failures in 85,729 papers | `reports/m1-parser-audit.md`, `reports/m1-parse-failures.md` |
| `gpt-6-luna` prices | $0.10 input, $0.01 cached input, $0.50 output per 1M tokens; alias, served model recorded per call | `configs/llm.yaml` |

## Cost estimates

Assumptions, replaced by ledger figures after the pilot: a generation call sees 12 evidence
chunks of at most 6,000 tokens plus about 500 tokens of instructions and answers in about 400
tokens; a judge call sees question, reference answer and generated answer, about 1,500 tokens,
and answers in about 200 tokens.

| Item | Per case | Cases | Estimate |
|---|---|---|---|
| Generation, `gpt-6-luna` effort `none` | ≈ $0.0009 | | |
| Judge, `gpt-6-luna` effort `low` | ≈ $0.0003 | | |
| Pilot | ≈ $0.0012 | 20 | ≈ $0.03 |
| Development set, repeated while prompts settle | ≈ $0.0012 | 100 × up to 4 passes | ≈ $0.50 |
| Frozen sample, one configuration | ≈ $0.0012 | 400 | ≈ $0.50 |
| Frozen sample, second configuration (section expansion) | ≈ $0.0012 | 400 | ≈ $0.50 |
| Judge calibration | none, hand labels | 50 | $0 |
| Retrieval on 1,914 queries | none, local GPU | | $0 |

Whole plan: about **$1.50 to $2**, well under one day's cap even if the estimate is off by two.
A full 1,914-case QA run would be about $2.50 per configuration and is not planned; it stays
available later if an interval needs tightening. Reruns are free: responses are cached by
request digest (L2), and the judge's cache key includes the rubric digest.

## Why 400 cases, and what they support

The 95% interval half-width for an accuracy near 0.90, with the finite-population correction
over 1,914 queries:

| QA cases | Half-width | Estimated cost |
|---|---|---|
| 100 | ±0.057 | $0.12 |
| 200 | ±0.039 | $0.24 |
| **400** | **±0.026** | **$0.50** |
| 800 | ±0.016 | $1.00 |
| 1,914 | exact | $2.40 |

400 cases support an absolute accuracy with a stated interval and a paired comparison of two
configurations on identical cases, where exact McNemar reliably detects a difference of about
four to five points. The sample is stratified by `type` (extractive, abstractive) and capped at
**two queries per gold document**, because queries from one paper succeed or fail together and
an uncapped sample would understate its own variance. 387 gold documents allow up to 774 at
that cap, so 400 plus a disjoint 100-case development set fits.

What 400 cases do not support: a direct comparison with haiku.rag's 0.92 to 0.97. Those depend
on their judge and rubric. O5 reuses the same answer-equivalence rubric and calibrates our judge
against 50 blind hand labels, and the report states the judge beside every number.

## Decisions

1. **Corpus text: our own parser, from the PDFs.** O2 downloads the 1,000 PDFs named in
   `pdf_urls.json` and runs `pypdf-text-v2` and `paragraph-pack-v1` over them, so the benchmark
   measures the pipeline that serves `copilot_v2`. ORB's own Mistral-OCR sections are read only
   to locate the gold section inside our chunks (O3). A Docling adapter is out of scope for the
   text slice; if O2's parse failures or O3's section mapping show the parser is the limit, that
   becomes its own task behind the `PageAdapter` boundary.
2. **A separate database and release.** `copilot_orb` (PostgreSQL) and `orb-v1` (Qdrant, both
   collections), built by the existing `corpus export` and `search build-index`, never activated.
3. **The sample:** seed 42, stratified by `type`, capped at two per gold document; 400 frozen,
   100 development, disjoint; ids and labels in `data/fixtures/orb/`, text under `DATA_DIR`.
4. **Models:** generator `openai-gpt-6-luna` (effort `none`); judge `openai-gpt-6-luna-low`
   (effort `low`), the same provider. §11 asks for a judge separate from the generator; the
   project funds one provider, so the correlated-bias limitation is stated in every report and
   measured by the hand-label calibration, the E6/E7 position. This supersedes E6's
   `claude-opus-5`, which the LLM plan already flagged for revision.
5. **Retrieval metrics:** the harness's Recall@10/50, nDCG@10 and MRR@10, plus MRR at depth 5
   for the row beside haiku.rag's MAP, plus the new `section_hit@1`. Two variants of the first
   stage are scored, `candidates: papers` and `candidates: chunks`, and all four modes; no
   decision block.
6. **Answering metrics:** QA accuracy by the judge with a Wilson interval and a family bootstrap
   (family = gold document); `cited_map`, the MAP of cited documents against the gold document;
   `provenance_validity`, the share of answers whose every evidence id passed structural
   validation without repair; `abstention_rate`; cost and served models per run.
7. **Generation shape: one retrieve-then-answer pass**, spec §9's research mode, with the
   section-expansion variant as the one ablation. haiku.rag's multi-round agent loop with a
   sandbox is recorded as a later variant, not built here.
8. **Data sent to the provider:** accepted for OpenAI on 2026-10-06 (progress tracker). DeepSeek
   is not approved and no variant in this plan may name it.

## What this plan changes elsewhere

| Document | Change | Where |
|---|---|---|
| Spec §11 | ORB's text slice named as a G4 answer benchmark beside E5's question set; `section_hit@1` defined | O3, O5 |
| Benchmarks plan E6/E7 | Judge model becomes `gpt-6-luna` via the existing chat client; the answer-equivalence rubric is a sixth judge prompt; calibration reuses E7's κ floor. E6's five reference-free metrics are **not** replaced: they still apply to E5's unanswerable and multi-paper cases, which ORB lacks | O5 |
| M4 plan P4.1 | O4 delivers `evidence/{retrieve,citations}.py` with P4.1's interfaces and its public-corpus acceptance cases. P4.1 keeps the private-upload entry point and the two-paper misleading-abstract integration case | O4 |
| M4 plan P4.2 | O4 delivers `models/generation.py`, the `Generator` adapter with a deterministic fake, and the structured-claims output with one repair. P4.2 keeps routing, SSE streams, persistence and the HTTP API | O4 |
| `configs/evaluation.yaml` | `benchmarks.orb` entry: repo, revision, checksums, counts, licence, local path | O1 |
| `CLAUDE.md` | ORB commands and the licence rule | O5 |

## Sequencing

1. O1 and O2 need no model calls beyond the local embedder and reranker. O2's chunk build is
   small: about 1,000 papers, on the order of 100k chunks, under an hour on the GPU.
2. O3 runs on an idle GPU and spends nothing.
3. O4 is code and tests; its live fixture calls are under $0.10.
4. O5: pilot, development passes, calibration labels, then the frozen 400 once per configuration.
5. P3.1 is unaffected and can proceed in parallel; O4's modules are the ones P4.1 and P4.2 build
   on, so M4 starts with them in place.

## File structure

```text
backend/src/copilot/evaluation/orb.py                 fetch at a pinned revision, checksums, text-slice sampling, dataset files (O1)
backend/src/copilot/evaluation/orb_load.py            load the 1,000 papers into copilot_orb through the existing identity, parser and chunker (O2)
backend/src/copilot/evaluation/sections.py            locate a gold section inside our chunks; section_hit@1 (O3)
backend/src/copilot/evaluation/retrieval.py           per-query best-chunk id and section_hit (O3, modify)
backend/src/copilot/evidence/retrieve.py              bounded evidence retrieval; section expansion variant (O4)
backend/src/copilot/evidence/citations.py             structural validation of evidence ids and quotes (O4)
backend/src/copilot/models/generation.py              Generator adapter over the chat client; deterministic fake (O4)
backend/src/copilot/evaluation/qa.py                  the answering run: cases, generation, judge, per-case results, pairing (O5)
backend/src/copilot/evaluation/judge.py               answer-equivalence judge with reason-before-verdict output (O5)
backend/src/copilot/cli.py                            eval orb-fetch, orb-dataset, orb-load, qa, pair (O1, O2, O5, modify)
configs/evaluation.yaml                               benchmarks.orb (O1, modify)
configs/evidence.yaml                                 budgets: 10 papers, 40 chunk candidates, 12 evidence, 3 per paper, 6,000 tokens, expansion chars (O4)
configs/experiments/orb-text-retrieval.yaml           four modes × two candidate sources, no decision block (O3)
configs/experiments/orb-text-qa.yaml                  generator, judge, variants flat / section-expanded, budget (O5)
prompts/generate/grounded-claims-v1.yaml              the generation prompt, structured claims with evidence ids (O4)
prompts/judge/answer-equivalence-v1.yaml              the judge prompt, adapted from haiku.rag's rubric with attribution (O5)
data/fixtures/orb/{queries,qrels,splits}.jsonl        ids, labels, type, split; no text (O1)
data/fixtures/qa-smoke/                               synthetic cases and recorded judge responses for CI (O5)
reports/orb-text.md                                   evidence: parse, retrieval, QA, calibration, cost (O3, O5)
reports/retrieval/orb-text/                           the harness's standard outputs (O3)
reports/qa/orb-text/                                  metrics, per-case parquet, report (O5)
```

## Review Focus

1. **Benchmark text reaches git or a report.** Every file under `data/fixtures/orb/` and
   `reports/` is checked for query, answer and section text by a test that fails on any string
   from the hydrated dataset. *Tested in O1 and O5.*
2. **The sample depends on arrival order.** Sorting before shuffling, as `assign_splits` does,
   makes the 400 ids a function of the seed and the ids only. A second machine produces the same
   ids. *Tested in O1.*
3. **A gold document resolves to the wrong paper.** The arXiv id is stored as a paper identifier
   at load time and the dataset builder maps `doc_id` through it, never through a title.
   Versioned ids (`2407.01528v3`) are kept as written. *Tested in O2.*
4. **An evidence id the model invented.** Structural validation rejects any id outside the
   evidence actually sent, allows one repair, and then the case is an abstention with
   `provenance_validity` 0 for that case, never a silent drop. *Tested in O4.*
5. **The judge is swayed by the answer's length or confidence.** The rubric scores agreement
   with the reference's core facts, extra correct detail is neutral, and the output schema writes
   the reason before the verdict. The 50 blind labels measure whether that held. *Tested in O5.*
6. **A run spends past its budget.** The preflight refuses an over-budget run, the ledger stops
   it mid-run, and a killed run keeps the cases it finished. *Tested in O5.*
7. **A frozen case is read while debugging.** The development set is a separate file; the QA
   runner refuses `--split frozen` without `--frozen`, and the report records how many times the
   frozen split has been scored per configuration. *Tested in O5.*

---

### Task O1: Fetch ORB at a pinned revision and freeze the text slice

**Depends on:** nothing. No model calls. Contacts Hugging Face once, offline.

**Files:**
- Create: `backend/src/copilot/evaluation/orb.py`, `data/fixtures/orb/{queries,qrels,splits}.jsonl`
- Modify: `configs/evaluation.yaml` (`benchmarks.orb`), `backend/src/copilot/cli.py`
  (`eval orb-fetch`, `eval orb-dataset`)
- Test: `backend/tests/unit/test_orb.py`

**Interfaces:**
- `ORB_REVISION = "63f6b052ff83508b08e242db42263ee708815c26"`, `ORB_REPO = "vectara/open_ragbench"`;
- `OrbPaths(root: Path)` with `.labels`, `.corpus`, `.pdfs`, under `${DATA_DIR}/benchmarks/orb/`;
- `fetch_orb(paths, *, revision: str = ORB_REVISION, token: str | None) -> dict[str, str]`,
  the four label files and the 1,000 corpus JSON files, refusing an unpinned revision and
  returning sha256 per file;
- `OrbQuery(query_id, query, type, source, doc_id, section_id, answer)`;
- `load_orb(paths) -> list[OrbQuery]`;
- `text_slice(queries) -> list[OrbQuery]`, `source == "text"`;
- `sample_qa(queries, *, seed: int = 42, frozen: int = 400, development: int = 100, per_document: int = 2) -> dict[str, str]`,
  query id to `"frozen" | "development" | "retrieval_only"`, stratified by `type`, sorted
  before shuffling, at most `per_document` sampled queries per `doc_id` across both sets;
- `write_orb_dataset(queries, assignment, directory, *, include_text: bool)`, the three
  files in the retrieval harness's shape: `family_id = doc_id`, `split` from the assignment,
  qrels with `doc_id` and `section_id`, grade 1;
- `hydrate_orb(dataset, paths)`, re-attaching query text and reference answers at run time.

- [ ] **Step 1: Write the failing unit tests** in `backend/tests/unit/test_orb.py`:

```python
from copilot.evaluation.orb import OrbQuery, sample_qa, text_slice


def _queries():
    out = []
    for doc in range(20):
        for i in range(6):
            kind = "extractive" if i % 2 else "abstractive"
            source = "text" if i < 5 else "text-image"
            out.append(OrbQuery(f"q{doc}-{i}", "t", kind, source, f"24{doc:02d}.00001v1", 0, "a"))
    return out


def test_text_slice_keeps_only_text_sourced_queries():
    assert {q.source for q in text_slice(_queries())} == {"text"}


def test_sampling_is_stratified_capped_and_order_independent():
    queries = text_slice(_queries())
    first = sample_qa(queries, seed=42, frozen=20, development=10, per_document=2)
    second = sample_qa(list(reversed(queries)), seed=42, frozen=20, development=10, per_document=2)
    assert first == second
    frozen = [q for q in queries if first[q.query_id] == "frozen"]
    dev = [q for q in queries if first[q.query_id] == "development"]
    assert len(frozen) == 20 and len(dev) == 10
    assert {q.query_id for q in frozen}.isdisjoint({q.query_id for q in dev})
    kinds = [q.type for q in frozen]
    assert abs(kinds.count("extractive") - kinds.count("abstractive")) <= 1
    per_doc = {}
    for q in frozen + dev:
        per_doc[q.doc_id] = per_doc.get(q.doc_id, 0) + 1
    assert max(per_doc.values()) <= 2


def test_the_repository_copy_carries_no_text(tmp_path):
    from copilot.evaluation.orb import write_orb_dataset
    queries = text_slice(_queries())
    write_orb_dataset(queries, sample_qa(queries), tmp_path, include_text=False)
    body = (tmp_path / "queries.jsonl").read_text()
    assert '"query"' not in body and '"answer"' not in body
```

- [ ] **Step 2: Run them and confirm they fail on the missing module.**
  Run: `uv run --project backend pytest backend/tests/unit/test_orb.py -q`
  Expected: `ModuleNotFoundError: No module named 'copilot.evaluation.orb'`.

- [ ] **Step 3: Implement `orb.py`.** Fetch with `huggingface_hub.hf_hub_download` at the pinned
  revision, the same pattern as `litsearch.py`; refuse anything but a 40-hex revision. Write
  checksums to `configs/evaluation.yaml` under `benchmarks.orb` with the reading date, licence
  `cc-by-nc-4.0`, `redistribution: local_use_only`, counts, and the text-slice counts from the
  facts table. Sampling: group the text slice by `type`, sort ids, shuffle with
  `random.Random(f"{seed}:{type}")`, walk each stratum round-robin taking a query only while its
  document is under the cap, fill `frozen` first then `development`; everything else is
  `retrieval_only`. Write the dataset files with the harness's record shapes so
  `eval validate-dataset` accepts them unchanged.

- [ ] **Step 4: Add the CLI.** `eval orb-fetch [--revision]` (needs `HF_TOKEN` only for rate
  limits) and `eval orb-dataset --out data/fixtures/orb` which writes the text-free copy and a
  hydratable copy under `${DATA_DIR}/benchmarks/orb/dataset/`.

- [ ] **Step 5: Run the fetch and the dataset build.** Record the file count, bytes, checksums
  and the final sample counts per `type`.

**Acceptance cases:** an unpinned revision is refused; checksums match the recorded ones on a
second fetch; `eval validate-dataset data/fixtures/orb` passes (families never span splits,
every query has a qrel); the repository files contain no query, answer or section text; the
sample is identical on two invocations; 400 frozen and 100 development with no document over
two.

**Suites:** `backend/tests/unit/test_orb.py`, `test_datasets.py`; Ruff; mypy strict.

**Commit:** `feat: pin Open RAG Bench and freeze its text-slice sample`

---

### Task O2: Load the ORB corpus through our pipeline and build the `orb-v1` release

**Depends on:** O1. Local GPU for the index build; no hosted calls.

**Files:**
- Create: `backend/src/copilot/evaluation/orb_load.py`
- Modify: `backend/src/copilot/cli.py` (`eval orb-load`), `configs/corpus.yaml` is **not**
  modified (arxiv.org is already an allowed host)
- Test: `backend/tests/integration/test_orb_load.py`

**Interfaces:**
- `orb_record(paper: Mapping) -> dict`, a record for `resolve_paper`: `source="orb"`,
  `source_revision=ORB_REVISION`, title, abstract, authors, `publication_year` from `published`,
  `pdf_url` from `pdf_urls.json`, `identifiers={"arxiv": <id with version>}`, no venue;
- `load_orb_corpus(paths, *, engine, parsing_config, transport, limit: int | None, progress) -> LoadReport`,
  per paper: resolve identity, download the PDF through `download_pdf` into
  `${DATA_DIR}/benchmarks/orb/pdfs/`, `parse_pdf_result`, `chunk_document`,
  `store_parsed_document`; resumable by `source_item_id`; keeps every PDF (this corpus is
  small and is re-read by O3's section mapping);
- `LoadReport(papers, parsed, failed: dict[str, int], chunks, chunk_tokens_p50, chunk_tokens_max)`.

- [ ] **Step 1: Write the failing integration test** in `backend/tests/integration/test_orb_load.py`,
  against the test database, with the repository fixture PDF served by an injected transport
  for two synthetic ORB papers:

```python
def test_two_orb_papers_load_with_arxiv_identifiers_and_chunks(migrated_database, tmp_path):
    report = load_orb_corpus(paths, engine=engine, parsing_config=config,
                             transport=fixture_transport, limit=None, progress=None)
    assert report.papers == 2 and report.parsed == 2 and report.failed == {}
    with session_scope(engine) as session:
        ids = session.execute(select(PaperIdentifier.value)
                              .where(PaperIdentifier.namespace == "arxiv")).scalars().all()
        assert sorted(ids) == ["2401.00001v2", "2401.00002v1"]
        assert session.execute(select(func.count(Chunk.id))).scalar_one() > 0
```

- [ ] **Step 2: Run it and confirm it fails on the missing module.**

- [ ] **Step 3: Implement the loader.** No job queue: a loop over `pdf_urls.json` with one
  transaction per paper, the same functions the worker's `resolve_record`, `download_pdf` and
  `parse_pdf` handlers call. A paper whose download or parse fails is recorded with its typed
  reason and the loop continues. The chunker and tokenizer come from `configs/parsing.yaml`, so
  `chunker_version` is `paragraph-pack-v1`.

- [ ] **Step 4: Run the real load** into `copilot_orb` (migrate it first with
  `alembic upgrade head` against that URL). Then, with the standard commands:
  `corpus export --run orb-v1 --out ${DATA_DIR}/exports/orb-v1`,
  `search build-index --manifest .../orb-v1/manifest.json --release orb-v1 --collections papers`,
  the same with `--collections chunks`, and `search validate-index --release orb-v1`.
  Do **not** activate.

- [ ] **Step 5: Assert the API refuses it.** With `QDRANT_COLLECTION_PREFIX` as in `.env`, a
  request for release `orb-v1` through the search API is `release_outside_namespace` or the
  release is simply not the active one; record which.

**Acceptance cases:** 1,000 records resolve to 1,000 papers with 0 identity conflicts (arXiv
ids are unique); every one of the 396 gold documents resolves through `paper_identifiers`; the
parse failure count and each failure's typed reason are recorded, with the PDF kept; chunk token
maximum is at most 1,200 and ordinals are contiguous (the M1 rebuild checks); the chunk
collection's digest equals the snapshot's; `validate-index` reports no problems; `orb-v1` is not
active after the task.

**Suites:** `test_orb_load.py`, `test_index_release.py`; Ruff; mypy strict.

**Commit:** `feat: load Open RAG Bench through the corpus pipeline into its own release`

---

### Task O3: Retrieval on all 1,914 text queries, with the gold section located

**Depends on:** O2. Idle GPU. No hosted calls.

**Files:**
- Create: `backend/src/copilot/evaluation/sections.py`, `configs/experiments/orb-text-retrieval.yaml`,
  `reports/retrieval/orb-text/`, `reports/orb-text.md` (first half)
- Modify: `backend/src/copilot/evaluation/retrieval.py` (record the top paper's best chunk id
  when candidates come through chunks; compute `section_hit@1`), `evaluation/report.py` (one
  column), `evaluation/datasets.py` only if a `section_id` qrel field needs carrying
- Test: `backend/tests/unit/test_sections.py`, `backend/tests/integration/test_eval_runner.py`
  (one case)

**Interfaces:**
- `locate_section(section_text: str, chunks: Sequence[ChunkRow]) -> set[UUID] | None`, the
  chunk ids whose text overlaps the ORB section, by the longest common normalized sentence
  run; `None` when nothing matches (counted as unmapped, never as a miss);
- `section_hit(best_chunk_id: UUID | None, gold_chunks: set[UUID] | None) -> bool | None`;
- the harness records per query `best_chunk_id` for the rank-1 paper under `candidates: chunks`
  and `section_hit@1` with its coverage (share of queries whose gold section mapped), the same
  rule as `judged_coverage`: no number without its coverage.

- [ ] **Step 1: Write the failing unit tests** for `locate_section` (exact section, section
  split across two chunks, section absent returns `None`, a one-sentence overlap is not enough)
  and `section_hit`.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement `sections.py`** over the ORB corpus JSON under `DATA_DIR` and the
  chunks of the loaded paper version. Normalize both sides with the lexical tokenizer; require
  at least three consecutive matching sentences or 40% of the section's sentences, whichever is
  smaller.

- [ ] **Step 4: Write the experiment config.** `dataset.path: data/fixtures/orb`,
  `query_text: benchmarks/orb/dataset/queries.jsonl`, `slice: all`, `labels: paper_id`,
  `corpus.release: orb-v1`, `papers: database`, the shared `configs/search.yaml`, metrics
  `recall_at: [10, 50]`, `ndcg_at: [10]`, `mrr_at: [5, 10]`, depth 50, bootstrap 1,000 / seed 42,
  variants `bm25`, `dense`, `hybrid`, `hybrid_rerank` each under `candidates: papers` and
  `candidates: chunks`, and **no `decision` block**. The run reads the whole text slice, so its
  split is a new value `retrieval_only` plus `development` plus `frozen`; the harness's split
  filter gains an `all_text` alias for this benchmark only, or the dataset is written with a
  single `retrieval` split and a separate QA assignment file. Choose the simpler and record it.

- [ ] **Step 5: Run on an idle GPU.** `eval retrieval --config configs/experiments/orb-text-retrieval.yaml --split all --out reports/retrieval/orb-text`,
  then `eval report` and `eval gaps`. A run that shared the GPU is set aside.

- [ ] **Step 6: Write the first half of `reports/orb-text.md`:** parse outcome from O2, the
  eight rows with intervals, MRR@5 beside haiku.rag's MAP rows with the judge-free caveat that
  their corpus text came from Docling and ours from pypdf, `section_hit@1` with its coverage,
  per-`type` facets, p50/p95, GPU state, and the read misses by `type`.

**Acceptance cases:** 1,914 queries scored, 0 failed; every manifest field the regression
rules require present; the `papers` and `chunks` candidate sources differ in their recorded
`candidate_source`; `section_hit@1` is reported with a coverage, and a query whose section did
not map is not counted as a miss; recorded reports re-render unchanged; `eval smoke`
reproducible.

**Suites:** `test_sections.py`, `test_eval_runner.py`, `test_regression.py`; Ruff; mypy strict.

**Commit:** `feat: measure retrieval on Open RAG Bench's text queries with gold sections`

---

### Task O4: Bounded evidence retrieval and a grounded generator with validated citations

**Depends on:** O3's release (for the integration case), L2's chat client and ledger. This is
P4.1 plus the generation adapter of P4.2, built here because the benchmark cannot run without
them; their interfaces are the M4 plan's.

**Files:**
- Create: `backend/src/copilot/evidence/{__init__,retrieve,citations}.py`,
  `backend/src/copilot/models/generation.py`, `configs/evidence.yaml`,
  `prompts/generate/grounded-claims-v1.yaml`
- Modify: `backend/src/copilot/contracts.py` only if `Evidence` or `Claim` need a field (for
  example `section_path` and `expanded: bool` on `Evidence`)
- Test: `backend/tests/unit/test_citations.py`, `backend/tests/unit/test_generation.py`,
  `backend/tests/integration/test_evidence.py`

**Interfaces** (P4.1 and P4.2 as written, plus the borrowed expansion):
- `retrieve_evidence(query: str, paper_ids: list[UUID], release_id: str, *, service, engine, config: EvidenceConfig) -> list[Evidence]`:
  chunk search inside the given papers, rerank at most 40 candidates, select at most 12 with at
  most 3 per paper, references excluded by default, within 6,000 generation-tokenizer tokens;
- `expand_to_section(evidence: Evidence, *, chunks: Sequence[ChunkRow], max_chars: int) -> Evidence`,
  haiku.rag's rule: if the hit's whole section fits `max_chars` and is at least 20% of it,
  return the section; if larger, grow outward from the hit, item by item, inside the section;
  never clip the hit's own text. Applied only under `evidence.expansion: section`;
- `build_context(evidence, token_budget, tokenizer) -> list[Evidence]`;
- `validate_citations(claims: list[dict], allowed_ids: set[str]) -> list[str]`, sorted unknown
  ids, plus the P4.1 rules: at least one evidence id per factual claim, unique references, any
  quoted span present in the evidence text;
- `Generator` (Protocol, existing): `answer(question, evidence, *, request_id) -> GroundedAnswer`;
- `ChatGenerator(client: ChatClient, model: LlmModel, prompt: GenerationPrompt)` on the L2
  client, structured output `{claims: [{text, evidence_ids}], insufficient_evidence: bool}`;
  on unknown ids or a schema failure, exactly one repair call that names the unknown ids; then
  `GroundedAnswer(claims=[], insufficient_evidence=True)` with `repair_attempted=True` and the
  reason;
- `FakeGenerator`, deterministic: cites the first evidence id per claim, and on a fixture
  question emits a fabricated id so the validator path is exercised in CI.

- [ ] **Step 1: Write the failing tests.** P4.1's `test_unknown_evidence_is_rejected` as
  written in the M4 plan; `test_expansion_returns_the_section_when_it_fits_and_grows_outward_when_not`;
  `test_a_fabricated_id_gets_one_repair_then_insufficiency` with the chat client's
  `MockTransport` returning a bad answer then a good one, and a second case returning two bad
  answers; `test_no_evidence_is_insufficient_without_a_model_call` (the ledger sees no
  reservation).

- [ ] **Step 2: Run them and confirm they fail on the missing modules.**

- [ ] **Step 3: Implement** in this order: `citations.py` (pure), `retrieve.py` over the search
  service's chunk retrieval restricted to `paper_ids` (a Qdrant filter on `paper_id`, the
  existing `evidence_filter`), `expand_to_section`, `generation.py`. The prompt treats evidence
  as data, numbers each evidence block by its id, and requires claims to carry ids. Every
  generation call reserves its worst case in the ledger with `purpose="evaluation"` and the
  run's id.

- [ ] **Step 4: Integration case** over `orb-v1`: three real ORB questions from the
  development set with their gold paper among the top 10, `FakeGenerator`, assert evidence
  counts, per-paper caps, the token budget in the model tokenizer, and that a cross-release
  evidence id is rejected.

- [ ] **Step 5: Two live calls** under $0.10, recorded as fixtures under
  `backend/tests/fixtures/llm/` with the same README discipline as L2.

**Acceptance cases:** P4.1's list for the public corpus: unknown ids, cross-release ids, absent
pages stay null, quotes not in source rejected, references excluded by default, one paper cannot
fill the context, the budget counts model tokens, no evidence returns insufficiency, paper-level
miss recorded against a global chunk baseline. Plus: one repair maximum; a refused or timed-out
generation is a typed failure, never an empty answer; expansion never drops the hit text; the
fake generator's fabricated id is caught.

**Suites:** `test_citations.py`, `test_generation.py`, `test_evidence.py`, `test_llm_client.py`;
Ruff; mypy strict; `eval smoke`.

**Commit:** `feat: retrieve bounded evidence and generate answers with validated citations`

---

### Task O5: The answering run: judge, pilot, calibration, frozen sample, report

**Depends on:** O1, O3, O4. **This task spends money**, about $1.50 in total.

**Files:**
- Create: `backend/src/copilot/evaluation/qa.py`, `backend/src/copilot/evaluation/judge.py`,
  `prompts/judge/answer-equivalence-v1.yaml`, `configs/experiments/orb-text-qa.yaml`,
  `data/fixtures/qa-smoke/{cases.jsonl,evidence.jsonl,judge-responses.json,expected.json}`,
  `reports/qa/README.md`, `reports/qa/orb-text/`, `docs/judge-rubric.md`
- Modify: `backend/src/copilot/cli.py` (`eval qa`, `eval pair`, `eval qa-smoke`),
  `reports/orb-text.md` (second half), `CLAUDE.md`, spec §11, benchmarks plan E6/E7 notes
- Test: `backend/tests/unit/test_judge.py`, `backend/tests/unit/test_pairing.py`,
  `backend/tests/integration/test_qa_runner.py`

**Interfaces:**
- `JudgeVerdict(reason: str, equivalent: bool)`, reason first in the schema on purpose;
- `AnswerEquivalenceJudge(client, model, prompt).judge(question, expected, answer) -> JudgeVerdict`,
  with the rubric adapted from haiku.rag's `ANSWER_EQUIVALENCE_RUBRIC` (MIT, attributed in the
  prompt file): equivalent when the answer carries the reference's core facts and conclusions,
  extra correct detail neutral, not equivalent when it contradicts, omits meaning-changing
  content or does not address the question; a malformed verdict is retried once, then recorded
  `judge_error` and excluded from the denominator with a count;
- `run_qa(config, split, out, *, max_spend_usd, frozen: bool = False, limit) -> dict`: for
  each case, top-10 papers by the shipped search, `retrieve_evidence`, `Generator.answer`,
  `validate_citations`, then the judge; per-case JSONL under
  `${DATA_DIR}/eval/qa/<run_id>/cases.jsonl` (question, answer, evidence ids, verdict, reason,
  cost, served models) and a text-free Parquet under `reports/`; refuses `split == "frozen"`
  without `frozen=True`; preflights the worst case and stops on the ledger like `eval retrieval`;
- metrics per variant: `accuracy` (judged correct / judged) with a Wilson interval and a
  family bootstrap by gold document, `cited_map`, `provenance_validity`, `abstention_rate`,
  `judge_errors`, cost, served models, prompt digests;
- `pair(treated: Path, baseline: Path) -> PairReport`: joined on `query_id`, exact McNemar on
  verdicts, sign test on `cited_map`, the smallest discordance split the exact test would
  reject, refusing files with no common case or a duplicate key;
- `calibration(labels: Path, cases: Path) -> dict`: Cohen's κ, raw agreement and the confusion
  matrix between hand labels and the judge over the same cases.

- [ ] **Step 1: Write the failing tests.** `test_judge_schema_puts_reason_before_verdict`;
  `test_a_malformed_verdict_is_retried_once_then_a_judge_error`; `test_mcnemar_exact_matches_a_hand_computed_case`
  (b = 8, c = 2 gives p = 0.109; b = 10, c = 1 gives p = 0.012); `test_pair_refuses_disjoint_files`;
  `test_the_frozen_split_needs_the_flag`; `test_qa_smoke_is_reproducible` over the synthetic
  fixture with recorded judge responses and `FakeGenerator`.

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement `judge.py`, `qa.py`, pairing and the smoke set.** The judge's cache
  key includes the prompt digest, so a rubric edit never serves an old verdict. The runner
  warms the search service and the generator once outside timing. Cost accounting reuses the
  L5 pattern: one ledger run per variant, `cost.metered_usd` in the manifest.

- [ ] **Step 4: Write `orb-text-qa.yaml`:** generator `openai-gpt-6-luna`, judge
  `openai-gpt-6-luna-low`, `evidence: configs/evidence.yaml`, variants `flat` (no expansion) and
  `section` (`evidence.expansion: section`), budget `max_spend_usd: 2.50`, and a pre-registered
  rule for the one comparison: `section` is preferred over `flat` only if McNemar's p < 0.05
  in its favour and `cited_map` does not fall by more than 0.03.

- [ ] **Step 5: Pilot, 20 development cases**, `--limit 20`. Record real cost per case and
  per call from `llm spend --run`; replace this plan's estimates in the report.

- [ ] **Step 6: Development passes**, 100 cases, until the prompts are settled: read every
  judged-wrong answer and every repair. Record how many passes were made. Commit the prompts
  before step 8.

- [ ] **Step 7: Calibration.** Hand-label 50 development answers blind, same rubric, before
  looking at the judge's verdicts for them. Compute κ. Below 0.6 the accuracy numbers are
  reported as uncalibrated and the rubric is fixed before the frozen run, not the labels.

- [ ] **Step 8: The frozen 400**, once per variant, `--frozen`, on an idle GPU. Then
  `eval pair` for `section` against `flat`.

- [ ] **Step 9: Write the second half of `reports/orb-text.md`:** commands, GPU state, costs
  from the ledger, served models, the two variants' metrics with intervals, the pairing, the
  calibration table, per-`type` facets, and five read failures with their category (retrieval
  miss, evidence present but answer wrong, judge disagreement, abstention). Update the tracker,
  CLAUDE.md, spec §11 and the E6/E7 notes.

**Acceptance cases:** a recorded-fixture smoke run reproduces byte-identical aggregates; the
frozen split cannot be scored without the flag and the report records each scoring; a run past
its budget stops and keeps finished cases; no question, answer or evidence text under
`reports/` or `data/`; every judge call's prompt digest is in the manifest; `cited_map` is
computed only on cases with a gold document (all of them here) and says so; κ is reported with
its label count; the pairing's p-value matches a hand calculation.

**Suites:** `test_judge.py`, `test_pairing.py`, `test_qa_runner.py`, `test_regression.py`;
Ruff; mypy strict; `eval smoke` and `eval qa-smoke`.

**Commit:** `feat: judge grounded answers on Open RAG Bench's frozen text sample`

## Exit checkpoint

Done when O1–O5 are committed with evidence, `reports/orb-text.md` carries retrieval over all
1,914 text queries and judged answering over the frozen 400 for both variants with cost,
calibration and the pairing, and spec §11, the benchmarks plan and the tracker say what shipped.
G4 is not claimed by this plan: its human-audited supported-claim rate needs E6's claim-level
faithfulness metric and E5's unanswerable cases, which ORB does not contain. What this plan
gives G4 is its provenance-validity evidence and the first calibrated answer accuracy. Next: the
image and table slices (1,131 queries) only after a picture-handling decision, since the text
pipeline cannot answer them; and P3.1.
