# Retrieval and Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve explainable search with measured BM25, dense, hybrid and reranked baselines.

**Architecture:** The evaluation harness freezes judgment splits before tuning. Versioned indexes serve a typed search service with controlled reranking and explicit fallbacks.

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

## Entry checkpoint and working conventions

Entry: G1 evidence exists, corpus snapshots validate and parse limitations are documented. Read spec §§5–7 and 11–12. No large-corpus scaling until pilot storage and throughput are measured.

Run commands from the repository root. Paths name future files. Each task contains a concrete regression example and critical implementation logic; finish the stated contracts and acceptance cases in the same task. These snippets are design artifacts, not tested application code. A task is typically several focused work sessions; each checkbox may be split into 2–5 minute edit/run actions while executing. No task is complete merely because its example test passes.

For each task: capture the expected behavioral failure, implement one behavior at a time, run the listed suite, inspect the diff, and record command/exit status, report path and commit in the progress tracker. Commit only that task's files after review. Keep external API tests opt-in and deterministic fixtures in normal CI. Missing dependencies are setup failures, not the intended red test.

## P2.1: Establish labeled retrieval data, metrics and a true BM25 baseline

**Depends on:** P1.5

**Files:**

- Create `backend/src/copilot/evaluation/{datasets,metrics}.py`, `backend/src/copilot/search/lexical.py`.
- Create `backend/tests/unit/test_metrics.py`, `backend/tests/unit/test_bm25.py`.
- Create `data/fixtures/retrieval/{queries,qrels,splits}.jsonl`, `configs/evaluation.yaml`, `docs/labeling-guide.md`.

**Interfaces:** `ndcg_at_k(ranked: list[str], grades: dict[str,int], k: int) -> float`; `recall_at_k(ranked, relevant, k) -> float`; `mrr_at_k(ranked, relevant, k) -> float`; `BM25.fit(documents: dict[str,str]) -> None`; `BM25.search(query: str, limit: int) -> list[tuple[str,float]]`. Split records carry query_id, family_id, workflow and split.

- [ ] Write the regression test below in `backend/tests/unit/test_metrics.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.evaluation.metrics import ndcg_at_k

def test_ndcg_obeys_grades_and_deduplicates():
    grades = {"a": 3, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b"], grades, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], grades, 2) < 1.0
    assert ndcg_at_k(["a", "a", "b"], grades, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {}, 10) == 0.0
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_metrics.py backend/tests/unit/test_bm25.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
import math

def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int) -> float:
    ranked = list(dict.fromkeys(ranked))[:k]
    gains = [grades.get(item, 0) for item in ranked]
    ideal = sorted(grades.values(), reverse=True)[:k]
    def dcg(values):
        return sum((2 ** grade - 1) / math.log2(i + 2)
                   for i, grade in enumerate(values))
    denominator = dcg(ideal)
    return dcg(gains) / denominator if denominator else 0.0
```

Implement BM25 using the exact tokenizer and parameters in spec §7. Oracle term contribution is `log(1+(N-df+0.5)/(df+0.5))*tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))`. Empty corpus/query returns empty; zero-length documents do not divide by zero. Score ties by paper ID. Use a tiny hand-calculated corpus to assert scores numerically. Metrics deduplicate IDs, validate k>0, reject invalid labels and separately report judged coverage. The zero-grade behavior above is the smoke oracle, not permission to hide unjudged production results. Create 30 initial genuine query tasks, record their author and source, pool candidates blindly, and freeze family assignments before tuning. Define a CLI `eval validate-dataset --path PATH` that rejects family overlap and missing source IDs.

- [ ] Verify the additional acceptance cases: ideal/reversed/empty rankings; repeated IDs; multiple relevant papers; unjudged coverage; leakage from paraphrase families rejected; BM25 rare-term and length normalization match hand calculations; same corpus and tokenizer produce deterministic scores.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_metrics.py backend/tests/unit/test_bm25.py -q` again, then run dataset validation on the fixture and pilot judgments. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m2-labels.md` with dataset revision, split counts, judgment rules and known limitations; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: establish retrieval metrics labels and bm25 baseline`.

## P2.2: Index paper and chunk vectors with atomic release switching

**Depends on:** P2.1

**Files:**

- Create `backend/src/copilot/models/embeddings.py`, `backend/src/copilot/search/{dense,index}.py`, `configs/models.yaml`.
- Modify `backend/src/copilot/corpus/releases.py` and `backend/src/copilot/cli.py`.
- Create `backend/tests/unit/test_vectors.py`, `backend/tests/integration/test_index_release.py`.

**Interfaces:** `validate_vectors(vectors: list[list[float]], dimensions: int) -> None`; `EmbeddingModel.encode` as spec §6; `build_index(manifest: Path, model_revision: str) -> str`; `activate_release(release_id: str) -> None`; `DenseRetriever.search(query: str, filters: PaperFilters, limit: int, release_id: str) -> list[tuple[str,float]]`.

- [ ] Write the regression test below in `backend/tests/unit/test_vectors.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.models.embeddings import validate_vectors

def test_incompatible_vectors_fail_before_upsert():
    validate_vectors([[1.0, 0.0]], dimensions=2)
    with pytest.raises(ValueError, match="dimension"):
        validate_vectors([[1.0]], dimensions=2)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_vectors([[float("nan"), 0.0]], dimensions=2)
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_vectors.py backend/tests/integration/test_index_release.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
import math

def validate_vectors(vectors: list[list[float]], dimensions: int) -> None:
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError("dimension_mismatch")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("nonfinite_vector")
```

Pin BGE-M3 model commit and tokenizer in models.yaml; implement a deterministic two-dimensional model for fixtures and a production adapter returning 1024 normalized floats. Persist preprocessing and text truncation statistics. Encode batches with retry-by-batch and checksum cache; changed model/checksum invalidates cache. Create paper/chunk collections with dense vectors, the spec's exact BM25 sparse weighting (no additional server IDF modifier), and indexed filter payloads. Validate sparse production ranking against the P2.1 oracle under equivalent settings; label any implementation difference. Enforce staging collection names and counts before switching the PostgreSQL pointer transactionally. Each request captures the collection pair once. CLI: `search build-index --manifest data/releases/pilot/manifest.json --models configs/models.yaml`; `corpus activate --release RELEASE_ID`. Require explicit release ID; do not auto-activate a partial build.

- [ ] Verify the additional acceptance cases: NaN/dimension mismatch rejected before writes; same input upserts idempotently; filtered query excludes wrong venue/year; query/document model revision mismatch rejected; crash between collection builds leaves previous release active; requests in flight keep their original pair; rollback restores both collections; restore yields matching canary rankings.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_vectors.py backend/tests/integration/test_index_release.py -q` again, then run all indexing integration tests with actual test Qdrant and build/activate the pilot index. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m2-index.md` with vector counts, memory/storage measurements, model revisions and rollback evidence; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: build reproducible retrieval indexes and release switching`.

## P2.3: Implement reproducible fusion and bounded cross-encoder reranking

**Depends on:** P2.2

**Files:**

- Create `backend/src/copilot/search/{fusion,rerank,service}.py`, `configs/search.yaml`.
- Create `backend/tests/unit/test_fusion.py`, `backend/tests/integration/test_search_service.py`.

**Interfaces:** `rrf(rankings: list[list[str]], k: int=60) -> list[tuple[str,float]]`; `Reranker.score` as spec §6; `SearchService.search(request: SearchRequest) -> SearchResponse`. The service hydrates canonical paper metadata separately and captures one active release for all branches.

- [ ] Write the regression test below in `backend/tests/unit/test_fusion.py` and add the additional acceptance cases listed for this task.

```python
import pytest
from copilot.search.fusion import rrf

def test_fusion_counts_each_list_once():
    result = dict(rrf([["a", "a", "b"], ["b", "c"]]))
    assert result["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert result["a"] == pytest.approx(1 / 61)
    assert rrf([["b", "a"], ["a", "b"]])[0][0] == "a"
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_fusion.py backend/tests/integration/test_search_service.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
from collections import defaultdict

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    if k < 1:
        raise ValueError("invalid_rrf_k")
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
```

Fetch 100 per branch with identical filters, fuse, rerank first 50 and return up to 20 by default. BM25/dense modes bypass fusion as specified; hybrid bypasses reranker. Reranker adapter uses pinned BAAI/bge-reranker-v2-m3 query/text scoring, evaluation mode, bounded batching and pair token budget. Reorder results by returned scores while preserving input-ID alignment. Store all stage scores and revision metadata for offline analysis, but don't expose them as probabilities. Deadline budgets: candidate branches 1 second each concurrently, reranker 1.5 seconds; total service budget 3 seconds before typed failure/fallback. Make test timeouts injected and deterministic, not sleeps. Only retry model errors if the total deadline permits.

- [ ] Verify the additional acceptance cases: duplicates/ties; identical branch filters; empty candidates; reranker reordering; mismatched score count/nonfinite values; one branch failure; both branch failure; timeout fallback explicitly degraded; no recall improvement claimed for reranker-only changes.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_fusion.py backend/tests/integration/test_search_service.py -q` again, then run complete search-service tests and one pinned-model pilot query set with stage timing. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m2-ranking.md` with trace examples and deterministic fallback tests; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add hybrid fusion and bounded cross-encoder reranking`.

## P2.4: Expose search, metadata, related papers and stable pagination

**Depends on:** P2.3

**Files:**

- Create `backend/src/copilot/search/{api,cache}.py`, modify `backend/src/copilot/app.py`.
- Create `backend/tests/integration/test_search_api.py`, `backend/tests/unit/test_cursor.py`.

**Interfaces:** HTTP endpoints from spec §10 for search/papers/related; `encode_cursor(request_id: str, offset: int, expires_at: int, secret: bytes) -> str`; `decode_cursor(token: str, secret: bytes, now: int) -> dict`. Cached ranking key includes query, filters, release and ranker revisions. Related papers exclude the seed and use dense neighbors initially.

- [ ] Write the regression test below in `backend/tests/integration/test_search_api.py` and add the additional acceptance cases listed for this task.

```python
def test_search_validation_and_metadata(api_client):
    invalid = api_client.post("/v1/search", json={"query": "", "mode": "hybrid"})
    assert invalid.status_code == 422
    response = api_client.post("/v1/search", json={
        "query": "contrastive learning", "mode": "hybrid",
        "filters": {"venues": [], "fulltext_only": False}, "limit": 2
    })
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) <= 2
    assert body["corpus_release_id"]
    assert all("paper_id" in item and "rank" in item for item in body["items"])
```

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_search_api.py backend/tests/unit/test_cursor.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
from fastapi import APIRouter, Request
from copilot.contracts import SearchRequest, SearchResponse

router = APIRouter(prefix="/v1")

@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request):
    return request.app.state.search_service.search(body)
```

Use the specification's `SearchResponse.papers: dict[str, PaperSummary]` metadata map. PaperSummary contains title/authors/venue/year/abstract/pdf_url/fulltext_indexed. Keep RankedPaper as ranking data. No ambiguous untyped third-party metadata. Configure test fixture to seed/activate synthetic papers before this suite. Use HMAC-SHA256 signed JSON cursors (URL-safe base64) and verify signatures constant-time; offsets reference cached top 200 ordering for ten minutes. Cache server-side in PostgreSQL initially. Unknown ID returns 404, merged ID resolves canonical metadata, expired cursor returns typed 410. Apply limit/year validation before serving. Include request IDs in errors and successes. Add public endpoints to OpenAPI and snapshot only structural contracts, not unstable autogenerated ordering.

- [ ] Verify the additional acceptance cases: invalid year interval; unknown/merged paper IDs; signed cursor tampering; expiry; stable page boundaries with no duplicates; release change retains cached result version until expiry; filters in related search; source URL unavailable state; test mode can't expose private indexes.

- [ ] Run `uv run --project backend pytest backend/tests/integration/test_search_api.py backend/tests/unit/test_cursor.py -q` again, then run all search API tests against migrated services and export OpenAPI via `uv run --project backend python -m copilot.cli api export-schema --out frontend/openapi.json`. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/m2-api.md` and the versioned OpenAPI schema; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: expose versioned paper search and related results`.

## P2.5: Run ablations and choose the first search configuration

**Depends on:** P2.4

**Files:**

- Create `backend/src/copilot/evaluation/{retrieval,report,regression}.py`, modify `backend/src/copilot/cli.py`.
- Create `backend/tests/unit/test_regression.py`, `configs/experiments/retrieval.yaml`, `reports/retrieval/README.md`.

**Interfaces:** `regressed(baseline: float, candidate: float, max_drop: float=0.03) -> bool`; `run_retrieval(config: Path, split: str, out: Path) -> dict`; CLI `eval retrieval --config configs/experiments/retrieval.yaml --split validation --out reports/retrieval/pilot`.

- [ ] Write the regression test below in `backend/tests/unit/test_regression.py` and add the additional acceptance cases listed for this task.

```python
from copilot.evaluation.regression import regressed

def test_absolute_threshold_and_boundary():
    assert regressed(0.70, 0.66)
    assert not regressed(0.70, 0.67)
    assert not regressed(0.01, 0.00)
    assert not regressed(0.60, 0.65)
```

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_regression.py -q`. Expected before implementation: failure because the required behavior or module is missing; confirm the specific failing assertion after imports resolve.

- [ ] Implement the contract using the following logic/configuration, then complete the behavior described immediately below it.

```python
def regressed(baseline: float, candidate: float, max_drop: float = 0.03) -> bool:
    if not 0 <= baseline <= 1 or not 0 <= candidate <= 1:
        raise ValueError("invalid_metric")
    return baseline - candidate > max_drop + 1e-12
```

Run BM25, dense, hybrid and hybrid+reranker on the same corpus and candidate pool. Then isolate embedding model and pair-token-budget changes one at a time. Compare BGE-M3 against BGE-small-en-v1.5 as a smaller cost/latency baseline, with separate dimensions/collections determined from pinned config; do not swap models into existing vectors. Capture per-query ranked IDs, relevant grades, candidate recall, judged coverage, p50/p95, failures, storage and inference cost. Bootstrap query-family differences 1,000 times with seed 42. Tune on development, choose on validation, run locked test only for a release decision. Reject a run missing any manifest version. Never require a predetermined improvement; retain simpler configurations if gains aren't supported. Define `eval smoke` with synthetic embeddings and frozen outputs for later CI.

- [ ] Verify the additional acceptance cases: threshold is absolute; missing manifest fields fail; incomparable corpus/split revisions refuse regression comparison; query failures stay in denominators; benchmark output reproducible under fixture seed; held-out split cannot be used as tuning input.

- [ ] Run `uv run --project backend pytest backend/tests/unit/test_regression.py -q` again, then run `uv run --project backend python -m copilot.cli eval smoke` and all four real baseline variants on the pilot validation set. Expected: all listed cases pass, with no skipped required integration checks.

- [ ] Save `reports/retrieval/pilot/report.md`, manifest JSON, per-query results and a promote/retain decision; update this task's status in the progress tracker; inspect `git diff --check` and `git diff`, stage only the Files listed here, and commit with message `feat: add reproducible retrieval ablations and regression reports`.

## Exit checkpoint

G2 requires all four comparable baseline reports, index rollback evidence and a usable API. Choose a configuration based on validation quality/latency; a negative experiment does not block progress. Record locked-test results once for this release. Next: Plan 3.

