"""The search service over a real release: fusion, reranking, deadlines and fallbacks."""

from __future__ import annotations

import threading
from uuid import uuid4

import pytest

from copilot.contracts import ListwiseResult, PaperFilters, SearchRequest
from copilot.corpus.releases import capture_release
from copilot.models.embeddings import FixtureEmbedding
from copilot.models.llm import LlmError
from copilot.search.dense import DenseRetriever
from copilot.search.fusion import rrf
from copilot.search.rerank import FixtureReranker, RerankError
from copilot.search.service import (
    TIMEOUT,
    Outcome,
    PaperStore,
    SearchConfig,
    SearchService,
    SearchUnavailable,
    ThreadedStageRunner,
)
from copilot.search.sparse import SparseRetriever

pytestmark = pytest.mark.integration

CONFIG = SearchConfig(candidates_per_branch=100, rrf_k=60, rerank_depth=50)
QUERY = "graph contrastive learning"


@pytest.fixture(scope="module")
def corpus(search_corpus):
    return search_corpus


class Recording:
    """A branch that records exactly what it was asked."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple] = []

    def search(self, query, filters, limit, release_id):
        self.calls.append((query, filters, limit, release_id))
        return self.inner.search(query, filters, limit, release_id)


class Failing:
    def search(self, query, filters, limit, release_id):
        raise ConnectionError("qdrant went away")


class Fixed:
    def __init__(self, hits) -> None:
        self.hits = hits

    def search(self, query, filters, limit, release_id):
        return list(self.hits)


class ScriptedRunner:
    """Deterministic deadlines: the named stages time out without anything waiting."""

    def __init__(self, timeouts=()) -> None:
        self.timeouts = set(timeouts)
        self.stages: list[str] = []

    def run(self, stages):
        outcomes = {}
        for name, (fn, budget) in stages.items():
            self.stages.append(name)
            if name in self.timeouts:
                outcomes[name] = Outcome(error="timeout", seconds=budget)
                continue
            try:
                outcomes[name] = Outcome(value=fn())
            except Exception as error:  # noqa: BLE001 - mirrors the threaded runner
                outcomes[name] = Outcome(error=getattr(error, "code", type(error).__name__))
        return outcomes


class Reversing:
    identity = "fixture/reversing"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        # Later fused candidates score higher, so reranking reverses the order.
        return [float(index) for index in range(len(texts))]


class Returning:
    identity = "fixture/returning"

    def __init__(self, make) -> None:
        self.make = make

    def score(self, query, texts):
        return self.make(texts)


class FailsOnce:
    identity = "fixture/fails-once"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA error: device-side assert")
        return FixtureReranker().score(query, texts)


def _service(corpus, **overrides) -> SearchService:
    parts = {
        "engine": corpus.engine,
        "lexical": SparseRetriever(corpus.engine, corpus.client),
        "dense": DenseRetriever(corpus.engine, corpus.client, FixtureEmbedding()),
        "reranker": FixtureReranker(),
        "config": CONFIG,
        "runner": ScriptedRunner(),
    }
    parts.update(overrides)
    return SearchService(**parts)


def _request(mode: str, **filters) -> SearchRequest:
    return SearchRequest(query=QUERY, mode=mode, filters=PaperFilters(**filters), limit=20)


def _ids(response) -> list[str]:
    return [str(item.paper_id) for item in response.items]


def test_hybrid_rerank_returns_ranked_papers_with_metadata_and_every_stage_score(corpus):
    service = _service(corpus, runner=ThreadedStageRunner())
    response, trace = service.search_with_trace(_request("hybrid_rerank"))

    assert response.corpus_release_id == corpus.release
    assert not response.degraded and response.warnings == []
    assert [item.rank for item in response.items] == list(range(1, len(response.items) + 1))
    scores = [item.score for item in response.items]
    assert scores == sorted(scores, reverse=True)
    top = response.items[0]
    assert set(top.scores) >= {"rrf", "rerank"}
    assert top.score == top.scores["rerank"]
    paper = response.papers[str(top.paper_id)]
    assert paper.authors == ["Ada Lovelace"] and paper.fulltext_indexed
    assert trace.models == {
        "embedding": FixtureEmbedding().identity,
        "reranker": FixtureReranker.identity,
    }
    assert set(trace.stages) >= {"lexical", "dense", "fusion", "rerank", "hydrate"}


def test_both_branches_receive_the_same_query_filters_and_release(corpus):
    lexical = Recording(SparseRetriever(corpus.engine, corpus.client))
    dense = Recording(DenseRetriever(corpus.engine, corpus.client, FixtureEmbedding()))
    filters = {"venues": ["ACL"], "year_from": 2024}
    response = _service(corpus, lexical=lexical, dense=dense).search(_request("hybrid", **filters))

    assert lexical.calls == dense.calls
    ((query, sent, limit, release),) = lexical.calls
    assert (query, limit, release) == (QUERY, 100, corpus.release)
    assert sent == PaperFilters(**filters)
    assert all(paper.venue == "ACL" and paper.year >= 2024 for paper in response.papers.values())


def test_single_branch_modes_bypass_fusion_and_the_reranker(corpus):
    reranker = Reversing()
    for mode, score_name in (("bm25", "bm25"), ("dense", "dense")):
        response, trace = _service(corpus, reranker=reranker).search_with_trace(_request(mode))
        assert "fusion" not in trace.stages and "rerank" not in trace.stages
        branch = "lexical" if mode == "bm25" else "dense"
        assert _ids(response) == [pid for pid, _ in trace.stages[branch]["candidates"]][:20]
        assert all(set(item.scores) == {score_name} for item in response.items)
    assert reranker.calls == 0


def test_hybrid_fuses_by_rrf_and_never_calls_the_reranker(corpus):
    reranker = Reversing()
    response, trace = _service(corpus, reranker=reranker).search_with_trace(_request("hybrid"))
    lexical = [pid for pid, _ in trace.stages["lexical"]["candidates"]]
    dense = [pid for pid, _ in trace.stages["dense"]["candidates"]]
    assert _ids(response) == [pid for pid, _ in rrf([lexical, dense], k=60)][:20]
    assert reranker.calls == 0


def test_no_candidates_is_an_empty_page_not_an_error(corpus):
    empty = Fixed([])
    response = _service(corpus, lexical=empty, dense=empty).search(_request("hybrid_rerank"))
    assert response.items == [] and response.papers == {}
    assert not response.degraded


def test_the_reranker_reorders_the_fused_set_without_changing_it(corpus):
    """Candidate recall is the fused set's: a reranker-only change cannot improve it."""

    service = _service(corpus, reranker=Reversing())
    reranked, trace = service.search_with_trace(_request("hybrid_rerank"))
    fused = [pid for pid, _ in trace.stages["fusion"]["candidates"]][: CONFIG.rerank_depth]
    assert _ids(reranked) == list(reversed(fused))[:20]
    assert set(_ids(reranked)) <= set(fused)
    plain = _service(corpus).search(_request("hybrid"))
    assert set(_ids(reranked)) == set(_ids(plain))


@pytest.mark.parametrize(
    ("make", "warning"),
    [
        (lambda texts: [1.0] * (len(texts) - 1), "rerank_invalid_scores"),
        (lambda texts: [float("nan")] * len(texts), "rerank_invalid_scores"),
    ],
)
def test_unusable_reranker_scores_leave_the_fusion_order_and_say_so(corpus, make, warning):
    response, trace = _service(corpus, reranker=Returning(make)).search_with_trace(
        _request("hybrid_rerank")
    )
    fused = [pid for pid, _ in trace.stages["fusion"]["candidates"]]
    assert _ids(response) == fused[:20]
    assert response.degraded and response.warnings == [warning]
    assert all("rerank" not in item.scores for item in response.items)


def test_one_failed_branch_serves_the_other_with_a_warning(corpus):
    response, trace = _service(corpus, dense=Failing()).search_with_trace(_request("hybrid"))
    lexical = [pid for pid, _ in trace.stages["lexical"]["candidates"]]
    assert _ids(response) == lexical[:20]
    assert response.degraded and response.warnings == ["dense_unavailable"]
    assert trace.stages["dense"]["error"] == "ConnectionError"


def test_both_branches_failing_is_a_retryable_error_not_an_empty_page(corpus):
    with pytest.raises(SearchUnavailable) as raised:
        _service(corpus, lexical=Failing(), dense=Failing()).search(_request("hybrid_rerank"))
    assert raised.value.code == "candidates_unavailable" and raised.value.retryable
    with pytest.raises(SearchUnavailable, match="candidates_unavailable"):
        _service(corpus, lexical=Failing()).search(_request("bm25"))


def test_timeouts_fall_back_explicitly(corpus):
    branch = _service(corpus, runner=ScriptedRunner({"dense"})).search(_request("hybrid"))
    assert branch.degraded and branch.warnings == ["dense_timeout"]
    assert branch.items

    service = _service(corpus, runner=ScriptedRunner({"rerank"}))
    response, trace = service.search_with_trace(_request("hybrid_rerank"))
    fused = [pid for pid, _ in trace.stages["fusion"]["candidates"]]
    assert _ids(response) == fused[:20]
    assert response.degraded and response.warnings == ["rerank_timeout"]
    # A timeout spent its whole budget, so it is not retried.
    assert trace.stages["rerank"]["errors"] == ["timeout"]


def test_a_model_error_is_retried_once_within_the_deadline(corpus):
    reranker = FailsOnce()
    response, trace = _service(corpus, reranker=reranker).search_with_trace(
        _request("hybrid_rerank")
    )
    assert reranker.calls == 2
    assert not response.degraded
    assert trace.stages["rerank"]["attempts"] == 2


def test_an_exhausted_total_deadline_skips_the_reranker(corpus):
    # The request starts at 0 and its branches are budgeted at 0; by the time
    # the reranker is due, the clock reads past the 3-second total.
    ticks = iter([0.0, 0.0])
    service = _service(corpus, clock=lambda: next(ticks, 99.0))
    response = service.search(_request("hybrid_rerank"))
    assert response.degraded and response.warnings == ["rerank_timeout"]


def test_the_release_is_read_once_per_request(corpus):
    reads: list[int] = []

    def capture():
        reads.append(1)
        return capture_release(corpus.engine)

    lexical = Recording(SparseRetriever(corpus.engine, corpus.client))
    dense = Recording(DenseRetriever(corpus.engine, corpus.client, FixtureEmbedding()))
    _service(corpus, capture=capture, lexical=lexical, dense=dense).search(
        _request("hybrid_rerank")
    )
    assert len(reads) == 1
    assert lexical.calls[0][3] == dense.calls[0][3] == corpus.release


def test_no_active_release_is_a_retryable_error(corpus):
    with pytest.raises(SearchUnavailable, match="corpus_not_ready"):
        _service(corpus, capture=lambda: None).search(_request("hybrid"))


def test_the_reranker_error_type_is_what_the_trace_records(corpus):
    """A typed rerank error keeps its code; anything else is recorded by type."""

    def raise_typed(texts):
        raise RerankError("score_count_mismatch")

    _, trace = _service(corpus, reranker=Returning(raise_typed)).search_with_trace(
        _request("hybrid_rerank")
    )
    assert trace.stages["rerank"]["errors"] == ["score_count_mismatch"] * 2


def test_a_candidate_missing_from_the_database_is_dropped_and_reported(corpus):
    ghost = str(uuid4())
    lexical = Fixed([(ghost, 9.0), (corpus.ids["Graph transformers at scale"], 1.0)])
    for mode in ("hybrid", "hybrid_rerank"):
        response = _service(corpus, lexical=lexical).search(_request(mode))
        assert ghost not in _ids(response)
        assert "metadata_missing" in response.warnings and response.degraded


def test_abandoned_model_work_cannot_starve_the_lexical_branch():
    """Overrunning reranks keep their threads; BM25 must still find one of its own.

    Found by the pilot on a contended GPU: every rerank overran, kept running,
    and a shared pool would fill with them until even BM25 could not start.
    """

    held = threading.Event()
    runner = ThreadedStageRunner()
    try:
        for _ in range(12):
            assert runner.run({"rerank": (held.wait, 0.01)})["rerank"].error == TIMEOUT
        outcome = runner.run({"lexical": (lambda: ["a"], 5.0)})["lexical"]
        assert outcome.error is None and outcome.value == ["a"]
    finally:
        held.set()
        runner.close()


def test_a_stage_still_queued_at_its_deadline_never_runs():
    held = threading.Event()
    ran: list[int] = []
    runner = ThreadedStageRunner()
    try:
        assert runner.run({"dense": (held.wait, 0.01)})["dense"].error == TIMEOUT
        # Queued behind the overrunning call: its answer would arrive too late to use.
        assert runner.run({"dense": (lambda: ran.append(1), 0.01)})["dense"].error == TIMEOUT
        held.set()
        assert runner.run({"dense": (lambda: "next", 5.0)})["dense"].value == "next"
        assert ran == []
    finally:
        held.set()
        runner.close()


def test_hydrate_time_includes_the_metadata_the_reranker_reads(corpus):
    """The reranker's texts come from the metadata load; the trace must not lose it."""

    now = [0.0]

    class TimedStore(PaperStore):
        def load(self, paper_ids):
            if paper_ids:
                now[0] += 0.25
            return super().load(paper_ids)

    for mode in ("hybrid", "hybrid_rerank"):
        now[0] = 0.0
        service = _service(corpus, papers=TimedStore(corpus.engine), clock=lambda: now[0])
        _, trace = service.search_with_trace(_request(mode))
        # One load each: hybrid_rerank's page is drawn from the head it already read.
        assert trace.stages["hydrate"]["seconds"] == pytest.approx(0.25)


# --- hybrid_rerank_llm (LLM reranking plan, L4) --------------------------------------------


class FakeListwise:
    """A listwise reranker that reverses the head, or fails the way it is told to."""

    identity = "fake/listwise@v1"

    def __init__(self, reverse: bool = False, fail: str | None = None) -> None:
        self.reverse = reverse
        self.fail = fail
        self.calls = 0
        self.budgets: list[float] = []

    def order(self, query, texts, *, timeout, purpose, request_id=None, run_id=None):
        self.calls += 1
        self.budgets.append(timeout)
        if self.fail == "spend_cap":
            raise LlmError("llm_spend_cap")
        if self.fail == "unparseable":
            raise RerankError("llm_rerank_unparseable")
        if self.fail == "server_error":
            raise LlmError("llm_server_error", retryable=True)
        order = list(range(len(texts)))
        return ListwiseResult(order=order[::-1] if self.reverse else order,
                              ranked_by_model=len(texts), cost_usd=0.0012,
                              served_model="fake-model", cached=False)


@pytest.fixture
def service_factory(corpus):
    def make(listwise):
        timeouts = {"llm"} if listwise is not None and listwise.fail == "timeout" else set()
        return _service(corpus, listwise=listwise, runner=ScriptedRunner(timeouts))

    return make


def _deep(limit: int = 50) -> SearchRequest:
    return SearchRequest(query=QUERY, mode="hybrid_rerank_llm", filters=PaperFilters(),
                         limit=limit)


def test_the_llm_reorders_the_reranked_head_without_changing_its_members(service_factory):
    service = service_factory(listwise=FakeListwise(reverse=True))
    request = _deep()
    plain = service.rank(request.model_copy(update={"mode": "hybrid_rerank"}))
    deep = service.rank(request)
    assert [pid for pid, _ in deep.ordering.items] == [pid for pid, _ in plain.ordering.items][::-1]
    assert not deep.ordering.warnings
    assert all("llm_rank" in deep.ordering.scores[pid] for pid, _ in deep.ordering.items)
    ranks = [deep.ordering.scores[pid]["llm_rank"] for pid, _ in deep.ordering.items]
    assert ranks == list(range(1, len(ranks) + 1))
    stage = deep.trace.stages["llm"]
    assert stage["ranked_by_model"] == len(ranks) and stage["cost_usd"] == 0.0012
    assert stage["served_model"] == "fake-model" and stage["cached"] is False


@pytest.mark.parametrize("failure, warning", [
    ("spend_cap", "llm_spend_cap"),
    ("unparseable", "llm_rerank_unparseable"),
    ("timeout", "llm_rerank_timeout"),
    ("server_error", "llm_rerank_failed"),
])
def test_an_llm_failure_serves_the_cross_encoder_order_and_says_why(
    service_factory, failure, warning
):
    service = service_factory(listwise=FakeListwise(fail=failure))
    request = _deep()
    plain = service.rank(request.model_copy(update={"mode": "hybrid_rerank"}))
    deep = service.rank(request)
    assert deep.ordering.items == plain.ordering.items
    assert warning in deep.ordering.warnings


def test_without_a_configured_llm_the_mode_degrades_rather_than_failing(service_factory):
    deep = service_factory(listwise=None).rank(
        SearchRequest(query="graph", mode="hybrid_rerank_llm", filters=PaperFilters()))
    assert "llm_rerank_unavailable" in deep.ordering.warnings and deep.ordering.items


def test_the_llm_has_its_own_budget_beyond_the_three_second_total(service_factory):
    listwise = FakeListwise(reverse=True)
    service_factory(listwise=listwise).rank(_deep())
    assert CONFIG.total_seconds < listwise.budgets[0] <= CONFIG.llm_rerank_seconds


def test_warm_up_never_calls_the_llm(corpus, service_factory):
    listwise = FakeListwise()
    service_factory(listwise=listwise).warm(capture_release(corpus.engine))
    assert listwise.calls == 0


def test_the_llm_mode_has_its_own_cache_identity(service_factory):
    service = service_factory(listwise=FakeListwise())
    assert service.identity("hybrid_rerank_llm")["listwise"] == "fake/listwise@v1"
    assert "listwise" not in service.identity("hybrid_rerank")
    plain = service.identity("hybrid_rerank")
    assert {key: service.identity("hybrid_rerank_llm")[key] for key in plain} == plain
