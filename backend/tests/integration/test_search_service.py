"""The search service over a real release: fusion, reranking, deadlines and fallbacks."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import Engine, text

from copilot.contracts import PaperFilters, SearchRequest
from copilot.corpus.export import export_snapshot
from copilot.corpus.releases import activate_release, capture_release
from copilot.db.session import assert_safe_test_database, cleanup_test_vectors
from copilot.models.embeddings import FixtureEmbedding
from copilot.search.dense import DenseRetriever
from copilot.search.fusion import rrf
from copilot.search.index import (
    CHUNKS,
    PAPERS,
    CollectionSettings,
    IndexConfig,
    build_index,
    index_validator,
)
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

RELEASE = "search-a"
TABLES = (
    "chunks",
    "paper_versions",
    "paper_identifiers",
    "paper_authors",
    "authors",
    "papers",
    "venues",
)
# (venue, year, title, abstract)
CORPUS = (
    ("ICLR", 2024, "Graph neural networks for molecules", "Message passing on molecular graphs."),
    ("ICLR", 2023, "Contrastive learning of visual representations", "Augmented views agree."),
    ("ACL", 2024, "Graph contrastive learning for text", "Word graphs, contrasted."),
    ("ACL", 2023, "Retrieval augmented generation", "Retrieve passages, then generate."),
    ("NeurIPS", 2024, "Diffusion models for image synthesis", "Denoising generates images."),
    ("NeurIPS", 2023, "Scaling laws for language models", "Loss falls as a power law."),
    ("ICLR", 2024, "Graph transformers at scale", "Attention over graph nodes."),
    ("ACL", 2024, "Contrastive sentence embeddings", "Sentence pairs, contrasted."),
)
CONFIG = SearchConfig(candidates_per_branch=100, rrf_k=60, rerank_depth=50)
QUERY = "graph contrastive learning"


@dataclass
class Corpus:
    engine: Engine
    client: QdrantClient
    ids: dict[str, str]


def _seed(engine: Engine) -> dict[str, str]:
    ids: dict[str, str] = {}
    with engine.begin() as connection:
        author = connection.execute(
            text(
                "insert into authors (id, name, normalized_name) values"
                " (gen_random_uuid(), 'Ada Lovelace', 'ada lovelace') returning id"
            )
        ).scalar_one()
        for venue, year, title, abstract in CORPUS:
            paper_id = str(uuid4())
            ids[title] = paper_id
            venue_id = connection.execute(
                text(
                    "insert into venues (id, name, track) values (gen_random_uuid(), :name,"
                    " 'main') on conflict (name, track) do update set name = excluded.name"
                    " returning id"
                ),
                {"name": venue},
            ).scalar_one()
            connection.execute(
                text(
                    "insert into papers (id, title, abstract, publication_year, venue_id,"
                    " first_seen_at, acceptance_type, metadata_status) values (:id, :title,"
                    " :abstract, :year, :venue, now(), 'poster', 'active')"
                ),
                {
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "venue": venue_id,
                },
            )
            connection.execute(
                text(
                    "insert into paper_authors (paper_id, position, author_id) values"
                    " (:paper, 0, :author)"
                ),
                {"paper": paper_id, "author": author},
            )
            version = connection.execute(
                text(
                    "insert into paper_versions (id, paper_id, source, source_url,"
                    " source_revision, content_sha256, redistribution, parser_version,"
                    " parse_status) values (gen_random_uuid(), :paper, 'papercli', 'u', 'rev',"
                    " :sha, 'unknown', 'pypdf-text-v2', 'parsed') returning id"
                ),
                {"paper": paper_id, "sha": paper_id.replace("-", "")[:32] * 2},
            ).scalar_one()
            connection.execute(
                text(
                    "insert into chunks (id, paper_version_id, section_path, section_ordinal,"
                    " kind, ordinal, text, token_count, parser_version, chunker_version) values"
                    " (gen_random_uuid(), :version, 'Method', 0, 'body', 0, :text, 8,"
                    " 'pypdf-text-v2', 'paragraph-pack-v1')"
                ),
                {"version": version, "text": abstract},
            )
    return ids


@pytest.fixture(scope="module")
def corpus(migrated_database, test_settings, tmp_path_factory) -> Iterator[Corpus]:
    assert_safe_test_database(test_settings)
    client = QdrantClient(url=test_settings.qdrant_url, timeout=60)
    with migrated_database.begin() as connection:
        connection.execute(text("delete from active_release"))
        connection.execute(text("delete from corpus_releases"))
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
    cleanup_test_vectors(test_settings, client)
    ids = _seed(migrated_database)
    root = tmp_path_factory.mktemp("search")
    snapshot = root / "exports" / RELEASE
    export_snapshot(RELEASE, snapshot, engine=migrated_database)
    model = FixtureEmbedding()
    build_index(
        Path(snapshot / "manifest.json"),
        model,
        engine=migrated_database,
        client=client,
        prefix=test_settings.qdrant_collection_prefix,
        data_dir=root / "data",
        config=IndexConfig(
            collections={
                PAPERS: CollectionSettings(max_tokens=64),
                CHUNKS: CollectionSettings(max_tokens=64),
            },
            batch_points=4,
            canary_queries=2,
            canary_limit=3,
            canary_terms=3,
        ),
        release_id=RELEASE,
    )
    activate_release(
        migrated_database,
        RELEASE,
        validator=index_validator(client, model_identity=model.identity, dimensions=2),
    )
    try:
        yield Corpus(migrated_database, client, ids)
    finally:
        client.close()


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


def _service(corpus: Corpus, **overrides) -> SearchService:
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

    assert response.corpus_release_id == RELEASE
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
    assert (query, limit, release) == (QUERY, 100, RELEASE)
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
    assert lexical.calls[0][3] == dense.calls[0][3] == RELEASE


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
