"""The search service: one release, two candidate branches, fusion, bounded reranking.

A request reads the active release once and hands that release to every stage,
so the lexical and dense branches can never answer from different collection
pairs. The branches run concurrently with identical filters; hybrid modes fuse
them by RRF, and ``hybrid_rerank`` reorders at most 50 fused candidates with the
cross-encoder. Every stage has a deadline, and missing one degrades the answer
explicitly rather than failing it (spec §7):

* a reranker that times out, errors or returns unusable scores leaves the RRF
  order, with a warning;
* one failed branch leaves the other branch's order, with a warning;
* both failing is a typed, retryable error — never an empty page that looks
  like "no results".

``hybrid_rerank_llm`` is opt-in deep search: after the cross-encoder, a hosted
LLM reorders the reranked head listwise, under its own deadline. It can only
reorder; any failure keeps the cross-encoder's order, with a warning that says
why, and without a configured LLM the mode degrades rather than failing.

Stage timings, per-stage candidates and scores, and the model revisions go into
a ``SearchTrace`` kept beside the response for offline analysis; the scores are
ranking signals and the response never presents them as probabilities.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

import yaml
from sqlalchemy import Engine, text

from ..contracts import (
    ListwiseResult,
    PaperFilters,
    PaperSummary,
    RankedPaper,
    Reranker,
    SearchRequest,
    SearchResponse,
)
from ..corpus.releases import ReleaseRecord, capture_release
from .fusion import RRF_K, rrf
from .lexical import paper_text
from .rerank import rerank_order, validate_scores

TIMEOUT = "timeout"


class SearchUnavailable(RuntimeError):
    """No answer can be given; ``retryable`` says whether trying again may help."""

    def __init__(self, code: str, detail: str = "", *, retryable: bool = True) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class SearchConfig:
    candidates_per_branch: int = 100
    # Where both branches find candidates (P2.6, decided 2026-10-06): the paper
    # collection, the chunk collection with each paper scored by its best evidence
    # chunk, or each branch's RRF of both.
    candidates_source: str = "papers"
    rrf_k: int = RRF_K
    rerank_depth: int = 50
    branch_seconds: float = 1.0
    rerank_seconds: float = 1.5
    total_seconds: float = 3.0
    rerank_retries: int = 1
    cache_depth: int = 200
    cache_ttl_seconds: int = 600
    # Deep search only: the LLM's own budget, and the whole request's.
    llm_rerank_seconds: float = 20.0
    llm_total_seconds: float = 25.0


CANDIDATE_SOURCES = ("papers", "chunks", "both")


def load_search_config(path: str | Path) -> SearchConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    deadlines = raw.get("deadlines_seconds") or {}
    source = str((raw.get("candidates") or {}).get("source", "papers"))
    if source not in CANDIDATE_SOURCES:
        raise ValueError(f"candidates.source must be one of {', '.join(CANDIDATE_SOURCES)}")
    return SearchConfig(
        candidates_per_branch=int((raw.get("candidates") or {}).get("per_branch", 100)),
        candidates_source=source,
        rrf_k=int((raw.get("fusion") or {}).get("rrf_k", RRF_K)),
        rerank_depth=int((raw.get("rerank") or {}).get("depth", 50)),
        branch_seconds=float(deadlines.get("branch", 1.0)),
        rerank_seconds=float(deadlines.get("rerank", 1.5)),
        total_seconds=float(deadlines.get("total", 3.0)),
        rerank_retries=int(raw.get("rerank_retries", 1)),
        cache_depth=int((raw.get("cache") or {}).get("depth", 200)),
        cache_ttl_seconds=int((raw.get("cache") or {}).get("ttl_seconds", 600)),
        llm_rerank_seconds=float(deadlines.get("llm_rerank", 20.0)),
        llm_total_seconds=float(deadlines.get("llm_total", 25.0)),
    )


class CandidateRetriever(Protocol):
    """Both branches: a release id in, (paper id, branch score) pairs out."""

    def search(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> list[tuple[str, float]]: ...


class ListwiseReranker(Protocol):
    """A hosted LLM that orders a head of candidates; it never adds or drops one."""

    identity: str

    def order(
        self,
        query: str,
        texts: Sequence[str],
        *,
        timeout: float,
        purpose: str,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> ListwiseResult: ...


@dataclass
class Outcome:
    """What one stage produced within its budget: a value, or why not."""

    value: Any = None
    error: str | None = None
    seconds: float = 0.0


Stage = tuple[Callable[[], Any], float]


class StageRunner(Protocol):
    """Runs stages concurrently, each against its own deadline."""

    def run(self, stages: Mapping[str, Stage]) -> dict[str, Outcome]: ...


class ThreadedStageRunner:
    """Stages on per-kind thread lanes; a stage past its deadline is reported, not awaited.

    Python cannot cancel a running thread, so a stage that overruns keeps
    running in the background and its result is discarded. Each stage kind has
    its own lane, so overrunning model calls hold only their own lane's threads,
    never the ones the lexical branch needs. A model lane has one worker, so
    abandoned calls wait their turn rather than piling onto the device, and a
    stage still queued when its deadline passes is cancelled without running:
    its answer could no longer be used.
    """

    # The LLM is remote: concurrent deep searches must not queue behind one another.
    LANES: Mapping[str, int] = {"lexical": 4, "dense": 1, "rerank": 1, "llm": 8}

    def __init__(self, lanes: Mapping[str, int] | None = None) -> None:
        self._sizes = dict(self.LANES if lanes is None else lanes)
        self._lanes: dict[str, ThreadPoolExecutor] = {}
        self._lock = threading.Lock()

    def _lane(self, name: str) -> ThreadPoolExecutor:
        with self._lock:
            lane = self._lanes.get(name)
            if lane is None:
                lane = ThreadPoolExecutor(
                    max_workers=self._sizes.get(name, 1), thread_name_prefix=f"search-{name}"
                )
                self._lanes[name] = lane
            return lane

    def run(self, stages: Mapping[str, Stage]) -> dict[str, Outcome]:
        started = time.monotonic()
        futures = {name: self._lane(name).submit(fn) for name, (fn, _) in stages.items()}
        outcomes: dict[str, Outcome] = {}
        for name, future in futures.items():
            budget = stages[name][1]
            remaining = max(0.0, budget - (time.monotonic() - started))
            try:
                value = future.result(timeout=remaining)
            except FutureTimeout:
                # Succeeds only while the stage is still queued behind its lane.
                future.cancel()
                outcomes[name] = Outcome(error=TIMEOUT, seconds=budget)
                continue
            except Exception as error:  # noqa: BLE001 - every stage failure is typed below
                outcomes[name] = Outcome(
                    error=getattr(error, "code", type(error).__name__),
                    seconds=time.monotonic() - started,
                )
                continue
            outcomes[name] = Outcome(value=value, seconds=time.monotonic() - started)
        return outcomes

    def close(self) -> None:
        with self._lock:
            for lane in self._lanes.values():
                lane.shutdown(wait=False, cancel_futures=True)
            self._lanes.clear()


@dataclass(frozen=True)
class PaperRow:
    paper_id: str
    title: str
    abstract: str | None
    authors: tuple[str, ...]
    venue: str
    year: int
    pdf_url: str | None
    fulltext: bool

    @property
    def text(self) -> str:
        """The canonical paper text the reranker reads (spec §7)."""

        return paper_text(self.title, self.abstract)

    def summary(self) -> PaperSummary:
        return PaperSummary(
            title=self.title,
            authors=list(self.authors),
            venue=self.venue,
            year=self.year,
            abstract=self.abstract,
            pdf_url=self.pdf_url,
            fulltext_indexed=self.fulltext,
        )


_PAPER_ROWS = """
select p.id::text as paper_id,
       p.title,
       p.abstract,
       coalesce(p.publication_year, 0) as year,
       coalesce(v.name, '') as venue,
       p.pdf_url,
       exists (
           select 1 from paper_versions pv
           where pv.paper_id = p.id and pv.parse_status = 'parsed'
       ) as fulltext,
       coalesce((
           select array_agg(a.name order by pa.position)
           from paper_authors pa join authors a on a.id = pa.author_id
           where pa.paper_id = p.id
       ), '{}') as authors
from papers p
left join venues v on v.id = p.venue_id
where p.id = any(cast(:ids as uuid[]))
"""


class PaperStore:
    """Canonical paper metadata, hydrated separately from ranking (plan P2.3)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def load(self, paper_ids: Sequence[str]) -> dict[str, PaperRow]:
        if not paper_ids:
            return {}
        with self._engine.connect() as connection:
            rows = connection.execute(text(_PAPER_ROWS), {"ids": list(paper_ids)}).mappings()
            return {
                str(row["paper_id"]): PaperRow(
                    paper_id=str(row["paper_id"]),
                    title=str(row["title"]),
                    abstract=row["abstract"],
                    authors=tuple(str(name) for name in row["authors"]),
                    venue=str(row["venue"]),
                    year=int(row["year"]),
                    pdf_url=row["pdf_url"],
                    fulltext=bool(row["fulltext"]),
                )
                for row in rows
            }


class PaperSource(Protocol):
    """Canonical metadata by paper id: PostgreSQL in service, a fixture in the smoke set."""

    def load(self, paper_ids: Sequence[str]) -> dict[str, PaperRow]: ...


@dataclass
class SearchTrace:
    """Everything a stage decided, for offline analysis; never shown as probabilities."""

    request_id: str
    release_id: str
    mode: str
    models: dict[str, str]
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def degraded(self) -> bool:
        return bool(self.warnings)


@dataclass(frozen=True)
class Ordering:
    """One request's whole ranking over one release: what a cursor pages through.

    ``items`` is at most the configured cache depth (200), and for
    ``hybrid_rerank`` only the reranked head. ``scores`` holds every stage score
    each item earned; ``warnings`` says how the ordering was degraded, if at all.
    """

    release_id: str
    items: tuple[tuple[str, float], ...]
    scores: Mapping[str, Mapping[str, float]]
    warnings: tuple[str, ...]


@dataclass
class Ranking:
    """A fresh ordering, the trace of how it was made and the metadata it already read."""

    request_id: UUID
    ordering: Ordering
    trace: SearchTrace
    rows: dict[str, PaperRow]


_BRANCH_SCORE = {"lexical": "bm25", "dense": "dense"}
_MODE_BRANCHES = {
    "bm25": ("lexical",),
    "dense": ("dense",),
    "hybrid": ("lexical", "dense"),
    "hybrid_rerank": ("lexical", "dense"),
    "hybrid_rerank_llm": ("lexical", "dense"),
}
_RERANKED = ("hybrid_rerank", "hybrid_rerank_llm")
# LlmError and RerankError codes the deep-search stage names; any other is a failure.
_LLM_WARNINGS = {
    TIMEOUT: "llm_rerank_timeout",
    "llm_timeout": "llm_rerank_timeout",
    "llm_spend_cap": "llm_spend_cap",
    "llm_rerank_unparseable": "llm_rerank_unparseable",
}


def _first_occurrence(hits: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """A branch that repeats an ID keeps it once, at its best (first) rank and score."""

    seen: dict[str, float] = {}
    for paper_id, score in hits:
        seen.setdefault(str(paper_id), float(score))
    return list(seen.items())


def _warning(stage: str, error: str) -> str:
    if error == TIMEOUT:
        return f"{stage}_timeout"
    if error in {"score_count_mismatch", "nonfinite_score"}:
        return f"{stage}_invalid_scores"
    return f"{stage}_unavailable"


class SearchService:
    """``search(request) -> SearchResponse``, with a trace of how it got there."""

    def __init__(
        self,
        *,
        engine: Engine | None,
        lexical: CandidateRetriever,
        dense: CandidateRetriever,
        reranker: Reranker | None,
        config: SearchConfig,
        runner: StageRunner | None = None,
        papers: PaperSource | None = None,
        capture: Callable[[], ReleaseRecord | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        listwise: ListwiseReranker | None = None,
    ) -> None:
        if engine is None and (papers is None or capture is None):
            # Without a database, metadata and the release must both come from the caller.
            raise ValueError("engine_required")
        self._branches: dict[str, CandidateRetriever] = {"lexical": lexical, "dense": dense}
        self._reranker = reranker
        self._listwise = listwise
        self._config = config
        self._runner = runner or ThreadedStageRunner()
        self._papers: PaperSource = papers or PaperStore(cast(Engine, engine))
        self._capture = capture or (lambda: capture_release(cast(Engine, engine)))
        self._clock = clock

    @property
    def config(self) -> SearchConfig:
        return self._config

    def identity(self, mode: str) -> dict[str, Any]:
        """Everything besides the release and its embedder that decides an ordering.

        A cached ordering is reusable only under the same identity, so a changed
        reranker, pair budget or fusion constant can never serve a stale ranking.
        """

        config = self._config
        branches = _MODE_BRANCHES[mode]
        identity: dict[str, Any] = {
            "branches": list(branches),
            "candidates": config.candidates_per_branch,
            "depth": config.cache_depth,
        }
        if config.candidates_source != "papers":
            # Found through another collection: an ordering cached under the paper
            # collection is not this ordering.
            identity["candidates_source"] = config.candidates_source
        if len(branches) > 1:
            identity["rrf_k"] = config.rrf_k
        if mode in _RERANKED:
            identity["reranker"] = str(getattr(self._reranker, "identity", "none"))
            identity["rerank_depth"] = config.rerank_depth
        if mode == "hybrid_rerank_llm":
            identity["listwise"] = str(getattr(self._listwise, "identity", "none"))
        return identity

    def warm(self, release: ReleaseRecord) -> dict[str, str]:
        """Pay first-call costs before serving: statistics, CUDA kernels, index pages.

        Run once at startup, outside every deadline. Without it the first
        request after a restart spends its dense budget loading kernels and is
        served degraded. A stage that fails here is reported, not raised: the
        request path degrades on its own, typed.
        """

        report: dict[str, str] = {}
        for name, branch in self._branches.items():
            try:
                branch.search("warm up", None, 1, release.id)
                report[name] = "ok"
            except Exception as error:  # noqa: BLE001 - reported, and handled per request
                report[name] = str(getattr(error, "code", type(error).__name__))
        if self._reranker is not None:
            try:
                validate_scores(self._reranker.score("warm up", ["warm up"]), 1)
                report["rerank"] = "ok"
            except Exception as error:  # noqa: BLE001 - reported, and handled per request
                report["rerank"] = str(getattr(error, "code", type(error).__name__))
        return report

    def close(self) -> None:
        for part in (self._runner, self._listwise):
            close = getattr(part, "close", None)
            if callable(close):
                close()

    def search(self, request: SearchRequest) -> SearchResponse:
        return self.search_with_trace(request)[0]

    def search_with_trace(self, request: SearchRequest) -> tuple[SearchResponse, SearchTrace]:
        started = self._clock()
        ranking = self.rank(request)
        response = self.page(
            ranking.ordering,
            offset=0,
            limit=request.limit,
            request_id=ranking.request_id,
            rows=ranking.rows,
            trace=ranking.trace,
        )
        ranking.trace.seconds = self._clock() - started
        return response, ranking.trace

    def rank(
        self,
        request: SearchRequest,
        *,
        release: ReleaseRecord | None = None,
        request_id: UUID | None = None,
    ) -> Ranking:
        """The whole ordering for a request, before any page is cut from it.

        A caller that already captured the release passes it in, so the cache key
        it computed and the ranking it stores name the same release.
        """

        started = self._clock()
        request_id = request_id or uuid4()
        # Read once: every stage of this request uses this release's pair.
        if release is None:
            release = self._capture()
        if release is None:
            raise SearchUnavailable("corpus_not_ready")
        trace = SearchTrace(
            request_id=str(request_id),
            release_id=release.id,
            mode=request.mode,
            models={
                "embedding": release.model_revision,
                "reranker": str(getattr(self._reranker, "identity", "none")),
            },
        )
        if request.mode == "hybrid_rerank_llm":
            trace.models["listwise"] = str(getattr(self._listwise, "identity", "none"))
        config = self._config

        def remaining() -> float:
            return config.total_seconds - (self._clock() - started)

        branches = _MODE_BRANCHES[request.mode]
        budget = max(0.0, min(config.branch_seconds, remaining()))
        stages: dict[str, Stage] = {}
        for name in branches:
            search = partial(
                self._branches[name].search,
                request.query,
                request.filters,
                config.candidates_per_branch,
                release.id,
            )
            stages[name] = (search, budget)
        outcomes = self._runner.run(stages)

        lists: dict[str, list[tuple[str, float]]] = {}
        for name in branches:
            outcome = outcomes[name]
            if outcome.error is None:
                hits = _first_occurrence(outcome.value)
                lists[name] = hits
                trace.stages[name] = {"seconds": round(outcome.seconds, 4), "candidates": hits}
            else:
                trace.stages[name] = {"seconds": round(outcome.seconds, 4), "error": outcome.error}
                trace.warnings.append(_warning(name, outcome.error))
        if not lists:
            trace.seconds = self._clock() - started
            raise SearchUnavailable("candidates_unavailable", ",".join(trace.warnings))

        scores: dict[str, dict[str, float]] = {}
        for name, hits in lists.items():
            for paper_id, score in hits:
                scores.setdefault(paper_id, {})[_BRANCH_SCORE[name]] = float(score)

        if len(lists) == 1:
            # One branch — by mode or because the other failed: its own order stands.
            ((name, hits),) = lists.items()
            ordered = [(paper_id, float(score)) for paper_id, score in hits]
        else:
            ordered = rrf([[pid for pid, _ in lists[name]] for name in branches], config.rrf_k)
            for paper_id, fused in ordered:
                scores[paper_id]["rrf"] = fused
            trace.stages["fusion"] = {"k": config.rrf_k, "candidates": ordered}

        rows: dict[str, PaperRow] = {}
        if request.mode in _RERANKED:
            ordered = self._rerank(request.query, ordered, scores, trace, remaining, rows)
        if request.mode == "hybrid_rerank_llm":

            def deep_remaining() -> float:
                return config.llm_total_seconds - (self._clock() - started)

            ordered = self._llm_rerank(
                request.query, ordered, scores, trace, deep_remaining, rows, request_id
            )
        ordered = ordered[: config.cache_depth]
        trace.seconds = self._clock() - started
        ordering = Ordering(
            release_id=release.id,
            items=tuple(ordered),
            scores={paper_id: scores.get(paper_id, {}) for paper_id, _ in ordered},
            warnings=tuple(trace.warnings),
        )
        return Ranking(request_id=request_id, ordering=ordering, trace=trace, rows=rows)

    def page(
        self,
        ordering: Ordering,
        *,
        offset: int,
        limit: int,
        request_id: UUID,
        rows: dict[str, PaperRow] | None = None,
        trace: SearchTrace | None = None,
    ) -> SearchResponse:
        """One page of an ordering, with the canonical metadata of what it shows.

        Ranks are positions in the whole ordering, so page two continues where
        page one stopped. A paper indexed but gone from the database is left out,
        never served with invented metadata, and the response says so.
        """

        window = list(ordering.items[offset : offset + limit])
        loaded = rows if rows is not None else {}
        tick = self._clock()
        loaded.update(
            self._papers.load([paper_id for paper_id, _ in window if paper_id not in loaded])
        )
        missing = [paper_id for paper_id, _ in window if paper_id not in loaded]
        warnings = list(ordering.warnings)
        if missing and "metadata_missing" not in warnings:
            warnings.append("metadata_missing")
        if trace is not None:
            stage = trace.stages.setdefault("hydrate", {"seconds": 0.0})
            stage["seconds"] = round(stage["seconds"] + self._clock() - tick, 4)
            stage["missing"] = missing
            if missing and "metadata_missing" not in trace.warnings:
                trace.warnings.append("metadata_missing")
        served = [
            (offset + index + 1, paper_id, score)
            for index, (paper_id, score) in enumerate(window)
            if paper_id in loaded
        ]
        return SearchResponse(
            request_id=request_id,
            corpus_release_id=ordering.release_id,
            items=[
                RankedPaper(
                    paper_id=UUID(paper_id),
                    score=score,
                    scores=dict(ordering.scores.get(paper_id, {})),
                    rank=rank,
                )
                for rank, paper_id, score in served
            ],
            papers={paper_id: loaded[paper_id].summary() for _, paper_id, _ in served},
            next_cursor=None,
            degraded=bool(warnings),
            warnings=warnings,
        )

    def _hydrate(
        self, paper_ids: Sequence[str], rows: dict[str, PaperRow], trace: SearchTrace
    ) -> None:
        """Load the metadata not yet loaded; every load counts toward one hydrate stage."""

        tick = self._clock()
        rows.update(self._papers.load([paper_id for paper_id in paper_ids if paper_id not in rows]))
        stage = trace.stages.setdefault("hydrate", {"seconds": 0.0})
        stage["seconds"] = round(stage["seconds"] + self._clock() - tick, 4)

    def _rerank(
        self,
        query: str,
        fused: list[tuple[str, float]],
        scores: dict[str, dict[str, float]],
        trace: SearchTrace,
        remaining: Callable[[], float],
        rows: dict[str, PaperRow],
    ) -> list[tuple[str, float]]:
        """Reorder at most ``depth`` fused candidates; on any failure keep the RRF order.

        Only the reranked head is returned: appending the unreranked tail would
        rank reranker logits against RRF scores, which measure different things.
        """

        head = fused[: self._config.rerank_depth]
        reranker = self._reranker
        if not head:
            return head
        if reranker is None:
            trace.warnings.append("rerank_unavailable")
            return head
        self._hydrate([paper_id for paper_id, _ in head], rows, trace)
        # Indexed but gone from the database: no text to judge and no metadata to
        # serve, so it leaves the candidates here, and says so, either way.
        if any(paper_id not in rows for paper_id, _ in head):
            trace.warnings.append("metadata_missing")
            head = [pair for pair in head if pair[0] in rows]
        candidates = [paper_id for paper_id, _ in head]
        texts = [rows[paper_id].text for paper_id in candidates]

        def score() -> list[float]:
            return validate_scores(reranker.score(query, texts), len(texts))

        attempts = 1 + max(0, self._config.rerank_retries)
        errors: list[str] = []
        for _ in range(attempts):
            budget = min(self._config.rerank_seconds, remaining())
            if budget <= 0:
                errors.append(TIMEOUT)
                break
            outcome = self._runner.run({"rerank": (score, budget)})["rerank"]
            if outcome.error is None:
                ordered = rerank_order(candidates, outcome.value)
                for paper_id, value in ordered:
                    scores[paper_id]["rerank"] = value
                trace.stages["rerank"] = {
                    "seconds": round(outcome.seconds, 4),
                    "attempts": len(errors) + 1,
                    "candidates": ordered,
                }
                return ordered
            errors.append(outcome.error)
            # A timeout used its whole budget; only a model error is worth a retry.
            if outcome.error == TIMEOUT:
                break
        trace.stages["rerank"] = {"errors": errors}
        trace.warnings.append(_warning("rerank", errors[-1]))
        return head

    def _llm_rerank(
        self,
        query: str,
        head: list[tuple[str, float]],
        scores: dict[str, dict[str, float]],
        trace: SearchTrace,
        remaining: Callable[[], float],
        rows: dict[str, PaperRow],
        request_id: UUID,
    ) -> list[tuple[str, float]]:
        """Let the hosted LLM reorder the reranked head; on any failure keep it as it is.

        Scores become ordinal (head size minus position, higher is better): the
        LLM gives an order, not a calibrated score. When the cross-encoder failed,
        the head is the RRF order and the LLM still reorders it, as LitSearch's
        listwise reranker reorders a BM25 list. The cross-encoder's own warning
        stays, so the ordering is still marked degraded.
        """

        listwise = self._listwise
        if not head:
            return head
        if listwise is None:
            trace.warnings.append("llm_rerank_unavailable")
            return head
        self._hydrate([paper_id for paper_id, _ in head], rows, trace)
        if any(paper_id not in rows for paper_id, _ in head):
            if "metadata_missing" not in trace.warnings:
                trace.warnings.append("metadata_missing")
            head = [pair for pair in head if pair[0] in rows]
        texts = [rows[paper_id].text for paper_id, _ in head]
        budget = min(self._config.llm_rerank_seconds, remaining())
        if budget <= 0:
            trace.stages["llm"] = {"error": TIMEOUT}
            trace.warnings.append("llm_rerank_timeout")
            return head
        order = partial(
            listwise.order,
            query,
            texts,
            timeout=budget,
            purpose="search",
            request_id=request_id,
        )
        outcome = self._runner.run({"llm": (order, budget)})["llm"]
        result = outcome.value
        if outcome.error is not None or not isinstance(result, ListwiseResult):
            error = outcome.error or "llm_result_invalid"
            trace.stages["llm"] = {"seconds": round(outcome.seconds, 4), "error": error}
            trace.warnings.append(_LLM_WARNINGS.get(error, "llm_rerank_failed"))
            return head
        if sorted(result.order) != list(range(len(head))):
            # The reranker promises a permutation; anything else is not used.
            trace.stages["llm"] = {
                "seconds": round(outcome.seconds, 4),
                "error": "not_a_permutation",
            }
            trace.warnings.append("llm_rerank_failed")
            return head
        ordered = [
            (head[index][0], float(len(head) - position))
            for position, index in enumerate(result.order)
        ]
        for position, (paper_id, _) in enumerate(ordered, 1):
            scores.setdefault(paper_id, {})["llm_rank"] = float(position)
        trace.stages["llm"] = {
            "seconds": round(outcome.seconds, 4),
            "ranked_by_model": result.ranked_by_model,
            "cost_usd": result.cost_usd,
            "served_model": result.served_model,
            "cached": result.cached,
            "candidates": ordered,
        }
        return ordered
