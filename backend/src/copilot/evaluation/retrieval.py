"""Retrieval experiments over a frozen split: run, score, bootstrap, record (P2.5).

An experiment config names a frozen dataset, a corpus release and a list of
variants — the four baselines plus single-factor ablations, each against the
variant it changes. Every variant runs the same queries through the real
search service, so they share one candidate pool and one set of deadlines.

Three rules hold throughout:

* A query that fails stays in every denominator: it scores zero and its time
  counts toward latency. A failure is never a missing row.
* Differences are paired and bootstrapped over query families, 1,000
  resamples at seed 42 (spec §11), so two variants are compared on the same
  resampled queries.
* The test split is locked. It runs only with ``locked_test`` for a release
  decision, and ``decide`` refuses anything but the validation split.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import numpy as np
import yaml
from pydantic import ValidationError

from ..contracts import ListwiseResult, PaperFilters, SearchRequest
from ..corpus.releases import ReleaseRecord
from ..search.service import (
    CANDIDATE_SOURCES,
    Outcome,
    PaperRow,
    SearchConfig,
    SearchService,
    SearchUnavailable,
    Stage,
)
from .datasets import SPLITS, Dataset, hydrate, read_dataset
from .metrics import evaluate
from .regression import validate_manifest

# The four baselines, cheapest first; the smoke set runs exactly these.
BASELINE_MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
MODES = (*BASELINE_MODES, "hybrid_rerank_llm")
RERANKED = ("hybrid_rerank", "hybrid_rerank_llm")
LLM_MODE = "hybrid_rerank_llm"
LLM_CONFIG = Path("configs/llm.yaml")
LISTWISE_PROMPT = Path("prompts/rerank/listwise-v1.yaml")
# Characters a word is assumed to take when the preflight prices a prompt: generous,
# so the worst case it reserves is never below what a call can cost.
PREFLIGHT_CHARS_PER_WORD = 7
WARMUP_QUERIES = 10
SLICES = ("in_domain", "all")
# Where a variant's first stage finds candidates (P2.6 step 4): the paper collection,
# the chunk collection collapsed to each paper's best evidence chunk, or each branch's
# RRF of both (``CANDIDATE_SOURCES``). Unset, a variant follows the shared search
# file, except over a packaged snapshot corpus, which has no chunks and stays at paper
# level. "papers" records nothing extra, so every run recorded before it reads as it was.
# Which id a label names: our corpus's paper id (matched by E1), or a LitSearch
# corpusid scored against a ``litsearch-v1`` release built from LitSearch's corpus.
LABELS = ("paper_id", "corpusid")
# Where paper metadata and reranker text come from: PostgreSQL for our corpus, the
# release's own snapshot for a corpus that was never ingested (LitSearch's).
PAPER_SOURCES = ("database", "snapshot")
TIMING_METHODOLOGY = (
    "Queries run one at a time through SearchService.rank with the production deadlines "
    "of configs/search.yaml; seconds are wall-clock around rank(), which includes both "
    "candidate branches, fusion, the reranked head's metadata read and the reranker, and "
    "excludes HTTP and page hydration. Before its first timed query each variant is warmed "
    "with the service's own warm-up and then 10 queries from another split (never the test "
    "split), whose results are discarded: GPU kernels warm per input shape, and without this "
    "the first variant to use a model would pay for every later one. p50/p95 interpolate "
    "linearly over every query of the split, failed ones included. GPU utilization and "
    "memory are sampled every second for the whole run, and other processes on the GPU are "
    "listed."
)


class ExperimentError(ValueError):
    """An experiment that cannot run, or a use of a split the rules forbid."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


# --- Configuration ------------------------------------------------------------------------


# What a variant may change about search (P2.6), by its path in configs/search.yaml:
# the SearchConfig field it sets, that field's type, and the modes it means anything
# for. Anything else in the file is shared by every variant of an experiment.
SEARCH_OVERRIDES: Mapping[tuple[str, str], tuple[str, type, tuple[str, ...]]] = {
    ("candidates", "per_branch"): ("candidates_per_branch", int, MODES),
    ("rerank", "depth"): ("rerank_depth", int, RERANKED),
    ("deadlines_seconds", "rerank"): ("rerank_seconds", float, RERANKED),
}


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    baseline: str | None = None
    changes: str | None = None
    models: str | None = None
    release: str | None = None
    pair_max_tokens: int | None = None
    # (SearchConfig field, value) pairs this variant overrides, sorted by field.
    search: tuple[tuple[str, float], ...] = ()
    # The configs/llm.yaml model key behind a hybrid_rerank_llm variant, and only one.
    llm: str | None = None
    # One of CANDIDATE_SOURCES, or None to follow the shared search file (see
    # ``candidate_source``).
    candidates: str | None = None


@dataclass(frozen=True)
class Experiment:
    path: Path
    sha256: str
    name: str
    dataset: Path
    query_text: str
    slice: str
    labels: str
    release: str
    papers: str
    search: Path
    models: Path
    recall_at: tuple[int, ...]
    ndcg_at: tuple[int, ...]
    mrr_at: tuple[int, ...]
    depth: int
    resamples: int
    seed: int
    confidence: float
    variants: tuple[Variant, ...]
    decision: Mapping[str, Any]

    def variant(self, name: str) -> Variant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise ExperimentError("unknown_variant", name)

    def models_for(self, variant: Variant) -> Path:
        return Path(variant.models) if variant.models else self.models

    def release_for(self, variant: Variant) -> str:
        return variant.release or self.release


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _search_overrides(name: str, mode: str, raw: object) -> tuple[tuple[str, float], ...]:
    """A variant's ``search`` block, refused unless every entry is a knob P2.6 may turn."""

    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ExperimentError("experiment_invalid", f"{name}.search")
    overrides: dict[str, float] = {}
    for section, values in raw.items():
        if not isinstance(values, Mapping):
            raise ExperimentError("experiment_invalid", f"{name}.search.{section}")
        for key, value in values.items():
            where = f"{name}.search.{section}.{key}"
            target = SEARCH_OVERRIDES.get((str(section), str(key)))
            if target is None:
                raise ExperimentError("experiment_invalid", f"{where} cannot be overridden")
            field_name, kind, modes = target
            if mode not in modes:
                raise ExperimentError("experiment_invalid", f"{where} means nothing for {mode}")
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ExperimentError("experiment_invalid", f"{where} must be a positive number")
            if kind is int and not isinstance(value, int):
                raise ExperimentError("experiment_invalid", f"{where} must be a whole number")
            overrides[field_name] = value
    return tuple(sorted(overrides.items()))


def candidate_source(experiment: Experiment, variant: Variant, config: SearchConfig) -> str:
    """What this variant's branches search: its own choice, else the shared file's.

    A corpus whose papers come from a snapshot was never parsed, so it has no chunk
    collection and is searched at paper level whatever the shared file says; the
    resolved value is what the manifest records.
    """

    if variant.candidates is not None:
        return variant.candidates
    if experiment.papers == "snapshot":
        return "papers"
    return config.candidates_source


def search_config_for(variant: Variant, base: SearchConfig) -> SearchConfig:
    """The search configuration one variant runs under: the shared file plus its overrides."""

    # Types were checked per field when the experiment loaded (SEARCH_OVERRIDES).
    overrides: dict[str, Any] = dict(variant.search)
    config = replace(base, **overrides)
    if config.rerank_seconds > config.total_seconds:
        # The reranker's budget is cut to what the total leaves, so a larger one is a
        # deadline the run claims and never applies.
        raise ExperimentError(
            "experiment_invalid",
            f"{variant.name}: rerank deadline {config.rerank_seconds} s exceeds the "
            f"{config.total_seconds} s total",
        )
    return config


# Deadlines only the opt-in LLM mode reads (LLM reranking plan, L4). Every other mode's
# recorded settings and digest leave them out, so adding them changed no recorded run.
DEEP_SEARCH_FIELDS = ("llm_rerank_seconds", "llm_total_seconds")


def search_values(
    config: SearchConfig, mode: str | None, candidates: str = "papers"
) -> dict[str, Any]:
    """The settings a mode reads; ``None`` is the shared file, without deep-search fields.

    A candidate source other than the paper collection is part of what was searched,
    so it is recorded here; the shipped source adds nothing, and every run recorded
    before it existed still reads as the same search.
    """

    values = asdict(config)
    if mode != "hybrid_rerank_llm":
        for name in DEEP_SEARCH_FIELDS:
            values.pop(name, None)
    # The file's own source is replaced by the resolved one, recorded only when it is
    # not the paper collection.
    values.pop("candidates_source", None)
    if candidates != "papers":
        values["candidates_source"] = candidates
    return values


def search_digest(config: SearchConfig, mode: str | None, candidates: str = "papers") -> str:
    """What a variant searched under, as one digest: equal only if every knob was."""

    values = json.dumps(
        search_values(config, mode, candidates), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(values.encode()).hexdigest()


def _llm_models() -> Mapping[str, Any]:
    from ..models.spend import load_llm_config

    return load_llm_config(LLM_CONFIG).models


def llm_preflight_usd(
    variants: Sequence[Variant], configs: Mapping[str, SearchConfig], *, queries: int
) -> float:
    """What a run's LLM variants could spend, priced before any query runs.

    Each query's prompt is assumed to be the whole reranked head, each candidate at
    300 words of 7 characters, and its answer the model's whole output budget.
    Cached answers cost nothing. This is a planning bound, not a guarantee: a
    reservation counts the JSON-escaped prompt, so an abstract heavy in long words or
    non-ASCII text can be reserved above it. The hard limits are the per-query
    mid-run guard and the ledger's daily cap, which stop a run, never let it
    overspend.
    """

    from ..models.spend import estimate_usd

    models = _llm_models()
    total = 0.0
    for variant in variants:
        if variant.llm is None:
            continue
        model = models[variant.llm]
        chars = 300 * PREFLIGHT_CHARS_PER_WORD * configs[variant.name].rerank_depth
        total += queries * estimate_usd(model.price, chars, model.max_output_tokens)
    return total


class MeteredListwise:
    """An evaluation's view of a listwise reranker: every call is billed to one run.

    The service asks for purpose ``search``; here every call is ``evaluation``,
    under this variant's own ledger run, and the facts the manifest reports —
    calls, cache hits, served models — are counted as they happen.
    """

    def __init__(self, inner: Any, ledger_run: str) -> None:
        self.inner = inner
        self.identity: str = str(inner.identity)
        self.ledger_run = ledger_run
        self.calls = 0
        self.cache_hits = 0
        self.served_models: set[str] = set()

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def order(
        self,
        query: str,
        texts: Sequence[str],
        *,
        timeout: float,
        purpose: str,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> ListwiseResult:
        self.calls += 1
        result: ListwiseResult = self.inner.order(
            query,
            texts,
            timeout=timeout,
            purpose="evaluation",
            request_id=request_id,
            run_id=self.ledger_run,
        )
        if result.cached:
            self.cache_hits += 1
        self.served_models.add(result.served_model)
        return result


def load_experiment(path: str | Path) -> Experiment:
    """Read and check an experiment config; an inconsistent one is refused before running."""

    config = Path(path)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ExperimentError("experiment_invalid", "root")
    try:
        dataset, metrics, bootstrap = raw["dataset"], raw["metrics"], raw["bootstrap"]
        variants = tuple(
            Variant(
                name=str(item["name"]),
                mode=str(item["mode"]),
                baseline=item.get("baseline"),
                changes=item.get("changes"),
                models=item.get("models"),
                release=item.get("release"),
                pair_max_tokens=item.get("pair_max_tokens"),
                search=_search_overrides(
                    str(item["name"]), str(item["mode"]), item.get("search")
                ),
                llm=None if item.get("llm") is None else str(item["llm"]),
                candidates=None if item.get("candidates") is None else str(item["candidates"]),
            )
            for item in raw["variants"]
        )
        experiment = Experiment(
            path=config,
            sha256=_file_sha256(config),
            name=str(raw["name"]),
            dataset=Path(dataset["path"]),
            query_text=str(dataset["query_text"]),
            slice=str(dataset.get("slice", "in_domain")),
            labels=str(dataset.get("labels", "paper_id")),
            release=str(raw["corpus"]["release"]),
            papers=str(raw["corpus"].get("papers", "database")),
            search=Path(raw["search"]),
            models=Path(raw["models"]),
            recall_at=tuple(int(k) for k in metrics["recall_at"]),
            ndcg_at=tuple(int(k) for k in metrics["ndcg_at"]),
            mrr_at=tuple(int(k) for k in metrics["mrr_at"]),
            depth=int(metrics["depth"]),
            resamples=int(bootstrap["resamples"]),
            seed=int(bootstrap["seed"]),
            confidence=float(bootstrap.get("confidence", 0.95)),
            variants=variants,
            decision=dict(raw.get("decision") or {}),
        )
    except (KeyError, TypeError) as error:
        raise ExperimentError("experiment_invalid", str(error)) from error
    names = [variant.name for variant in variants]
    if not names or len(names) != len(set(names)):
        raise ExperimentError("experiment_invalid", "variant names must be unique")
    for variant in variants:
        if variant.mode not in MODES:
            raise ExperimentError("experiment_invalid", f"{variant.name}.mode")
        if variant.baseline is not None and variant.baseline not in names:
            raise ExperimentError("experiment_invalid", f"{variant.name}.baseline")
        if (variant.mode == LLM_MODE) != (variant.llm is not None):
            raise ExperimentError(
                "experiment_invalid", f"{variant.name}.llm is required by {LLM_MODE} and only by it"
            )
        if variant.llm is not None and variant.llm not in _llm_models():
            raise ExperimentError("experiment_invalid", f"{variant.name}.llm {variant.llm} unknown")
        if variant.candidates is not None and variant.candidates not in CANDIDATE_SOURCES:
            raise ExperimentError(
                "experiment_invalid",
                f"{variant.name}.candidates must be one of {', '.join(CANDIDATE_SOURCES)}",
            )
    if experiment.slice not in SLICES:
        raise ExperimentError("experiment_invalid", "dataset.slice")
    if experiment.labels not in LABELS:
        raise ExperimentError("experiment_invalid", "dataset.labels")
    if experiment.papers not in PAPER_SOURCES:
        raise ExperimentError("experiment_invalid", "corpus.papers")
    if max(experiment.recall_at + experiment.ndcg_at + experiment.mrr_at) > experiment.depth:
        raise ExperimentError("experiment_invalid", "a cutoff exceeds metrics.depth")
    known = set(metric_names(experiment))
    for guard in experiment.decision.get("guards") or []:
        if not isinstance(guard, Mapping) or guard.get("metric") not in known:
            raise ExperimentError("experiment_invalid", "decision.guards: unknown metric")
        drop = guard.get("max_drop", experiment.decision.get("max_drop", 0.03))
        if isinstance(drop, bool) or not isinstance(drop, int | float) or drop <= 0:
            raise ExperimentError("experiment_invalid", "decision.guards: max_drop")
    return experiment


# --- Queries ------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    family_id: str
    split: str
    query_set: str
    text: str
    grades: Mapping[str, int]
    specificity: int | None = None

    @property
    def relevant(self) -> frozenset[str]:
        return frozenset(item for item, grade in self.grades.items() if grade > 0)


def dataset_digest(directory: Path) -> str:
    """One digest over the frozen ids, labels and splits: a changed label is a new dataset."""

    digest = hashlib.sha256()
    for name in ("queries.jsonl", "qrels.jsonl", "splits.jsonl"):
        digest.update(name.encode() + b"\n")
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def select_queries(
    dataset: Dataset, split: str, slice_: str, labels: str = "paper_id"
) -> list[EvalQuery]:
    """A split's queries with their labels in our corpus, in query-id order."""

    if split not in SPLITS:
        raise ExperimentError("unknown_split", split)
    assigned = {record.query_id: record for record in dataset.splits}
    from .litsearch import litsearch_paper_id

    grades: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels:
        if labels == "corpusid":
            grades.setdefault(qrel.query_id, {})[litsearch_paper_id(qrel.corpusid)] = qrel.grade
        elif qrel.paper_id is not None:
            grades.setdefault(qrel.query_id, {})[qrel.paper_id] = qrel.grade
    chosen = []
    for query in sorted(dataset.queries, key=lambda record: record.query_id):
        placement = assigned.get(query.query_id)
        if placement is None or placement.split != split:
            continue
        if slice_ == "in_domain" and not query.in_domain:
            continue
        query_grades = grades.get(query.query_id, {})
        if not any(grade > 0 for grade in query_grades.values()):
            # Nothing it asks for is in this corpus: recall would be undefined.
            continue
        chosen.append(
            EvalQuery(
                query_id=query.query_id,
                family_id=placement.family_id,
                split=split,
                query_set=query.query_set,
                text=query.query,
                grades=query_grades,
                specificity=query.specificity,
            )
        )
    return chosen


def load_queries(experiment: Experiment, split: str, data_dir: Path) -> list[EvalQuery]:
    """Frozen labels from the repository, query text from DATA_DIR (it has no license)."""

    source = data_dir / experiment.query_text
    texts = {
        str(row["query_id"]): str(row["query"])
        for row in (json.loads(line) for line in source.read_text(encoding="utf-8").splitlines())
        if row.get("query")
    }
    dataset = hydrate(read_dataset(experiment.dataset), texts)
    return select_queries(dataset, split, experiment.slice, experiment.labels)


# --- Scoring ------------------------------------------------------------------------------


def metric_names(experiment: Experiment) -> list[str]:
    names = [f"recall@{k}" for k in experiment.recall_at]
    names += [f"ndcg@{k}" for k in experiment.ndcg_at]
    names += [f"mrr@{k}" for k in experiment.mrr_at]
    return names


def cutoffs(experiment: Experiment) -> list[int]:
    return sorted(set(experiment.recall_at + experiment.ndcg_at + experiment.mrr_at))


def score(
    ranked: Sequence[str], grades: Mapping[str, int], experiment: Experiment
) -> dict[str, Any]:
    """Every configured metric for one query, each cutoff with its judged coverage."""

    scores: dict[str, Any] = {}
    for k in cutoffs(experiment):
        result = evaluate(ranked=ranked, grades=grades, k=k)
        if k in experiment.recall_at:
            scores[f"recall@{k}"] = result.recall_at_k
        if k in experiment.ndcg_at:
            scores[f"ndcg@{k}"] = result.ndcg_at_k
        if k in experiment.mrr_at:
            scores[f"mrr@{k}"] = result.mrr_at_k
        scores[f"judged@{k}"] = result.judged_coverage
    return scores


@dataclass
class QueryResult:
    variant: str
    query_id: str
    family_id: str
    query_set: str
    ranked: list[str]
    relevant: list[str]
    metrics: dict[str, Any]
    candidate_recall: float | None
    rerank_pool_recall: float | None
    seconds: float
    stages: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failure: str | None = None
    specificity: int | None = None

    def row(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "query_id": self.query_id,
            "family_id": self.family_id,
            "query_set": self.query_set,
            "specificity": self.specificity,
            "ranked": self.ranked,
            "relevant": self.relevant,
            **{key: value for key, value in sorted(self.metrics.items())},
            "candidate_recall": self.candidate_recall,
            "rerank_pool_recall": self.rerank_pool_recall,
            "seconds": self.seconds,
            "stages": json.dumps(self.stages, sort_keys=True),
            "warnings": self.warnings,
            "failure": self.failure,
        }


def _recall(pool: Iterable[str], relevant: frozenset[str]) -> float | None:
    if not relevant:
        return None
    return len(relevant.intersection(pool)) / len(relevant)


def _failed(
    variant: Variant, query: EvalQuery, experiment: Experiment, seconds: float, code: str
) -> QueryResult:
    zeros: dict[str, Any] = {name: 0.0 for name in metric_names(experiment)}
    zeros.update({f"judged@{k}": None for k in cutoffs(experiment)})
    return QueryResult(
        variant=variant.name,
        query_id=query.query_id,
        family_id=query.family_id,
        query_set=query.query_set,
        ranked=[],
        relevant=sorted(query.relevant),
        metrics=zeros,
        candidate_recall=0.0,
        rerank_pool_recall=0.0 if variant.mode in RERANKED else None,
        seconds=seconds,
        failure=code,
        specificity=query.specificity,
    )


def run_query(
    service: SearchService,
    variant: Variant,
    query: EvalQuery,
    release: ReleaseRecord,
    experiment: Experiment,
    clock: Callable[[], float] = time.monotonic,
) -> QueryResult:
    """One query through the service; a failure is a scored zero, never an absence."""

    started = clock()
    try:
        request = SearchRequest.model_validate(
            {
                "query": query.text,
                "mode": variant.mode,
                "filters": PaperFilters(),
                "limit": min(experiment.depth, 50),
            }
        )
    except ValidationError:
        return _failed(variant, query, experiment, clock() - started, "invalid_query")
    try:
        ranking = service.rank(request, release=release)
    except SearchUnavailable as error:
        return _failed(variant, query, experiment, clock() - started, error.code)
    seconds = clock() - started
    ranked = [paper_id for paper_id, _ in ranking.ordering.items[: experiment.depth]]
    stages = ranking.trace.stages
    pool = {
        paper_id
        for branch in ("lexical", "dense")
        for paper_id, _ in stages.get(branch, {}).get("candidates", [])
    }
    rerank = stages.get("rerank", {}).get("candidates")
    return QueryResult(
        variant=variant.name,
        query_id=query.query_id,
        family_id=query.family_id,
        query_set=query.query_set,
        ranked=ranked,
        relevant=sorted(query.relevant),
        metrics=score(ranked, query.grades, experiment),
        candidate_recall=_recall(pool, query.relevant),
        rerank_pool_recall=(
            _recall((paper_id for paper_id, _ in rerank), query.relevant)
            if rerank is not None
            else None
        ),
        seconds=seconds,
        stages={
            name: float(detail["seconds"])
            for name, detail in stages.items()
            if isinstance(detail, Mapping) and "seconds" in detail
        },
        warnings=list(ranking.ordering.warnings),
        specificity=query.specificity,
    )


# --- Aggregation --------------------------------------------------------------------------


def _by_family(
    results: Sequence[QueryResult], value: Callable[[QueryResult], float]
) -> tuple[list[str], np.ndarray]:
    """Mean per family, families in sorted order: the unit every resample draws."""

    groups: dict[str, list[float]] = {}
    for result in results:
        groups.setdefault(result.family_id, []).append(value(result))
    families = sorted(groups)
    return families, np.array([np.mean(groups[family]) for family in families], dtype=np.float64)


def _metric(name: str) -> Callable[[QueryResult], float]:
    def value(result: QueryResult) -> float:
        return float(result.metrics[name])

    return value


def _resample(count: int, resamples: int, seed: int) -> np.ndarray:
    # Same seed, same count, same draws: every metric and every variant pair of one
    # split is resampled over the identical family indices.
    return np.random.default_rng(seed).integers(0, count, size=(resamples, count))


def bootstrap_interval(
    values: np.ndarray, *, resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    """Percentile interval of the mean over family resamples."""

    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = values[_resample(len(values), resamples, seed)].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return (float(low), float(high))


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "max_ms": None}
    p50, p95 = np.percentile(np.array(values) * 1000, [50, 95])
    return {
        "p50_ms": round(float(p50), 1),
        "p95_ms": round(float(p95), 1),
        "max_ms": round(max(values) * 1000, 1),
    }


def summarize(results: Sequence[QueryResult], experiment: Experiment) -> dict[str, Any]:
    """Means with intervals, coverage, failures, degradation and latency for one variant."""

    summary: dict[str, Any] = {"queries": len(results)}
    for name in metric_names(experiment):
        _, values = _by_family(results, _metric(name))
        low, high = bootstrap_interval(
            values,
            resamples=experiment.resamples,
            seed=experiment.seed,
            confidence=experiment.confidence,
        )
        summary[name] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "low": low,
            "high": high,
        }
    for k in cutoffs(experiment):
        judged = [result.metrics[f"judged@{k}"] for result in results]
        known = [value for value in judged if value is not None]
        summary[f"judged@{k}"] = {
            "mean": float(np.mean(known)) if known else None,
            "unknown": len(judged) - len(known),
        }
    for name in ("candidate_recall", "rerank_pool_recall"):
        observed = [
            getattr(result, name) for result in results if getattr(result, name) is not None
        ]
        summary[name] = float(np.mean(observed)) if observed else None
    summary["failures"] = {
        "count": sum(1 for result in results if result.failure),
        "codes": dict(Counter(result.failure for result in results if result.failure)),
    }
    summary["degraded"] = {
        "count": sum(1 for result in results if result.warnings),
        "warnings": dict(Counter(w for result in results for w in result.warnings)),
    }
    summary["latency"] = _percentiles([result.seconds for result in results])
    stage_names = sorted({name for result in results for name in result.stages})
    summary["stages"] = {
        name: _percentiles([result.stages[name] for result in results if name in result.stages])
        for name in stage_names
    }
    summary["by_query_set"] = _facet(results, experiment, lambda result: result.query_set)
    if any(result.specificity is not None for result in results):
        summary["by_specificity"] = _facet(
            results, experiment, lambda result: str(result.specificity)
        )
    return summary


def _facet(
    results: Sequence[QueryResult], experiment: Experiment, key: Callable[[QueryResult], str]
) -> dict[str, Any]:
    """Mean metrics per value of one query attribute, with its query count."""

    groups: dict[str, list[QueryResult]] = {}
    for result in results:
        groups.setdefault(key(result), []).append(result)
    return {
        value: {"queries": len(subset)}
        | {
            name: float(np.mean([float(r.metrics[name]) for r in subset]))
            for name in metric_names(experiment)
        }
        for value, subset in sorted(groups.items())
    }


def paired(
    baseline: Sequence[QueryResult],
    candidate: Sequence[QueryResult],
    metric: str,
    experiment: Experiment,
) -> dict[str, Any]:
    """Candidate minus baseline, per family, with its bootstrap interval."""

    families_a, a = _by_family(baseline, lambda result: float(result.metrics[metric]))
    families_b, b = _by_family(candidate, lambda result: float(result.metrics[metric]))
    if families_a != families_b:
        raise ExperimentError("unpaired_results", metric)
    difference = b - a
    low, high = bootstrap_interval(
        difference,
        resamples=experiment.resamples,
        seed=experiment.seed,
        confidence=experiment.confidence,
    )
    return {
        "metric": metric,
        "baseline": float(a.mean()),
        "candidate": float(b.mean()),
        "difference": float(difference.mean()),
        "low": low,
        "high": high,
        "wins": int((difference > 0).sum()),
        "losses": int((difference < 0).sum()),
        "ties": int((difference == 0).sum()),
    }


def comparisons(
    results: Mapping[str, Sequence[QueryResult]], experiment: Experiment
) -> dict[str, Any]:
    """Each ablation against its baseline, and each mode against the one before it."""

    pairs: list[tuple[str, str]] = []
    order = [name for name in experiment.decision.get("order", []) if name in results]
    pairs += list(zip(order, order[1:], strict=False))
    pairs += [
        (variant.baseline, variant.name)
        for variant in experiment.variants
        if variant.baseline and variant.name in results and variant.baseline in results
    ]
    output: dict[str, Any] = {}
    for base, name in pairs:
        output[f"{name}_vs_{base}"] = {
            "baseline": base,
            "candidate": name,
            "metrics": {
                metric: paired(results[base], results[name], metric, experiment)
                for metric in metric_names(experiment)
            },
        }
    return output


# --- The decision -------------------------------------------------------------------------


def decide(
    split: str,
    results: Mapping[str, Sequence[QueryResult]],
    summaries: Mapping[str, Mapping[str, Any]],
    experiment: Experiment,
) -> dict[str, Any]:
    """Apply the config's pre-registered rules. Only the validation split may choose.

    Modes climb in cost order: a costlier mode replaces the current choice only
    if its primary-metric gain has a paired interval wholly above zero and its
    p95 fits the budget. A cheaper ablation replaces its baseline only if its
    interval rules out losing more than ``max_drop``. Otherwise the simpler
    configuration is retained — no improvement is required for progress.
    """

    if split != "validation":
        raise ExperimentError(
            "held_out_split" if split == "test" else "not_the_choice_split", split
        )
    rules = experiment.decision
    primary = str(rules.get("primary", "ndcg@10"))
    budget = float(rules.get("latency_p95_seconds", 3.0))
    max_drop = float(rules.get("max_drop", 0.03))
    # Opt-in (P2.6): a candidate with any degraded query measured a fallback — a
    # timeout or a missing stage — rather than itself, so it cannot be chosen.
    require_clean = bool(rules.get("require_undegraded", False))
    order = [name for name in rules.get("order", []) if name in results]
    if not order:
        raise ExperimentError("experiment_invalid", "decision.order")

    def p95(name: str) -> float:
        value = summaries[name]["latency"]["p95_ms"]
        return float("inf") if value is None else float(value) / 1000

    def degraded(name: str) -> int:
        return int((summaries[name].get("degraded") or {}).get("count", 0))

    def clean(name: str) -> bool:
        return not require_clean or degraded(name) == 0

    # Opt-in (P2.6 step 4): a step whose primary is a recall metric must also not lose
    # the top of the page. Each guard is a metric whose paired interval must rule out a
    # drop beyond its ``max_drop``; a step is promoted only if every guard holds.
    def guard_checks(base: str, name: str) -> list[dict[str, Any]]:
        checks = []
        for guard in rules.get("guards") or []:
            metric = str(guard["metric"])
            drop = float(guard.get("max_drop", max_drop))
            comparison = paired(results[base], results[name], metric, experiment)
            checks.append(
                {
                    "metric": metric,
                    "max_drop": drop,
                    "comparison": comparison,
                    "holds": comparison["low"] > -drop,
                }
            )
        return checks

    current = order[0]
    steps = []
    for name in order[1:]:
        comparison = paired(results[current], results[name], primary, experiment)
        better = comparison["low"] > 0
        fits = p95(name) <= budget
        # Both sides: a degraded baseline would flatter the candidate's gain.
        both_clean = clean(current) and clean(name)
        guards = guard_checks(current, name)
        guarded = all(check["holds"] for check in guards)
        steps.append(
            {
                "from": current,
                "to": name,
                "comparison": comparison,
                "p95_seconds": p95(name),
                "gain_supported": better,
                "fits_budget": fits,
                "degraded": degraded(name),
                "clean": both_clean,
                "guards": guards,
                "guards_hold": guarded,
                "promoted": better and fits and both_clean and guarded,
            }
        )
        if better and fits and both_clean and guarded:
            current = name
    ablations = []
    for variant in experiment.variants:
        if not variant.baseline or variant.name not in results or variant.baseline not in results:
            continue
        comparison = paired(results[variant.baseline], results[variant.name], primary, experiment)
        noninferior = comparison["low"] > -max_drop
        ablations.append(
            {
                "variant": variant.name,
                "baseline": variant.baseline,
                "changes": variant.changes,
                "comparison": comparison,
                "p95_seconds": p95(variant.name),
                "noninferior": noninferior,
                "clean": clean(variant.name),
                "adopt": noninferior and p95(variant.name) <= budget and clean(variant.name),
                "applies": experiment.variant(variant.baseline).mode
                == experiment.variant(current).mode,
            }
        )
    chosen = experiment.variant(current)
    adopted = [item for item in ablations if item["applies"] and item["adopt"]]
    return {
        "split": split,
        "primary": primary,
        "rules": dict(rules),
        "steps": steps,
        "ablations": ablations,
        "chosen": {
            "variant": current,
            "mode": chosen.mode,
            "adopted_ablations": [item["variant"] for item in adopted],
        },
    }


# --- Manifest -----------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def code_state() -> dict[str, Any]:
    """HEAD plus a digest of any uncommitted change, so a dirty run is still identified."""

    try:
        sha = _git("rev-parse", "HEAD").strip()
        diff = _git("diff", "HEAD")
        untracked = _git("ls-files", "--others", "--exclude-standard")
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": None}
    dirty = bool(diff.strip() or untracked.strip())
    return {
        "git_sha": sha,
        "dirty": dirty,
        "diff_sha256": hashlib.sha256((diff + untracked).encode()).hexdigest() if dirty else None,
    }


def hardware() -> dict[str, Any]:
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info: dict[str, Any] = {"cpu": cpu or platform.machine(), "cores": os.cpu_count()}
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        info["ram_gb"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        info["cuda"] = torch.version.cuda
    except ImportError:
        info.update({"torch": "not installed", "gpu": "none", "cuda": None})
    return info


class GpuSampler:
    """``nvidia-smi`` once a second for the run's duration, and who else was on the GPU."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._samples: list[tuple[int, int]] = []
        self._thread: threading.Thread | None = None
        self._others: set[str] = set()

    def _foreign(self) -> None:
        try:
            listing = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return
        self._others.update(
            pid.strip()
            for pid in listing.splitlines()
            if pid.strip() and pid.strip() != str(os.getpid())
        )

    def __enter__(self) -> GpuSampler:
        self._foreign()
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-l",
                    "1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return self

        def read() -> None:
            assert self._process is not None and self._process.stdout is not None
            for line in self._process.stdout:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    self._samples.append((int(parts[0]), int(parts[1])))

        self._thread = threading.Thread(target=read, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._foreign()
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=10)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def summary(self) -> dict[str, Any]:
        if not self._samples:
            return {"available": False}
        utilization = [sample[0] for sample in self._samples]
        memory = [sample[1] for sample in self._samples]
        return {
            "available": True,
            "samples": len(self._samples),
            "utilization_percent": {
                "p50": float(np.percentile(utilization, 50)),
                "max": max(utilization),
            },
            "memory_mib": {"min": min(memory), "max": max(memory)},
            # Our own worker processes are counted too; the report reads this beside
            # the memory floor, which shows what was resident before the run.
            "compute_processes_seen": sorted(self._others),
        }


def gpu_before_run(seconds: int = 5) -> dict[str, Any]:
    """What the GPU was doing before this run loaded anything: someone else's load.

    During a run our own reranking keeps utilization high, so contention can
    only be told apart beforehand. Above 20% here, the run's latencies — and any
    deadline it missed — are not this system's alone.
    """

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-l",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=seconds,
        ).stdout
    except subprocess.TimeoutExpired as expired:
        output = (
            expired.stdout.decode() if isinstance(expired.stdout, bytes) else (expired.stdout or "")
        )
    except OSError:
        return {"available": False}
    samples = [
        (int(parts[0]), int(parts[1]))
        for parts in (line.replace(" ", "").split(",") for line in output.splitlines())
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
    ]
    if not samples:
        return {"available": False}
    utilization = float(np.median([sample[0] for sample in samples]))
    return {
        "available": True,
        "samples": len(samples),
        "utilization_percent_p50": utilization,
        "memory_mib": max(sample[1] for sample in samples),
        "busy": utilization > 20,
    }


# --- The real run -------------------------------------------------------------------------


class ModelFactory(Protocol):
    """Where a run's models come from: the pinned weights, or fixtures in a test."""

    def embedder(self, models: Path) -> tuple[Any, int | None]: ...

    def reranker(self, models: Path) -> Any: ...

    def listwise(self, model_key: str, ledger: Any, cache: Any) -> Any: ...


class PinnedModels:
    """The pinned embedders and reranker under DATA_DIR, each loaded once per run."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._embedders: dict[Path, tuple[Any, int | None]] = {}
        self._rerankers: dict[Path, Any] = {}

    def embedder(self, models: Path) -> tuple[Any, int | None]:
        from ..models.embeddings import TransformerEmbedding, load_embedding_spec

        if models not in self._embedders:
            spec = load_embedding_spec(models)
            model = TransformerEmbedding(spec, self._data_dir)
            self._embedders[models] = (model, spec.max_tokens["papers"])
        return self._embedders[models]

    def reranker(self, models: Path) -> Any:
        from ..search.rerank import CrossEncoderReranker, load_reranker_spec

        if models not in self._rerankers:
            self._rerankers[models] = CrossEncoderReranker(
                load_reranker_spec(models), self._data_dir
            )
        return self._rerankers[models]

    def listwise(self, model_key: str, ledger: Any, cache: Any) -> Any:
        """The configured hosted LLM, with the key from the environment; never a fixture."""

        from ..config import Settings
        from ..models.llm import LlmError, build_client
        from ..search.llm_rerank import LlmListwiseReranker, load_prompt

        try:
            client = build_client(
                model_key, Settings(), ledger, cache=cache, llm_config=LLM_CONFIG  # type: ignore[call-arg]
            )
        except LlmError as error:
            raise ExperimentError(error.code, model_key) from error
        return LlmListwiseReranker(client, load_prompt(LISTWISE_PROMPT))


def snapshot_papers(release: ReleaseRecord) -> MemoryPapers:
    """Paper metadata and reranker text from the snapshot a release was built from.

    For a corpus that was never ingested into PostgreSQL — LitSearch's — this is
    the only copy of its titles and abstracts, and the same text the index encoded.
    """

    import pyarrow.parquet as pq

    from ..corpus.export import shard_paths, validate_manifest

    manifest_path = Path(str(release.counts.get("manifest", "")))
    manifest = validate_manifest(manifest_path)
    rows: dict[str, PaperRow] = {}
    for shard in shard_paths(manifest, "papers"):
        for row in pq.read_table(manifest_path.parent / shard).to_pylist():
            rows[str(row["paper_id"])] = PaperRow(
                paper_id=str(row["paper_id"]),
                title=str(row["title"] or ""),
                abstract=row.get("abstract"),
                authors=(),
                venue=str(row.get("venue") or ""),
                year=int(row.get("year") or 0),
                pdf_url=row.get("pdf_url"),
                fulltext=row.get("parse_status") == "parsed",
            )
    return MemoryPapers(rows)


def _release_facts(record: ReleaseRecord, client: Any) -> dict[str, Any]:
    build = record.counts.get("papers") if isinstance(record.counts, Mapping) else None
    vocabulary = build.get("vocabulary", {}) if isinstance(build, Mapping) else {}
    facts: dict[str, Any] = {
        "manifest_sha256": record.manifest_sha256,
        "model_revision": record.model_revision,
        "paper_collection": record.paper_collection,
        "bm25_vocabulary_sha256": vocabulary.get("sha256"),
    }
    try:
        info = client.get_collection(record.paper_collection)
        dense = info.config.params.vectors
        size = dense["dense"].size if isinstance(dense, Mapping) else None
        facts.update(
            {
                "points": info.points_count,
                "dense_dimensions": size,
                "dense_vector_bytes": (info.points_count or 0) * (size or 0) * 4,
            }
        )
    except Exception as error:  # noqa: BLE001 - a missing collection fails the run below
        raise ExperimentError("collection_missing", record.paper_collection) from error
    return facts


def run_retrieval(
    config: Path,
    split: str,
    out: Path,
    *,
    data_dir: Path,
    database_url: str,
    qdrant_url: str,
    locked_test: bool = False,
    limit_queries: int | None = None,
    models: ModelFactory | None = None,
    max_spend_usd: float | None = None,
    llm_daily_cap_usd: Decimal | None = None,
    llm_cache: Path | None = None,
) -> dict[str, Any]:
    """Every variant of an experiment over one split: manifest, metrics and per-query rows.

    A run with an LLM variant needs a budget (``max_spend_usd``). It is refused
    before anything loads when its worst case would pass that budget, and it stops,
    writing nothing, once the ledger shows its real spend past it.
    """

    from qdrant_client import QdrantClient

    from ..corpus.releases import load_release
    from ..db.session import make_engine
    from ..search.chunks import dense_branch, lexical_branch
    from ..search.index import CHUNKS
    from ..search.service import load_search_config

    experiment = load_experiment(config)
    if split not in SPLITS:
        raise ExperimentError("unknown_split", split)
    if split == "test" and not locked_test:
        raise ExperimentError(
            "locked_test_required", "the test split runs only for a release decision"
        )
    queries = load_queries(experiment, split, data_dir)
    if limit_queries is not None:
        queries = queries[:limit_queries]
    if not queries:
        raise ExperimentError("no_queries", split)
    # Warm-up queries come from another split, so no timed query is pre-cached,
    # and never from test, which is read only when it is being scored.
    warm_split = "validation" if split == "development" else "development"
    warm_queries = load_queries(experiment, warm_split, data_dir)[:WARMUP_QUERIES]
    # Every variant's configuration is settled before anything loads, so an override
    # that cannot apply fails in seconds rather than after the models warm up.
    search_config = load_search_config(experiment.search)
    configs = {
        variant.name: search_config_for(variant, search_config) for variant in experiment.variants
    }
    sources = {
        variant.name: candidate_source(experiment, variant, configs[variant.name])
        for variant in experiment.variants
    }
    llm_variants = [variant for variant in experiment.variants if variant.llm is not None]
    if llm_variants:
        if max_spend_usd is None:
            raise ExperimentError(
                "run_budget_required", "a run with an LLM variant needs --max-spend-usd"
            )
        worst = llm_preflight_usd(llm_variants, configs, queries=len(queries))
        if worst > max_spend_usd:
            raise ExperimentError(
                "run_budget_exceeded",
                f"worst case ${worst:.4f} for {len(queries)} queries > budget ${max_spend_usd:.4f}",
            )
    created = datetime.now(UTC)
    run_id = f"{experiment.name}-{split}-{created:%Y%m%dT%H%M%SZ}"

    before = gpu_before_run()
    engine = make_engine(database_url)
    client = QdrantClient(url=qdrant_url, timeout=60)
    releases = {
        name: load_release(engine, name)
        for name in {experiment.release_for(v) for v in experiment.variants}
    }
    facts = {name: _release_facts(record, client) for name, record in releases.items()}
    for variant in experiment.variants:
        # A chunk-level first stage needs the release's chunk collection to have been
        # built; a release with only its paper collection cannot answer through chunks.
        build = releases[experiment.release_for(variant)].counts.get(CHUNKS)
        if sources[variant.name] != "papers" and not (
            isinstance(build, Mapping) and build.get("points")
        ):
            raise ExperimentError(
                "chunks_not_built",
                f"{variant.name}: release {experiment.release_for(variant)} has no chunk "
                "collection to find candidates in",
            )
    papers = (
        {name: snapshot_papers(record) for name, record in releases.items()}
        if experiment.papers == "snapshot"
        else {}
    )

    factory = models or PinnedModels(data_dir)
    embedders = {
        path: factory.embedder(path)
        for path in {experiment.models_for(variant) for variant in experiment.variants}
    }
    base_reranker = None
    rerankers: dict[int | None, Any] = {}
    if any(variant.mode in RERANKED for variant in experiment.variants):
        base_reranker = factory.reranker(experiment.models)
        rerankers[getattr(base_reranker, "pair_max_tokens", None)] = base_reranker
    ledger = None
    cache = None
    if llm_variants:
        from ..models.llm import ResponseCache
        from ..models.spend import SpendLedger

        try:
            ledger = SpendLedger(engine, llm_daily_cap_usd)
        except ValueError as error:
            raise ExperimentError(str(error), "LLM_DAILY_SPEND_CAP_USD") from error
        cache = ResponseCache(llm_cache or data_dir / "cache" / "llm")

    results: dict[str, list[QueryResult]] = {}
    warmups: dict[str, float] = {}
    variants_manifest: dict[str, Any] = {}
    with GpuSampler() as sampler:
        for variant in experiment.variants:
            path = experiment.models_for(variant)
            release = releases[experiment.release_for(variant)]
            reranker = None
            if variant.mode in RERANKED and base_reranker is not None:
                budget = variant.pair_max_tokens or getattr(base_reranker, "pair_max_tokens", None)
                if budget not in rerankers:
                    rerankers[budget] = base_reranker.with_pair_budget(budget)
                reranker = rerankers[budget]
            embedder, max_tokens = embedders[path]
            listwise = None
            if variant.llm is not None:
                # Each LLM variant bills its own ledger run, so its cost is its own.
                listwise = MeteredListwise(
                    factory.listwise(variant.llm, ledger, cache),
                    f"{run_id}.{variant.name}.{uuid4().hex[:8]}",
                )
            source = sources[variant.name]
            k = configs[variant.name].rrf_k
            service = SearchService(
                engine=engine,
                lexical=lexical_branch(engine, client, source=source, k=k),
                dense=dense_branch(
                    engine, client, embedder, source=source, max_tokens=max_tokens, k=k
                ),
                reranker=reranker,
                config=configs[variant.name],
                papers=papers.get(release.id),
                listwise=listwise,
            )
            # Warm-up never spends: an LLM variant warms as the mode it builds on.
            warm_variant = replace(variant, mode="hybrid_rerank") if listwise else variant
            try:
                started = time.monotonic()
                service.warm(release)
                for query in warm_queries:
                    run_query(service, warm_variant, query, release, experiment)
                warmups[variant.name] = round(time.monotonic() - started, 3)
                rows: list[QueryResult] = []
                for query in queries:
                    rows.append(run_query(service, variant, query, release, experiment))
                    if listwise is not None and ledger is not None and max_spend_usd is not None:
                        spent = sum(
                            ledger.run_spend(entry["llm"]["ledger_run_id"])
                            for entry in variants_manifest.values()
                            if "llm" in entry
                        ) + ledger.run_spend(listwise.ledger_run)
                        if spent > max_spend_usd:
                            raise ExperimentError(
                                "run_budget_exceeded",
                                f"spent ${spent:.4f} > budget ${max_spend_usd:.4f}; "
                                "nothing written",
                            )
                results[variant.name] = rows
            finally:
                service.close()
                if listwise is not None:
                    listwise.close()
            variants_manifest[variant.name] = {
                "mode": variant.mode,
                "release_id": release.id,
                "embedding": embedder.identity,
                "embedding_precision": getattr(embedder, "precision", None),
                "reranker": reranker.identity if reranker is not None else "none",
                "reranker_precision": getattr(reranker, "precision", None),
                "baseline": variant.baseline,
                "changes": variant.changes,
                # What this variant searched under, so two that differ never pass as equal.
                "search": search_values(
                    configs[variant.name], variant.mode, sources[variant.name]
                ),
                "search_overrides": dict(variant.search),
                "search_sha256": search_digest(
                    configs[variant.name], variant.mode, sources[variant.name]
                ),
                "candidates": sources[variant.name],
                "warmup_seconds": warmups[variant.name],
                "warmup_queries": {"split": warm_split, "count": len(warm_queries)},
            }
            if listwise is not None and ledger is not None:
                variants_manifest[variant.name]["llm"] = _llm_facts(listwise, engine)

    main = releases[experiment.release]
    manifest = build_manifest(
        experiment,
        split=split,
        queries=queries,
        sources={record.source for record in read_dataset(experiment.dataset).queries},
        corpus=main,
        releases=facts,
        variants=variants_manifest,
        search_config=search_config,
        locked_test=locked_test,
        limited=limit_queries,
        gpu={**sampler.summary(), "before_run": before},
        created=created,
    )
    return write_run(out, split, experiment, manifest, results)


def _llm_facts(listwise: MeteredListwise, engine: Any) -> dict[str, Any]:
    """What an LLM variant was and what it cost, from the counts and the spend ledger.

    Tokens and cost are what this run was billed: cached answers add neither.
    """

    from ..models.spend import spend_report

    inner = listwise.inner
    billed = spend_report(engine, run_id=listwise.ledger_run)
    entries = billed["models"].values()
    return {
        "model_key": inner.model.key,
        "model": inner.model.model,
        "pinned": inner.model.pinned,
        "served_models": sorted(listwise.served_models),
        "identity": listwise.identity,
        "prompt": inner.prompt.name,
        "prompt_sha256": inner.prompt.sha256,
        "words": inner.words,
        "calls": listwise.calls,
        "cache_hits": listwise.cache_hits,
        "input_tokens": sum(entry["input_tokens"] for entry in entries),
        "cached_input_tokens": sum(entry["cached_input_tokens"] for entry in entries),
        "output_tokens": sum(entry["output_tokens"] for entry in entries),
        "cost_usd": billed["total_usd"],
        "ledger_run_id": listwise.ledger_run,
    }


def build_manifest(
    experiment: Experiment,
    *,
    split: str,
    queries: Sequence[EvalQuery],
    sources: Iterable[str] = (),
    corpus: ReleaseRecord,
    releases: Mapping[str, Any],
    variants: Mapping[str, Any],
    search_config: SearchConfig,
    locked_test: bool,
    limited: int | None,
    gpu: Mapping[str, Any],
    created: datetime | None = None,
) -> dict[str, Any]:
    parsing = yaml.safe_load(Path("configs/parsing.yaml").read_text(encoding="utf-8")) or {}
    parser = (parsing.get("parser") or {}).get("parser_version")
    chunker = (parsing.get("chunker") or {}).get("chunker_version")
    if experiment.papers == "snapshot":
        # A benchmark corpus is packaged text: nothing was parsed or chunked.
        parser = chunker = "none: packaged title and abstract"
    created = created or datetime.now(UTC)
    metered = {
        name: float(entry["llm"]["cost_usd"])
        for name, entry in variants.items()
        if isinstance(entry, Mapping) and isinstance(entry.get("llm"), Mapping)
    }
    cost: dict[str, Any] = {
        "metered_usd": 0.0,
        "note": "self-hosted models and services; no priced model call is made",
    }
    if metered:
        cost = {
            "metered_usd": round(sum(metered.values()), 6),
            "by_variant": metered,
            "note": "hosted LLM calls as the spend ledger recorded them; cached answers cost "
            "nothing; every other model and service is self-hosted",
        }
    return {
        "run_id": f"{experiment.name}-{split}-{created:%Y%m%dT%H%M%SZ}",
        "created_at": created.isoformat(),
        "code": code_state(),
        "experiment": {
            "name": experiment.name,
            "path": str(experiment.path),
            "config_sha256": experiment.sha256,
        },
        "corpus": {
            "release_id": corpus.id,
            "manifest_sha256": corpus.manifest_sha256,
            "model_revision": corpus.model_revision,
            "parser_version": parser,
            "chunker_version": chunker,
            "papers": experiment.papers,
            "releases": dict(releases),
        },
        "dataset": {
            "path": str(experiment.dataset),
            "digest": dataset_digest(experiment.dataset),
            "split": split,
            "slice": experiment.slice,
            "labels": experiment.labels,
            "sources": sorted(sources),
            "queries": len(queries),
            "limited_to": limited,
        },
        "search": {
            "path": str(experiment.search),
            "config_sha256": _file_sha256(experiment.search),
            "values": search_values(search_config, None),
        },
        "variants": dict(variants),
        "seed": experiment.seed,
        "bootstrap": {
            "resamples": experiment.resamples,
            "seed": experiment.seed,
            "confidence": experiment.confidence,
            "unit": "query family",
        },
        "hardware": hardware(),
        "timing": {"methodology": TIMING_METHODOLOGY, "gpu": dict(gpu)},
        "cost": cost,
        "locked_test": locked_test,
    }


def write_run(
    out: Path,
    split: str,
    experiment: Experiment,
    manifest: Mapping[str, Any],
    results: Mapping[str, Sequence[QueryResult]],
) -> dict[str, Any]:
    """manifest.json, metrics.json and per_query.parquet under ``out/<split>/``."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_manifest(manifest)
    summaries = {name: summarize(rows, experiment) for name, rows in results.items()}
    metrics: dict[str, Any] = {
        "manifest": dict(manifest),
        "variants": {name: {"summary": summary} for name, summary in summaries.items()},
        "comparisons": comparisons(results, experiment),
    }
    if split == "validation":
        metrics["decision"] = decide(split, results, summaries, experiment)
    directory = out / split
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    rows = [result.row() for name in results for result in results[name]]
    pq.write_table(pa.Table.from_pylist(rows), directory / "per_query.parquet")
    return metrics


# --- The smoke set: synthetic, service-free, frozen ---------------------------------------


class InlineRunner:
    """Stages one after another, no threads and no deadlines: a deterministic runner."""

    def run(self, stages: Mapping[str, Stage]) -> dict[str, Outcome]:
        outcomes: dict[str, Outcome] = {}
        for name, (fn, _) in stages.items():
            try:
                outcomes[name] = Outcome(value=fn())
            except Exception as error:  # noqa: BLE001 - mirrors the threaded runner's typing
                outcomes[name] = Outcome(error=str(getattr(error, "code", type(error).__name__)))
        return outcomes


class MemoryPapers:
    def __init__(self, rows: Mapping[str, PaperRow]) -> None:
        self._rows = dict(rows)

    def load(self, paper_ids: Sequence[str]) -> dict[str, PaperRow]:
        return {paper_id: self._rows[paper_id] for paper_id in paper_ids if paper_id in self._rows}


class MemoryLexical:
    """The P2.1 BM25 oracle, exact and in memory."""

    def __init__(self, documents: Mapping[str, str]) -> None:
        from ..search.lexical import BM25

        self._bm25 = BM25()
        self._bm25.fit(documents)

    def search(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> list[tuple[str, float]]:
        return self._bm25.search(query, limit)


class MemoryDense:
    """Exhaustive cosine over fixture vectors; ties broken by id so the order is total."""

    def __init__(self, documents: Mapping[str, str], model: Any) -> None:
        self._model = model
        self._ids = sorted(documents)
        self._vectors = model.encode_array([documents[item] for item in self._ids]).vectors

    def search(
        self, query: str, filters: PaperFilters | None, limit: int, release_id: str
    ) -> list[tuple[str, float]]:
        vector = self._model.encode_array([query]).vectors[0]
        scores = self._vectors @ vector
        ranked = sorted(
            zip(self._ids, scores.tolist(), strict=True), key=lambda pair: (-pair[1], pair[0])
        )
        return [(item, float(value)) for item, value in ranked[:limit]]


SMOKE_EXPERIMENT_NAME = "retrieval-smoke"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def smoke_experiment(fixture: Path) -> Experiment:
    return Experiment(
        path=fixture,
        sha256="fixture",
        name=SMOKE_EXPERIMENT_NAME,
        dataset=fixture,
        query_text="",
        slice="all",
        labels="paper_id",
        release="smoke",
        papers="database",
        search=Path("configs/search.yaml"),
        models=Path("configs/models.yaml"),
        recall_at=(10, 50),
        ndcg_at=(10,),
        mrr_at=(10,),
        depth=50,
        resamples=1000,
        seed=42,
        confidence=0.95,
        variants=tuple(Variant(name=mode, mode=mode) for mode in BASELINE_MODES),
        decision={"primary": "ndcg@10", "order": list(BASELINE_MODES)},
    )


def run_smoke(fixture: Path) -> dict[str, Any]:
    """The four baselines over the synthetic fixture: fixture models, no services, no network."""

    from ..models.embeddings import FixtureEmbedding
    from ..search.lexical import paper_text
    from ..search.rerank import FixtureReranker

    papers = {row["paper_id"]: row for row in _jsonl(fixture / "papers.jsonl")}
    documents = {pid: paper_text(row["title"], row["abstract"]) for pid, row in papers.items()}
    rows = {
        pid: PaperRow(
            paper_id=pid,
            title=row["title"],
            abstract=row["abstract"],
            authors=(),
            venue=row["venue"],
            year=int(row["year"]),
            pdf_url=None,
            fulltext=False,
        )
        for pid, row in papers.items()
    }
    grades: dict[str, dict[str, int]] = {}
    for qrel in _jsonl(fixture / "qrels.jsonl"):
        grades.setdefault(qrel["query_id"], {})[qrel["paper_id"]] = int(qrel["grade"])
    queries = [
        EvalQuery(
            query_id=row["query_id"],
            family_id=row["family_id"],
            split=row["split"],
            query_set=row["query_set"],
            text=row["query"],
            grades=grades.get(row["query_id"], {}),
        )
        for row in sorted(_jsonl(fixture / "queries.jsonl"), key=lambda item: item["query_id"])
    ]
    model = FixtureEmbedding()
    release = ReleaseRecord(
        id="smoke",
        manifest_sha256="0" * 64,
        paper_collection="smoke_paper_abstracts",
        chunk_collection="smoke_paper_chunks",
        model_revision=model.identity,
        status="ready",
        counts={},
    )
    service = SearchService(
        engine=None,
        lexical=MemoryLexical(documents),
        dense=MemoryDense(documents, model),
        reranker=FixtureReranker(),
        config=SearchConfig(),
        runner=InlineRunner(),
        papers=MemoryPapers(rows),
        capture=lambda: release,
    )
    experiment = smoke_experiment(fixture)
    ticks = iter(range(10**9))
    output: dict[str, Any] = {}
    for variant in experiment.variants:
        results = [
            run_query(
                service, variant, query, release, experiment, clock=lambda: float(next(ticks))
            )
            for query in queries
            if query.relevant
        ]
        summary = summarize(results, experiment)
        rankings = [[result.query_id, result.ranked] for result in results]
        output[variant.name] = {
            "metrics": {name: round(summary[name]["mean"], 6) for name in metric_names(experiment)},
            "intervals": {
                name: [round(summary[name]["low"], 6), round(summary[name]["high"], 6)]
                for name in metric_names(experiment)
            },
            "queries": len(results),
            "failures": summary["failures"]["count"],
            "rankings_sha256": hashlib.sha256(json.dumps(rankings).encode()).hexdigest(),
        }
    return output


def check_smoke(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Spec §11's gate against the frozen outputs, and whether any ranking moved at all."""

    from .regression import GATED_METRICS, regressed

    regressions: list[dict[str, Any]] = []
    changed: list[str] = []
    for name, frozen in expected.items():
        current = actual.get(name)
        if current is None:
            regressions.append({"variant": name, "metric": "missing"})
            continue
        for metric in GATED_METRICS:
            before, after = float(frozen["metrics"][metric]), float(current["metrics"][metric])
            if regressed(before, after):
                regressions.append(
                    {"variant": name, "metric": metric, "frozen": before, "now": after}
                )
        if current["rankings_sha256"] != frozen["rankings_sha256"]:
            changed.append(name)
    return {"passed": not regressions, "regressions": regressions, "rankings_changed": changed}
