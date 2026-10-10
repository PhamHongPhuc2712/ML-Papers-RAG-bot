"""The answering run (ORB plan O5): retrieve, answer with citations, validate, judge.

Each case takes the shipped search's top papers, evidence inside them, the grounded
generator's claims (validated, one repair) and the judge's verdict. An abstention is
scored wrong without a judge call; a generation refused or timed out is a typed
failure, kept in every denominator; a judge error is excluded from the accuracy
denominator and counted. Per-case text (question, reference, answer, reason) is
written under ``DATA_DIR``, never under ``reports/``: the repository copy is ids,
labels, verdicts and costs.

Development cases tune prompts. The frozen cases are scored once per configuration,
only with ``frozen=True``, and the output directory keeps a registry of every
scoring. A run is refused before anything loads when its worst case passes the
budget, and stopped, keeping finished cases, once the ledger's spend for the run
passes it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ..contracts import Evidence, GroundedAnswer
from ..models.generation import GenerationOutcome
from ..models.llm import LlmError
from .judge import JudgeError, JudgeOutcome
from .pairing import wilson
from .retrieval import bootstrap_interval, code_state, dataset_digest, gpu_before_run, hardware

QA_SMOKE = Path("data/fixtures/qa-smoke")
RUNNABLE_SPLITS = ("development", "frozen")
SMOKE_SPLIT = "smoke"
# The worst case of one generation prompt, in characters: the evidence budget in tokens
# at four characters each, plus the instructions and the question.
PROMPT_CHARS_PER_TOKEN = 4
GENERATION_OVERHEAD_CHARS = 2_000
JUDGE_PROMPT_CHARS = 4_000


class QaError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


# --- Configuration ------------------------------------------------------------------------


@dataclass(frozen=True)
class QaVariant:
    name: str
    evidence: dict[str, Any] = field(default_factory=dict)
    changes: str | None = None


@dataclass(frozen=True)
class QaExperiment:
    path: Path
    sha256: str
    name: str
    dataset: Path
    benchmark: str
    release: str
    search: Path
    models: Path
    evidence: Path
    generator: str
    judge: str
    generate_prompt: Path
    judge_prompt: Path
    papers: int
    resamples: int
    seed: int
    confidence: float
    variants: tuple[QaVariant, ...]
    budget_usd: float
    decision: Mapping[str, Any]
    # (SearchConfig field, value) pairs applied over the shared search file for the whole
    # experiment: the first stage that hands papers to the evidence retriever. Parsed by the
    # retrieval harness's own rule, so only a knob it may turn is accepted.
    search_overrides: tuple[tuple[str, Any], ...] = ()

    def variant(self, name: str) -> QaVariant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        raise QaError("unknown_variant", name)


def _qa_search_overrides(raw: object) -> tuple[tuple[str, Any], ...]:
    from .retrieval import ExperimentError, _search_overrides

    try:
        return _search_overrides("search_overrides", "hybrid_rerank", raw)
    except ExperimentError as error:
        raise QaError("experiment_invalid", str(error)) from error


def qa_search_config(experiment: QaExperiment) -> Any:
    """The shared search file with the experiment's overrides, checked as the harness does."""

    from dataclasses import replace

    from ..search.service import load_search_config

    config = replace(load_search_config(experiment.search), **dict(experiment.search_overrides))
    if config.rerank_seconds > config.total_seconds:
        raise QaError("experiment_invalid", "rerank deadline exceeds the total")
    return config


def load_qa_experiment(path: str | Path) -> QaExperiment:
    target = Path(path)
    raw_bytes = target.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    try:
        variants = tuple(
            QaVariant(
                name=str(item["name"]),
                evidence=dict(item.get("evidence") or {}),
                changes=item.get("changes"),
            )
            for item in raw["variants"]
        )
        bootstrap = raw.get("bootstrap") or {}
        experiment = QaExperiment(
            path=target,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            name=str(raw["name"]),
            dataset=Path(raw["dataset"]["path"]),
            benchmark=str(raw["dataset"].get("benchmark", "orb")),
            release=str(raw["corpus"]["release"]),
            search=Path(raw["search"]),
            models=Path(raw["models"]),
            evidence=Path(raw["evidence"]),
            generator=str(raw["generator"]),
            judge=str(raw["judge"]),
            generate_prompt=Path(raw["prompts"]["generate"]),
            judge_prompt=Path(raw["prompts"]["judge"]),
            papers=int(raw.get("papers", 10)),
            resamples=int(bootstrap.get("resamples", 1000)),
            seed=int(bootstrap.get("seed", 42)),
            confidence=float(bootstrap.get("confidence", 0.95)),
            variants=variants,
            budget_usd=float((raw.get("budget") or {}).get("max_spend_usd", 0.0)),
            decision=dict(raw.get("decision") or {}),
            search_overrides=_qa_search_overrides(raw.get("search_overrides")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise QaError("experiment_invalid", str(error)) from error
    names = [variant.name for variant in experiment.variants]
    if not names or len(names) != len(set(names)):
        raise QaError("experiment_invalid", "variant names must be unique")
    if experiment.benchmark != "orb":
        raise QaError("experiment_invalid", "dataset.benchmark")
    return experiment


# --- Cases ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class QaCase:
    query_id: str
    family_id: str
    type: str
    question: str
    reference: str
    gold_paper_id: str | None
    # Fixture cases carry their top papers; live cases get them from the search service.
    papers: tuple[str, ...] | None = None


class Pipeline(Protocol):
    """What finds papers and evidence for a case: the live services, or fixtures."""

    def papers(self, case: QaCase) -> list[str]: ...

    def evidence(
        self, case: QaCase, papers: Sequence[str], variant: QaVariant
    ) -> list[Evidence]: ...


class GeneratorLike(Protocol):
    identity: str

    def generate(
        self,
        question: str,
        evidence: Sequence[Evidence],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> GenerationOutcome: ...


class JudgeLike(Protocol):
    identity: str

    def judge(
        self,
        question: str,
        reference: str,
        answer: str,
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> JudgeOutcome: ...


@dataclass
class CaseResult:
    variant: str
    query_id: str
    family_id: str
    type: str
    papers: list[str]
    gold_in_papers: bool | None
    evidence_count: int
    evidence_papers: list[str]
    cited_papers: list[str]
    cited_map: float | None
    provenance_valid: bool
    repair_attempted: bool
    abstained: bool
    equivalent: bool | None
    judge_error: str | None
    failure: str | None
    generation_calls: int
    judge_calls: int
    generation_cost_usd: float
    judge_cost_usd: float
    served_models: list[str]
    seconds: float
    problems: list[str] = field(default_factory=list)
    # Text, for the copy under DATA_DIR only.
    question: str = ""
    reference: str = ""
    answer: str = ""
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        """The repository's copy: no question, reference, answer or reason text."""

        return {
            "variant": self.variant,
            "query_id": self.query_id,
            "family_id": self.family_id,
            "type": self.type,
            "papers": self.papers,
            "gold_in_papers": self.gold_in_papers,
            "evidence_count": self.evidence_count,
            "evidence_papers": self.evidence_papers,
            "cited_papers": self.cited_papers,
            "cited_map": self.cited_map,
            "provenance_valid": self.provenance_valid,
            "repair_attempted": self.repair_attempted,
            "abstained": self.abstained,
            "equivalent": self.equivalent,
            "judge_error": self.judge_error,
            "failure": self.failure,
            "generation_calls": self.generation_calls,
            "judge_calls": self.judge_calls,
            "generation_cost_usd": self.generation_cost_usd,
            "judge_cost_usd": self.judge_cost_usd,
            "served_models": self.served_models,
            "seconds": self.seconds,
            "problems": self.problems,
            "evidence_ids": self.evidence_ids,
        }

    def text_row(self) -> dict[str, Any]:
        return {
            **self.row(),
            "question": self.question,
            "reference": self.reference,
            "answer": self.answer,
            "reason": self.reason,
        }


def cited_map(cited_papers: Sequence[str], gold_paper_id: str | None) -> float | None:
    """Average precision of the cited papers against the one gold paper: 1/rank or 0."""

    if gold_paper_id is None:
        return None
    for rank, paper_id in enumerate(cited_papers, 1):
        if paper_id == gold_paper_id:
            return 1.0 / rank
    return 0.0


def answer_text(answer: GroundedAnswer) -> str:
    return " ".join(claim.text.strip() for claim in answer.claims if claim.text.strip())


def answer_case(
    case: QaCase,
    pipeline: Pipeline,
    generator: GeneratorLike,
    judge: JudgeLike,
    variant: QaVariant,
    *,
    run_id: str,
    clock: Callable[[], float] = time.monotonic,
) -> CaseResult:
    started = clock()
    papers = list(case.papers) if case.papers is not None else pipeline.papers(case)
    result = CaseResult(
        variant=variant.name,
        query_id=case.query_id,
        family_id=case.family_id,
        type=case.type,
        papers=papers,
        gold_in_papers=(case.gold_paper_id in papers) if case.gold_paper_id else None,
        evidence_count=0,
        evidence_papers=[],
        cited_papers=[],
        cited_map=None,
        provenance_valid=False,
        repair_attempted=False,
        abstained=False,
        equivalent=None,
        judge_error=None,
        failure=None,
        generation_calls=0,
        judge_calls=0,
        generation_cost_usd=0.0,
        judge_cost_usd=0.0,
        served_models=[],
        seconds=0.0,
        question=case.question,
        reference=case.reference,
    )
    evidence = pipeline.evidence(case, papers, variant)
    by_id = {item.id: item for item in evidence}
    result.evidence_count = len(evidence)
    result.evidence_papers = list(dict.fromkeys(str(item.source_id) for item in evidence))
    result.evidence_ids = [item.id for item in evidence]
    try:
        outcome = generator.generate(case.question, evidence, request_id=uuid4(), run_id=run_id)
    except LlmError as error:
        # A refused or timed-out generation is a typed failure, never an empty answer.
        result.failure = f"generation:{error.code}"
        result.seconds = clock() - started
        return result
    result.generation_calls = outcome.calls
    result.generation_cost_usd = outcome.cost_usd
    result.served_models = list(outcome.served_models)
    result.repair_attempted = outcome.repair_attempted
    result.provenance_valid = outcome.provenance_valid
    result.problems = list(outcome.problems)
    cited: list[str] = []
    for claim in outcome.answer.claims:
        for label in claim.evidence_ids:
            item = by_id.get(label)
            if item is not None:
                cited.append(str(item.source_id))
    result.cited_papers = list(dict.fromkeys(cited))
    result.cited_map = cited_map(result.cited_papers, case.gold_paper_id)
    result.answer = answer_text(outcome.answer)
    if outcome.answer.insufficient_evidence and not outcome.answer.claims:
        # An abstention is wrong on a benchmark where every question has an answer;
        # the judge is not asked.
        result.abstained = True
        result.equivalent = False
        result.seconds = clock() - started
        return result
    try:
        verdict = judge.judge(
            case.question, case.reference, result.answer, request_id=uuid4(), run_id=run_id
        )
    except JudgeError as error:
        result.judge_error = error.code
    except LlmError as error:
        result.judge_error = f"judge:{error.code}"
    else:
        result.judge_calls = verdict.calls
        result.judge_cost_usd = verdict.cost_usd
        result.equivalent = verdict.verdict.equivalent
        result.reason = verdict.verdict.reason
        if verdict.served_model:
            result.served_models.append(verdict.served_model)
    result.seconds = clock() - started
    return result


# --- Aggregation --------------------------------------------------------------------------


def _family_means(
    rows: Sequence[Mapping[str, Any]], value: Callable[[Mapping[str, Any]], float | None]
) -> np.ndarray:
    groups: dict[str, list[float]] = {}
    for row in rows:
        number = value(row)
        if number is None:
            continue
        groups.setdefault(str(row["family_id"]), []).append(float(number))
    return np.array([float(np.mean(values)) for _, values in sorted(groups.items())])


def summarize_qa(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int, confidence: float
) -> dict[str, Any]:
    """Accuracy over judged cases with Wilson and family-bootstrap intervals, and the rest."""

    cases = len(rows)
    judged = [row for row in rows if isinstance(row.get("equivalent"), bool)]
    correct = sum(1 for row in judged if row["equivalent"])
    accuracy = correct / len(judged) if judged else 0.0
    families = _family_means(judged, lambda row: 1.0 if row["equivalent"] else 0.0)
    maps = [float(row["cited_map"]) for row in rows if row.get("cited_map") is not None]
    by_type: dict[str, dict[str, Any]] = {}
    for kind in sorted({str(row["type"]) for row in rows}):
        subset = [row for row in judged if row["type"] == kind]
        by_type[kind] = {
            "cases": sum(1 for row in rows if row["type"] == kind),
            "judged": len(subset),
            "accuracy": (
                sum(1 for row in subset if row["equivalent"]) / len(subset) if subset else None
            ),
        }
    return {
        "cases": cases,
        "judged": len(judged),
        "correct": correct,
        "accuracy": {
            "mean": accuracy,
            "wilson": list(wilson(correct, len(judged))),
            "interval": list(
                bootstrap_interval(families, resamples=resamples, seed=seed, confidence=confidence)
            ),
            "families": int(len(families)),
        },
        "cited_map": {
            "mean": float(np.mean(maps)) if maps else None,
            "cases": len(maps),
            "note": "every case has one gold paper; MAP is 1/rank of the gold among cited",
        },
        "provenance_validity": (
            sum(1 for row in rows if row.get("provenance_valid")) / cases if cases else 0.0
        ),
        "abstention_rate": (
            sum(1 for row in rows if row.get("abstained")) / cases if cases else 0.0
        ),
        "repairs": sum(1 for row in rows if row.get("repair_attempted")),
        "judge_errors": sum(1 for row in rows if row.get("judge_error")),
        "failures": sum(1 for row in rows if row.get("failure")),
        "gold_in_papers": (
            sum(1 for row in rows if row.get("gold_in_papers")) / cases if cases else 0.0
        ),
        "by_type": by_type,
        "cost_usd": float(
            sum(
                float(row.get("generation_cost_usd") or 0.0)
                + float(row.get("judge_cost_usd") or 0.0)
                for row in rows
            )
        ),
    }


# --- Outputs ------------------------------------------------------------------------------


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)


def write_qa_outputs(
    out: Path,
    split: str,
    run_id: str,
    *,
    results: Mapping[str, Sequence[CaseResult]],
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    text_dir: Path,
) -> dict[str, Path]:
    """The repository's text-free outputs, and the per-case text under ``text_dir``."""

    written: dict[str, Path] = {}
    for name, rows in results.items():
        directory = out / split / f"{run_id}.{name}"
        directory.mkdir(parents=True, exist_ok=True)
        _write_parquet(directory / "cases.parquet", [row.row() for row in rows])
        (directory / "metrics.json").write_text(
            json.dumps(metrics[name], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / "manifest.json").write_text(
            json.dumps({**manifest, "variant": name}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        text_path = text_dir / run_id / name / "cases.jsonl"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(
            "".join(json.dumps(row.text_row(), sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        written[name] = directory
    return written


def record_frozen_scoring(out: Path, run_id: str, variants: Sequence[str]) -> dict[str, Any]:
    """The registry of every scoring of the frozen split, per configuration."""

    path = out / "frozen-scorings.json"
    registry: dict[str, list[str]] = {}
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    for name in variants:
        registry.setdefault(name, []).append(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {name: len(runs) for name, runs in registry.items()}


# --- The smoke set ------------------------------------------------------------------------


class FixturePipeline:
    """Top papers and evidence read from the smoke fixtures."""

    def __init__(self, evidence: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._evidence = evidence

    def papers(self, case: QaCase) -> list[str]:
        return list(case.papers or [])

    def evidence(self, case: QaCase, papers: Sequence[str], variant: QaVariant) -> list[Evidence]:
        return [
            Evidence(
                id=str(item["id"]),
                source_kind="paper",
                source_id=UUID(str(item["paper_id"])),
                version_id="fixture",
                chunk_id=UUID(str(item["chunk_id"])),
                text=str(item["text"]),
                section=str(item.get("section", "Method")),
            )
            for item in self._evidence.get(case.query_id, [])
        ]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def run_qa_smoke(
    fixture: Path = QA_SMOKE, *, out: Path | None = None, update: bool = False
) -> dict[str, Any]:
    """The answering loop over synthetic cases with recorded verdicts: CI's check."""

    from ..models.generation import FakeGenerator
    from .judge import RecordedJudge

    cases = [
        QaCase(
            query_id=str(row["query_id"]),
            family_id=str(row["family_id"]),
            type=str(row["type"]),
            question=str(row["question"]),
            reference=str(row["reference"]),
            gold_paper_id=str(row["gold_paper_id"]),
            papers=tuple(str(paper) for paper in row["papers"]),
        )
        for row in _load_jsonl(fixture / "cases.jsonl")
    ]
    evidence = {
        str(row["query_id"]): list(row["evidence"])
        for row in _load_jsonl(fixture / "evidence.jsonl")
    }
    responses = json.loads((fixture / "judge-responses.json").read_text(encoding="utf-8"))
    pipeline = FixturePipeline(evidence)
    generator = FakeGenerator()
    judge = RecordedJudge(responses)
    variant = QaVariant(name="smoke")
    ticks = iter(range(10**9))
    results = [
        answer_case(
            case,
            pipeline,
            generator,
            judge,
            variant,
            run_id="qa-smoke",
            clock=lambda: float(next(ticks)),
        )
        for case in cases
    ]
    rows = [result.row() for result in results]
    metrics = summarize_qa(rows, resamples=200, seed=42, confidence=0.95)
    digest = hashlib.sha256(
        json.dumps([[r.query_id, r.equivalent, r.cited_map, r.abstained] for r in results]).encode()
    ).hexdigest()
    output = {"metrics": metrics, "cases_sha256": digest}
    if out is not None:
        write_qa_outputs(
            out,
            SMOKE_SPLIT,
            "qa-smoke",
            results={"smoke": results},
            metrics={"smoke": metrics},
            manifest={"run_id": "qa-smoke", "fixture": str(fixture)},
            text_dir=out / "text",
        )
        # The test reads the smoke split's parquet at a fixed path.
        (out / SMOKE_SPLIT / "cases.parquet").parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(out / SMOKE_SPLIT / "cases.parquet", rows)
        (out / "text" / "cases.jsonl").write_text(
            "".join(json.dumps(r.text_row(), sort_keys=True) + "\n" for r in results),
            encoding="utf-8",
        )
    if update:
        (fixture / "expected.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return output


def check_qa_smoke(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Byte-identical aggregates and per-case outcomes against the frozen outputs."""

    problems: list[str] = []
    if actual.get("cases_sha256") != expected.get("cases_sha256"):
        problems.append("cases_changed")
    for key in (
        "cases",
        "judged",
        "correct",
        "judge_errors",
        "abstention_rate",
        "provenance_validity",
    ):
        if actual["metrics"].get(key) != expected["metrics"].get(key):
            problems.append(f"metrics.{key}")
    if actual["metrics"]["accuracy"] != expected["metrics"]["accuracy"]:
        problems.append("metrics.accuracy")
    return {"passed": not problems, "problems": problems}


# --- The live run -------------------------------------------------------------------------


class LivePipeline:
    """The shipped search for papers, the evidence retriever inside them."""

    def __init__(
        self,
        *,
        service: Any,
        release: Any,
        retrievers: Mapping[str, Any],
        papers: int,
    ) -> None:
        self._service = service
        self._release = release
        self._retrievers = retrievers
        self._papers = papers

    def papers(self, case: QaCase) -> list[str]:
        from ..contracts import PaperFilters, SearchRequest

        request = SearchRequest.model_validate(
            {
                "query": case.question,
                "mode": "hybrid_rerank",
                "filters": PaperFilters(),
                "limit": self._papers,
            }
        )
        ranking = self._service.rank(request, release=self._release)
        return [paper_id for paper_id, _ in ranking.ordering.items[: self._papers]]

    def evidence(self, case: QaCase, papers: Sequence[str], variant: QaVariant) -> list[Evidence]:
        return list(
            self._retrievers[variant.name].retrieve(case.question, papers, self._release.id)
        )


def load_qa_cases(
    experiment: QaExperiment, split: str, data_dir: Path, engine: Any
) -> list[QaCase]:
    """ORB's questions, references and gold papers for one answering split, in id order."""

    from .orb import OrbPaths, load_orb, paper_ids_for, read_qa_assignment, text_slice

    paths = OrbPaths(data_dir / "benchmarks" / "orb")
    assignment = read_qa_assignment(experiment.dataset)
    queries = [q for q in text_slice(load_orb(paths)) if assignment.get(q.query_id) == split]
    papers = paper_ids_for({q.doc_id for q in queries}, engine)
    return [
        QaCase(
            query_id=q.query_id,
            family_id=q.doc_id,
            type=q.type,
            question=q.query,
            reference=q.answer,
            gold_paper_id=papers.get(q.doc_id),
        )
        for q in sorted(queries, key=lambda q: q.query_id)
    ]


def qa_preflight_usd(
    experiment: QaExperiment, *, cases: int, variants: int, evidence_tokens: int
) -> float:
    """What a run could spend: two generation and two judge calls per case, at worst."""

    from ..models.spend import estimate_usd, load_llm_config

    config = load_llm_config("configs/llm.yaml")
    generator = config.models[experiment.generator]
    judge = config.models[experiment.judge]
    generation_chars = evidence_tokens * PROMPT_CHARS_PER_TOKEN + GENERATION_OVERHEAD_CHARS
    per_case = 2 * estimate_usd(
        generator.price, generation_chars, generator.max_output_tokens
    ) + 2 * estimate_usd(judge.price, JUDGE_PROMPT_CHARS, judge.max_output_tokens)
    return cases * variants * per_case


def run_qa(
    config: str | Path,
    split: str,
    out: str | Path,
    *,
    max_spend_usd: float,
    frozen: bool = False,
    limit: int | None = None,
    variants: Sequence[str] | None = None,
    data_dir: Path | None = None,
    database_url: str | None = None,
    qdrant_url: str | None = None,
    text_dir: Path | None = None,
) -> dict[str, Any]:
    """Answer and judge one split under a budget; see the module docstring."""

    experiment = load_qa_experiment(config)
    if split not in RUNNABLE_SPLITS:
        raise QaError("unknown_split", split)
    if split == "frozen" and not frozen:
        raise QaError("frozen_flag_required", "the frozen split is scored once per configuration")
    chosen = (
        [experiment.variant(name) for name in variants] if variants else list(experiment.variants)
    )
    out_dir = Path(out)

    from ..config import Settings
    from ..corpus.releases import load_release
    from ..db.session import make_engine
    from ..evidence.retrieve import EvidenceRetriever, load_evidence_config
    from ..models.generation import ChatGenerator, load_generation_prompt
    from ..models.llm import ResponseCache, build_client
    from ..models.spend import SpendLedger
    from ..search.chunks import dense_branch, lexical_branch
    from ..search.client import qdrant_client
    from ..search.service import SearchService
    from .judge import AnswerEquivalenceJudge, load_judge_prompt
    from .retrieval import PinnedModels, search_digest

    settings = Settings()  # type: ignore[call-arg]
    data_dir = data_dir or Path(settings.data_dir)
    text_dir = text_dir or data_dir / "eval" / "qa"
    engine = make_engine(database_url or settings.database_url)
    cases = load_qa_cases(experiment, split, data_dir, engine)
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise QaError("no_cases", split)
    evidence_config = load_evidence_config(experiment.evidence)
    worst = qa_preflight_usd(
        experiment,
        cases=len(cases),
        variants=len(chosen),
        evidence_tokens=evidence_config.token_budget,
    )
    if worst > max_spend_usd:
        raise QaError(
            "run_budget_exceeded",
            f"worst case ${worst:.4f} for {len(cases)} cases x {len(chosen)} variants"
            f" > budget ${max_spend_usd:.4f}",
        )
    created = datetime.now(UTC)
    run_id = f"{experiment.name}-{split}-{created:%Y%m%dT%H%M%SZ}"

    before = gpu_before_run()
    client = qdrant_client(qdrant_url or settings.qdrant_url, timeout=60)
    release = load_release(engine, experiment.release)
    factory = PinnedModels(data_dir)
    embedder, max_tokens = factory.embedder(experiment.models)
    reranker = factory.reranker(experiment.models)
    search_config = qa_search_config(experiment)
    service = SearchService(
        engine=engine,
        lexical=lexical_branch(engine, client, source="chunks", k=search_config.rrf_k),
        dense=dense_branch(
            engine, client, embedder, source="chunks", max_tokens=max_tokens, k=search_config.rrf_k
        ),
        reranker=reranker,
        config=search_config,
    )
    service.warm(release)
    retrievers = {
        variant.name: EvidenceRetriever(
            engine=engine,
            client=client,
            embedder=embedder,
            reranker=reranker,
            config=evidence_config.replace(**variant.evidence),
            data_dir=data_dir,
            max_tokens=max_tokens,
        )
        for variant in chosen
    }
    pipeline = LivePipeline(
        service=service, release=release, retrievers=retrievers, papers=experiment.papers
    )
    # The ledger of record is the main database's, so one daily cap covers every run.
    ledger = SpendLedger(make_engine(settings.database_url), settings.llm_daily_spend_cap_usd)
    cache = ResponseCache(data_dir / "cache" / "llm")
    generator = ChatGenerator(
        build_client(experiment.generator, settings, ledger, cache=cache),
        load_generation_prompt(experiment.generate_prompt),
    )
    judge = AnswerEquivalenceJudge(
        build_client(experiment.judge, settings, ledger, cache=cache),
        load_judge_prompt(experiment.judge_prompt),
    )

    results: dict[str, list[CaseResult]] = {}
    metrics: dict[str, Any] = {}
    stopped: dict[str, str] = {}
    ledger_runs: dict[str, str] = {}
    try:
        for variant in chosen:
            ledger_run = f"{run_id}.{variant.name}.{uuid4().hex[:8]}"
            ledger_runs[variant.name] = ledger_run
            rows: list[CaseResult] = []
            for case in cases:
                rows.append(
                    answer_case(case, pipeline, generator, judge, variant, run_id=ledger_run)
                )
                spent = sum(ledger.run_spend(run) for run in ledger_runs.values())
                if spent > max_spend_usd:
                    stopped[variant.name] = f"budget: ${spent:.4f} > ${max_spend_usd:.4f}"
                    break
            results[variant.name] = rows
            metrics[variant.name] = summarize_qa(
                [row.row() for row in rows],
                resamples=experiment.resamples,
                seed=experiment.seed,
                confidence=experiment.confidence,
            )
            metrics[variant.name]["ledger_cost_usd"] = ledger.run_spend(ledger_run)
            if variant.name in stopped:
                break
    finally:
        generator.close()
        judge.close()
        service.close()
        client.close()

    scorings = (
        record_frozen_scoring(out_dir, run_id, [v.name for v in chosen])
        if split == "frozen"
        else {}
    )
    manifest = {
        "run_id": run_id,
        "created_at": created.isoformat(),
        "experiment": {
            "path": str(experiment.path),
            "sha256": experiment.sha256,
            "name": experiment.name,
        },
        "split": split,
        "cases": len(cases),
        "limit": limit,
        "dataset": {"path": str(experiment.dataset), "sha256": dataset_digest(experiment.dataset)},
        "corpus": {
            "release": release.id,
            "manifest_sha256": release.manifest_sha256,
            "embedding": release.model_revision,
        },
        "search": {
            "path": str(experiment.search),
            "sha256": search_digest(search_config, "hybrid_rerank", "chunks"),
            "overrides": dict(experiment.search_overrides),
            "papers": experiment.papers,
            "reranker": str(getattr(reranker, "identity", "none")),
        },
        "evidence": {name: retriever.config.identity() for name, retriever in retrievers.items()},
        "generator": {"key": experiment.generator, "identity": generator.identity},
        "judge": {"key": experiment.judge, "identity": judge.identity},
        "prompts": {
            "generate": {
                "path": str(experiment.generate_prompt),
                "sha256": generator.prompt.sha256,
            },
            "judge": {"path": str(experiment.judge_prompt), "sha256": judge.prompt.sha256},
        },
        "variants": {
            v.name: {
                "evidence_overrides": v.evidence,
                "changes": v.changes,
                "ledger_run_id": ledger_runs.get(v.name),
            }
            for v in chosen
        },
        "budget_usd": max_spend_usd,
        "preflight_worst_usd": worst,
        "stopped": stopped,
        "frozen_scorings": scorings,
        "seed": experiment.seed,
        "code": code_state(),
        "hardware": hardware(),
        "gpu_before_run": before,
        "text_dir": str(text_dir),
    }
    written = write_qa_outputs(
        out_dir,
        split,
        run_id,
        results=results,
        metrics=metrics,
        manifest=manifest,
        text_dir=text_dir,
    )
    return {
        "run_id": run_id,
        "written": {name: str(path) for name, path in written.items()},
        "metrics": {
            name: {
                "accuracy": round(m["accuracy"]["mean"], 4),
                "judged": m["judged"],
                "cited_map": m["cited_map"]["mean"],
                "provenance_validity": round(m["provenance_validity"], 4),
                "abstention_rate": round(m["abstention_rate"], 4),
                "cost_usd": round(m["ledger_cost_usd"], 6),
            }
            for name, m in metrics.items()
        },
        "stopped": stopped,
    }


# --- The labelling sheet ------------------------------------------------------------------


def run_second_judge(
    run_dir: str | Path,
    *,
    judge_key: str,
    out: Path,
    max_spend_usd: float,
    limit: int | None = None,
    judge: Any | None = None,
    ledger: Any | None = None,
    data_dir: Path | None = None,
    prompt: str | Path | None = None,
) -> dict[str, Any]:
    """Re-judge a run's judged answers with a second model and measure agreement.

    ``prompt`` names another rubric; with it the judge of record may re-read its own
    verdicts under that rubric, and the agreement then measures the rubric, not the model.

    The developer's substitute (2026-10-09) for the 50 blind hand labels of O5 step 7,
    haiku.rag's method: an independent judge model reads the same question, reference
    and answer under the same rubric, and its verdicts stand in for the labels in
    ``calibration``. It measures model-model agreement, which is weaker evidence than a
    human's, and the report says so. Abstentions were never judged by the first judge and
    are not judged here. Every call goes through the ledger under the daily cap; the run
    is refused when its worst case exceeds ``max_spend_usd`` and stopped once its ledger
    spend does. The verdict sheet carries benchmark text and lives under ``DATA_DIR``;
    the returned summary carries none.
    """

    from ..models.spend import estimate_usd, load_llm_config
    from .judge import JudgeError
    from .pairing import calibration

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    prompt_path = Path(prompt) if prompt else Path(manifest["prompts"]["judge"]["path"])
    if manifest["judge"]["key"] == judge_key and prompt is None:
        raise QaError("same_judge", judge_key)
    text_path = (
        Path(manifest["text_dir"]) / manifest["run_id"] / manifest["variant"] / "cases.jsonl"
    )
    rows = [
        row
        for row in _load_jsonl(text_path)
        if isinstance(row.get("equivalent"), bool) and not row.get("abstained") and row["answer"]
    ]
    rows.sort(key=lambda row: str(row["query_id"]))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise QaError("no_cases", str(text_path))
    model = load_llm_config("configs/llm.yaml").models[judge_key]
    worst = len(rows) * 2 * estimate_usd(model.price, JUDGE_PROMPT_CHARS, model.max_output_tokens)
    if worst > max_spend_usd:
        raise QaError(
            "run_budget_exceeded",
            f"worst case ${worst:.4f} for {len(rows)} cases > budget ${max_spend_usd:.4f}",
        )

    if judge is None or ledger is None:
        from ..config import Settings
        from ..db.session import make_engine
        from ..models.llm import ResponseCache, build_client
        from ..models.spend import SpendLedger
        from .judge import AnswerEquivalenceJudge, load_judge_prompt

        settings = Settings()  # type: ignore[call-arg]
        data_dir = data_dir or Path(settings.data_dir)
        ledger = SpendLedger(make_engine(settings.database_url), settings.llm_daily_spend_cap_usd)
        judge = AnswerEquivalenceJudge(
            build_client(
                judge_key, settings, ledger, cache=ResponseCache(data_dir / "cache" / "llm")
            ),
            load_judge_prompt(prompt_path),
        )
    created = datetime.now(UTC)
    ledger_run = f"{manifest['run_id']}.{manifest['variant']}.rejudge.{uuid4().hex[:8]}"
    tag = judge_key if prompt is None else f"{judge_key}.{prompt_path.stem}"
    verdicts: list[dict[str, Any]] = []
    errors = 0
    stopped: str | None = None
    try:
        for row in rows:
            record: dict[str, Any] = {
                "query_id": row["query_id"],
                "question": row["question"],
                "reference": row["reference"],
                "answer": row["answer"],
                "equivalent": None,
                "reason": None,
                "judge_error": None,
                "served_model": "",
                "calls": 0,
                "cost_usd": 0.0,
            }
            try:
                outcome = judge.judge(
                    row["question"], row["reference"], row["answer"], run_id=ledger_run
                )
                record.update(
                    equivalent=outcome.verdict.equivalent,
                    reason=outcome.verdict.reason,
                    served_model=outcome.served_model,
                    calls=outcome.calls,
                    cost_usd=outcome.cost_usd,
                )
            except JudgeError as error:
                errors += 1
                record["judge_error"] = error.code
            verdicts.append(record)
            spent = ledger.run_spend(ledger_run)
            if spent > max_spend_usd:
                stopped = f"budget: ${spent:.4f} > ${max_spend_usd:.4f}"
                break
    finally:
        judge.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in verdicts),
        encoding="utf-8",
    )
    agreement = calibration(out, run_dir / "cases.parquet")
    summary = {
        "run_id": manifest["run_id"],
        "variant": manifest["variant"],
        "created_at": created.isoformat(),
        "judge_of_record": manifest["judge"],
        "second_judge": {
            "key": judge_key,
            "identity": getattr(judge, "identity", judge_key),
            "prompt": str(prompt_path),
        },
        "method": "model-model agreement, the developer's substitute for 50 blind hand labels; "
        "weaker evidence than a human's labels",
        "cases": len(rows),
        "judged": len(verdicts) - errors,
        "judge_errors": errors,
        "stopped": stopped,
        "ledger_run": ledger_run,
        "cost_usd": ledger.run_spend(ledger_run),
        "served_models": sorted({v["served_model"] for v in verdicts if v["served_model"]}),
        "sheet": str(out),
        "agreement": {k: v for k, v in agreement.items() if k != "disagreements"},
        "disagreements": agreement["disagreements"],
    }
    (run_dir / f"second-judge.{tag}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def write_label_sheet(
    run_dir: str | Path, *, out: Path, limit: int | None = None, seed: int = 42
) -> dict[str, Any]:
    """A blind labelling sheet for a run: question, reference and answer, no verdict.

    Reads the run's text copy (named by its manifest), takes every judged case in a
    seeded shuffle, and writes one JSON line per case for a human to label with the
    rubric in ``docs/judge-rubric.md``. The judge's verdict and reason are left out on
    purpose: the labels must be blind. The sheet carries benchmark text and lives under
    ``DATA_DIR``, never in the repository.
    """

    import random

    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    text_path = (
        Path(manifest["text_dir"]) / manifest["run_id"] / manifest["variant"] / "cases.jsonl"
    )
    rows = [row for row in _load_jsonl(text_path) if isinstance(row.get("equivalent"), bool)]
    rows.sort(key=lambda row: str(row["query_id"]))
    random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "reference": row["reference"],
                    "answer": row["answer"] or "(abstained: insufficient evidence)",
                    "equivalent": None,
                },
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return {"sheet": str(out), "cases": len(rows), "run_id": manifest["run_id"]}
