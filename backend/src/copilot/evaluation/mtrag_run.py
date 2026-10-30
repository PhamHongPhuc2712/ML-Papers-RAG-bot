"""MTRAG ClapNQ runs (agent experiment): passage retrieval, and answering in two modes.

Retrieval (`eval mtrag-retrieval`) searches each of the 208 judged turns once with the shipped
first stage, by one query source: the last user turn as written, the person's standalone
rewrite, or our condenser's rewrite of the last turn after the reference conversation. Its top
10 papers are its top 10 passages, scored as IBM scores them. It reads every judged turn and
decides nothing.

Answering (`eval mtrag`) runs both systems, the shipped pipeline and the research agent,
through one adapter each and two modes:

* ``live``: each conversation replayed turn by turn, each system seeing the people's questions
  and its *own* earlier answers (the agent also the passages those answers cited);
* ``gold_prefix``: each turn on its own after the reference conversation.

A turn is correct when the judge says its answer matches the reference, except on an
``UNANSWERABLE`` turn, which is correct exactly when the system abstained (no judge call).
An abstention on any other turn is wrong. A turn that fails (a spend stop, a timeout, a search
outage) is recorded with its code, counted wrong, and in ``live`` mode shown to the next turn as
an unanswered one. Per-turn text goes under ``DATA_DIR``; the repository keeps ids and metrics.
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
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ..chat.turns import INSUFFICIENT_TEXT, HistoryItem, answer_text
from ..contracts import Evidence
from ..evidence.retrieve import EvidenceError
from ..models.generation import GenerationOutcome
from ..models.llm import LlmError
from ..search.service import SearchUnavailable
from . import mtrag
from .faithfulness import faithfulness_metrics, paired_rate_difference
from .judge import JudgeError
from .pairing import mcnemar_exact, wilson
from .retrieval import bootstrap_interval, code_state, gpu_before_run, hardware

QUERY_SOURCES = ("lastturn", "rewrite", "condensed")
MODES = ("live", "gold_prefix")
SPLITS = ("development", "test", "all")
UNANSWERABLE = "UNANSWERABLE"
# Passages carried into an agent's follow-up: the newest cited first, at most this many.
CARRIED_MAX = 12
FAILURES = (LlmError, SearchUnavailable, EvidenceError)


class MtragRunError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


# --- What a system answers with -------------------------------------------------------------


@dataclass
class MtragAnswer:
    outcome: GenerationOutcome
    # What the answer was written from, as the generator saw it (ids E1..En).
    evidence: list[Evidence]
    query: str
    paper_ids: list[str]
    searched: bool
    searches: int
    model_calls: int
    stop: str
    cost_usd: float
    seconds: float
    warnings: list[str] = field(default_factory=list)
    # The agent's self-check: its result, claims it first found unsupported, claims dropped.
    check: tuple[str, int, int] | None = None


class MtragSystem(Protocol):
    name: str

    def answer(
        self, message: str, history: list[HistoryItem], carried: list[Evidence], *, run_id: str
    ) -> MtragAnswer: ...


class PipelineSystem:
    """The shipped turn: condense a follow-up, search once, answer. Carried passages unused."""

    def __init__(
        self, pipeline: Any, clock: Callable[[], float] = time.monotonic, *, name: str = "pipeline"
    ) -> None:
        self.name = name
        self._pipeline = pipeline
        self._clock = clock

    def answer(
        self, message: str, history: list[HistoryItem], carried: list[Evidence], *, run_id: str
    ) -> MtragAnswer:
        started = self._clock()
        state = self._pipeline.run(message, history, request_id=uuid4(), run_id=run_id)
        assert state.outcome is not None
        return MtragAnswer(
            outcome=state.outcome,
            evidence=list(state.evidence),
            query=state.standalone,
            paper_ids=list(state.paper_ids),
            searched=True,
            searches=1,
            model_calls=state.outcome.calls,
            stop="pipeline",
            cost_usd=state.cost_usd,
            seconds=self._clock() - started,
            warnings=list(state.warnings),
        )


class AgentSystem:
    """The research agent: it may answer a follow-up from carried passages without searching."""

    def __init__(
        self, turn: Any, clock: Callable[[], float] = time.monotonic, *, name: str = "agent"
    ) -> None:
        self.name = name
        self._turn = turn
        self._clock = clock

    def answer(
        self, message: str, history: list[HistoryItem], carried: list[Evidence], *, run_id: str
    ) -> MtragAnswer:
        started = self._clock()
        result = self._turn.run(message, history, carried, request_id=uuid4(), run_id=run_id)
        return MtragAnswer(
            outcome=result.outcome,
            evidence=list(result.evidence),
            query=result.standalone,
            paper_ids=list(result.paper_ids),
            searched=result.searches > 0,
            searches=result.searches,
            model_calls=result.model_calls
            + result.outcome.calls
            + (result.check.calls if result.check is not None else 0),
            stop=result.stop,
            cost_usd=result.cost_usd,
            seconds=self._clock() - started,
            warnings=list(result.warnings),
            check=None
            if result.check is None
            else (
                result.check.result,
                result.check.first_unsupported,
                result.check.final_unsupported_dropped,
            ),
        )


# --- Scoring one turn ------------------------------------------------------------------------


def average_precision(ranked: Sequence[str], relevant: Sequence[str]) -> float | None:
    """AP of a ranked list against a set of relevant passages; None when none is relevant."""

    gold = set(relevant)
    if not gold:
        return None
    hits = 0
    total = 0.0
    for rank, item in enumerate(dict.fromkeys(ranked), 1):
        if item in gold:
            hits += 1
            total += hits / rank
    return total / len(gold)


def cited_items(outcome: GenerationOutcome, evidence: Sequence[Evidence]) -> list[Evidence]:
    by_id = {item.id: item for item in evidence}
    cited = [
        by_id[label]
        for claim in outcome.answer.claims
        for label in claim.evidence_ids
        if label in by_id
    ]
    unique: dict[str, Evidence] = {}
    for item in cited:
        unique.setdefault(str(item.chunk_id or item.id), item)
    return list(unique.values())


def judge_question(turn: mtrag.MtragTurn, rewrites: Mapping[str, str]) -> str:
    """The person's standalone rewrite when there is one; else the user turns so far."""

    if turn.task_id in rewrites:
        return rewrites[turn.task_id]
    asked = [text for speaker, text in turn.history if speaker == "user"]
    return " / ".join([*asked, turn.question])


def score_turn(
    turn: mtrag.MtragTurn,
    outcome: GenerationOutcome,
    cited_passages: Sequence[str],
    judge: Any,
    *,
    run_id: str,
    question: str | None = None,
) -> dict[str, Any]:
    abstained = outcome.answer.insufficient_evidence and not outcome.answer.claims
    row: dict[str, Any] = {
        "abstained": abstained,
        "correct": None,
        "judge_calls": 0,
        "judge_cost_usd": 0.0,
        "judge_error": None,
        "cited_map": average_precision(cited_passages, turn.relevant),
        "reason": "",
    }
    if turn.answerability == UNANSWERABLE:
        row["correct"] = abstained
        return row
    if abstained:
        row["correct"] = False
        return row
    try:
        verdict = judge.judge(
            question or turn.question,
            turn.reference,
            answer_text(outcome.answer),
            request_id=uuid4(),
            run_id=run_id,
        )
    except JudgeError as error:
        row["judge_error"] = error.code
    except LlmError as error:
        row["judge_error"] = f"judge:{error.code}"
    else:
        row.update(
            correct=verdict.verdict.equivalent,
            judge_calls=verdict.calls,
            judge_cost_usd=verdict.cost_usd,
            reason=verdict.verdict.reason,
        )
    return row


def _history_items(pairs: Sequence[tuple[str, str]], turns: int) -> list[HistoryItem]:
    """(speaker, text) pairs as chat history, the last ``turns`` exchanges, as served."""

    items = [HistoryItem("user" if s == "user" else "assistant", t) for s, t in pairs]
    return items[-2 * turns :] if turns > 0 else []


# A turn's faithfulness columns when none was scored: every row has the same columns.
_NO_FAITHFULNESS: dict[str, Any] = {
    "claims": None,
    "supported_claims": None,
    "supported_rate": None,
    "faithfulness_error": None,
    "faithfulness_cost_usd": None,
    "supported_verdicts": None,
}


@dataclass
class TurnContext:
    judge: Any
    run_id: str
    rewrites: Mapping[str, str]
    passages: Callable[[Sequence[str]], dict[str, str]]
    history_turns: int = 3
    # The faithfulness scorer (2026-10-29), when the experiment names one.
    faithfulness: Any | None = None


def _answer_row(
    conversation: mtrag.MtragConversation,
    turn: mtrag.MtragTurn,
    system: MtragSystem,
    history: list[HistoryItem],
    carried: list[Evidence],
    context: TurnContext,
    mode: str,
) -> tuple[dict[str, Any], MtragAnswer | None]:
    base: dict[str, Any] = {
        "mode": mode,
        "system": system.name,
        "conversation_id": conversation.conversation_id,
        "task_id": turn.task_id,
        "turn": turn.turn,
        "answerability": turn.answerability,
        "multi_turn_type": turn.multi_turn_type,
        "has_qrels": bool(turn.relevant),
        "failure": None,
    }
    try:
        answer = system.answer(turn.question, history, carried, run_id=context.run_id)
    except FAILURES as error:
        code = str(getattr(error, "code", type(error).__name__))
        return {
            **base,
            "failure": code,
            "correct": False,
            "abstained": False,
            "judge_calls": 0,
            "judge_cost_usd": 0.0,
            "judge_error": None,
            "cited_map": None,
            "evidence_recall@10": None,
            "searched": None,
            "searches": 0,
            "model_calls": 0,
            "stop": "failed",
            "cost_usd": 0.0,
            "seconds": 0.0,
            "warnings": [],
            "check_result": None,
            "check_first_unsupported": None,
            "check_dropped": None,
            **_NO_FAITHFULNESS,
            "text": {"question": turn.question, "answer": "", "query": ""},
        }, None
    cited = cited_items(answer.outcome, answer.evidence)
    paper_ids = [str(item.source_id) for item in cited] + [
        str(item.source_id) for item in answer.evidence
    ]
    passages = context.passages(list(dict.fromkeys(paper_ids)))
    cited_passages = [passages[str(i.source_id)] for i in cited if str(i.source_id) in passages]
    used = [passages[str(i.source_id)] for i in answer.evidence if str(i.source_id) in passages]
    scored = score_turn(
        turn,
        answer.outcome,
        cited_passages,
        context.judge,
        run_id=context.run_id,
        question=judge_question(turn, context.rewrites),
    )
    recall = (
        mtrag.passage_metrics(list(dict.fromkeys(used)), turn.relevant)["recall@10"]
        if turn.relevant
        else None
    )
    reason = scored.pop("reason")
    faithful = (
        context.faithfulness.score(
            turn.question,
            answer.outcome.answer,
            answer.evidence,
            request_id=uuid4(),
            run_id=context.run_id,
        ).fields()
        if context.faithfulness is not None
        else dict(_NO_FAITHFULNESS)
    )
    return {
        **base,
        **scored,
        "evidence_recall@10": recall,
        "searched": answer.searched,
        "searches": answer.searches,
        "model_calls": answer.model_calls,
        "stop": answer.stop,
        "cost_usd": answer.cost_usd
        + scored["judge_cost_usd"]
        + float(faithful["faithfulness_cost_usd"] or 0.0),
        "seconds": answer.seconds,
        "warnings": answer.warnings,
        **faithful,
        "check_result": answer.check[0] if answer.check else None,
        "check_first_unsupported": answer.check[1] if answer.check else None,
        "check_dropped": answer.check[2] if answer.check else None,
        "text": {
            "question": turn.question,
            "answer": answer_text(answer.outcome.answer),
            "query": answer.query,
            "reason": reason,
        },
    }, answer


def run_live(
    conversation: mtrag.MtragConversation, system: MtragSystem, context: TurnContext
) -> list[dict[str, Any]]:
    """Turn by turn, each turn seeing the people's questions and this system's own answers."""

    exchanges: list[tuple[str, str]] = []
    carried: list[Evidence] = []
    rows = []
    for turn in conversation.turns:
        history = _history_items(exchanges, context.history_turns)
        row, answer = _answer_row(conversation, turn, system, history, carried, context, "live")
        rows.append(row)
        reply = INSUFFICIENT_TEXT
        if answer is not None:
            reply = answer_text(answer.outcome.answer) or INSUFFICIENT_TEXT
            fresh = cited_items(answer.outcome, answer.evidence)
            keys = {str(item.chunk_id or item.text) for item in fresh}
            older = [i for i in carried if str(i.chunk_id or i.text) not in keys]
            carried = (fresh + older)[:CARRIED_MAX]
        exchanges += [("user", turn.question), ("agent", reply)]
    return rows


def run_gold_prefix(
    conversation: mtrag.MtragConversation, system: MtragSystem, context: TurnContext
) -> list[dict[str, Any]]:
    """Each turn on its own, after the reference conversation; nothing carried."""

    rows = []
    for turn in conversation.turns:
        history = _history_items(turn.history, context.history_turns)
        row, _ = _answer_row(conversation, turn, system, history, [], context, "gold_prefix")
        rows.append(row)
    return rows


# --- Summaries -----------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if isinstance(row["correct"], bool)]
    correct = sum(1 for row in judged if row["correct"])
    return {
        "turns": len(rows),
        "judged": len(judged),
        "correct": correct,
        "accuracy": _rate(correct, len(judged)),
        "wilson": list(wilson(correct, len(judged))) if judged else None,
    }


def _conversation_means(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    groups: dict[str, list[float]] = {}
    for row in rows:
        if isinstance(row["correct"], bool):
            groups.setdefault(row["conversation_id"], []).append(1.0 if row["correct"] else 0.0)
    return np.array([float(np.mean(v)) for _, v in sorted(groups.items())])


def summarize_mtrag(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int, confidence: float
) -> dict[str, Any]:
    overall = _accuracy(rows)
    overall["interval"] = list(
        bootstrap_interval(
            _conversation_means(rows), resamples=resamples, seed=seed, confidence=confidence
        )
    )
    labels = sorted({str(row["answerability"]) for row in rows})
    unanswerable = [row for row in rows if row["answerability"] == UNANSWERABLE]
    abstained = [row for row in rows if row["abstained"]]
    refused_right = sum(1 for row in unanswerable if row["abstained"])
    followups = [row for row in rows if row["turn"] > 1 and row["failure"] is None]
    skipped = [row for row in followups if row["searched"] is False]
    searched = [row for row in followups if row["searched"] is True]
    maps = [row["cited_map"] for row in rows if row["cited_map"] is not None]
    recalls = [row["evidence_recall@10"] for row in rows if row["evidence_recall@10"] is not None]
    seconds = [row["seconds"] for row in rows if row["failure"] is None]
    return {
        **overall,
        "by_answerability": {
            label: _accuracy([row for row in rows if row["answerability"] == label])
            for label in labels
        },
        "refusal": {
            "unanswerable_turns": len(unanswerable),
            "recall": _rate(refused_right, len(unanswerable)),
            "precision": _rate(refused_right, len(abstained)),
            "abstentions": len(abstained),
        },
        "cited_map": float(np.mean(maps)) if maps else None,
        "evidence_recall@10": float(np.mean(recalls)) if recalls else None,
        "followups": {
            "turns": len(followups),
            "skipped_search": len(skipped),
            "skip_rate": _rate(len(skipped), len(followups)),
            "accuracy_when_skipped": _accuracy(skipped)["accuracy"],
            "accuracy_when_searched": _accuracy(searched)["accuracy"],
        },
        "searches_per_turn": float(np.mean([row["searches"] for row in rows])) if rows else None,
        "model_calls_per_turn": (
            float(np.mean([row["model_calls"] for row in rows])) if rows else None
        ),
        "failures": sum(1 for row in rows if row["failure"]),
        "judge_errors": sum(1 for row in rows if row["judge_error"]),
        "cost_usd": float(sum(row["cost_usd"] for row in rows)),
        "seconds_p50": float(np.percentile(seconds, 50)) if seconds else None,
        "seconds_p95": float(np.percentile(seconds, 95)) if seconds else None,
        "faithfulness": faithfulness_metrics(rows),
    }


def pair_systems(
    treated: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Exact McNemar over turns both systems were judged on, and a bootstrap over
    conversations of the accuracy difference."""

    base = {row["task_id"]: row for row in baseline}
    pairs = [
        (row, base[row["task_id"]])
        for row in treated
        if row["task_id"] in base
        and isinstance(row["correct"], bool)
        and isinstance(base[row["task_id"]]["correct"], bool)
    ]
    only_treated = sum(1 for t, b in pairs if t["correct"] and not b["correct"])
    only_baseline = sum(1 for t, b in pairs if b["correct"] and not t["correct"])
    groups: dict[str, list[float]] = {}
    for t, b in pairs:
        groups.setdefault(t["conversation_id"], []).append(
            float(t["correct"]) - float(b["correct"])
        )
    differences = np.array([float(np.mean(v)) for _, v in sorted(groups.items())])
    return {
        "turns": len(pairs),
        "only_treated": only_treated,
        "only_baseline": only_baseline,
        "mcnemar_p": mcnemar_exact(only_treated, only_baseline),
        "difference": float(np.mean([float(t["correct"]) - float(b["correct"]) for t, b in pairs]))
        if pairs
        else None,
        "difference_interval": list(
            bootstrap_interval(differences, resamples=resamples, seed=seed, confidence=confidence)
        ),
        "conversations": len(groups),
        "supported_rate": paired_rate_difference(
            treated,
            baseline,
            key="task_id",
            by="conversation_id",
            resamples=resamples,
            seed=seed,
            confidence=confidence,
        ),
    }


# --- The live stack --------------------------------------------------------------------------


@dataclass
class Stack:
    service: Any
    release: Any
    engine: Any
    retriever: Any
    generator: Any
    condenser: Any
    judge: Any | None
    ledger: Any
    chat: Any
    evidence_config: Any
    identity: dict[str, Any]
    closers: list[Any]

    def close(self) -> None:
        for item in self.closers:
            close = getattr(item, "close", None)
            if close is not None:
                close()


def build_stack(
    *,
    settings: Any,
    database_url: str,
    data_dir: Path,
    chat_path: Path,
    judge_key: str | None,
    judge_prompt: Path | None,
    qdrant_url: str | None = None,
    release_id: str = mtrag.RELEASE,
) -> Stack:
    """The serving chat's first stage, evidence, generator and condenser over MTRAG's release,
    every model call under the main database's ledger with purpose ``evaluation``."""

    from ..chat.condense import Condenser, load_condense_prompt
    from ..chat.config import chat_search_config, load_chat_config
    from ..corpus.releases import load_release
    from ..db.session import make_engine
    from ..evidence.retrieve import EvidenceRetriever, load_evidence_config
    from ..models.generation import ChatGenerator, StructuredGenerator, load_generation_prompt
    from ..models.lc_chat import build_ledger_chat_model
    from ..models.llm import ResponseCache, build_client
    from ..models.spend import SpendLedger
    from ..search.chunks import dense_branch, lexical_branch
    from ..search.client import qdrant_client
    from ..search.service import SearchService
    from .judge import AnswerEquivalenceJudge, load_judge_prompt
    from .retrieval import PinnedModels, search_digest

    chat = load_chat_config(chat_path)
    engine = make_engine(database_url)
    client = qdrant_client(qdrant_url or settings.qdrant_url, timeout=60)
    release = load_release(engine, release_id)
    factory = PinnedModels(data_dir)
    embedder, max_tokens = factory.embedder(Path("configs/models.yaml"))
    reranker = factory.reranker(Path("configs/models.yaml"))
    search_config = chat_search_config(chat)
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
    evidence_config = load_evidence_config(chat.evidence)
    retriever = EvidenceRetriever(
        engine=engine,
        client=client,
        embedder=embedder,
        reranker=reranker,
        config=evidence_config,
        data_dir=data_dir,
        max_tokens=max_tokens,
    )
    # The ledger of record is the main database's, so every run is in one place.
    ledger = SpendLedger(make_engine(settings.database_url), settings.llm_daily_spend_cap_usd)
    cache = ResponseCache(data_dir / "cache" / "llm")
    condense_prompt = load_condense_prompt(chat.condense_prompt)
    condenser = Condenser(
        build_ledger_chat_model(
            chat.model, settings, ledger, timeout=chat.condense_timeout, cache=cache
        ),
        condense_prompt,
        max_chars=chat.history_max_chars,
        purpose="evaluation",
    )
    prompt = load_generation_prompt(chat.generate_prompt)
    generator: Any
    if chat.generator == "structured":
        generator = StructuredGenerator(
            build_ledger_chat_model(
                chat.model, settings, ledger, timeout=chat.generate_timeout, cache=cache
            ),
            prompt,
            purpose="evaluation",
        )
    else:
        generator = ChatGenerator(
            build_client(chat.model, settings, ledger, cache=cache),
            prompt,
            timeout=chat.generate_timeout,
            purpose="evaluation",
        )
    judge = None
    if judge_key is not None and judge_prompt is not None:
        judge = AnswerEquivalenceJudge(
            build_client(judge_key, settings, ledger, cache=cache), load_judge_prompt(judge_prompt)
        )
    identity = {
        "corpus": {
            "release": release.id,
            "manifest_sha256": release.manifest_sha256,
            "embedding": release.model_revision,
        },
        "search": search_digest(search_config, "hybrid_rerank", "chunks"),
        "reranker": str(getattr(reranker, "identity", "none")),
        "evidence": evidence_config.identity(),
        "generator": generator.identity,
        "condense_prompt": f"{condense_prompt.name}:{condense_prompt.sha256[:12]}",
        "chat_config_sha256": chat.sha256,
        "judge": None if judge is None else judge.identity,
    }
    closers: list[Any] = [generator, service, client]
    if judge is not None:
        closers.append(judge)
    return Stack(
        service=service,
        release=release,
        engine=engine,
        retriever=retriever,
        generator=generator,
        condenser=condenser,
        judge=judge,
        ledger=ledger,
        chat=chat,
        evidence_config=evidence_config,
        identity=identity,
        closers=closers,
    )


def _paper_search(stack: Stack) -> Any:
    from ..chat.retriever import ServicePaperSearch

    return ServicePaperSearch(stack.service, lambda: stack.release, stack.chat.papers)


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows)), path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_inputs(
    data_dir: Path, domain: mtrag.MtragDomain = mtrag.CLAPNQ
) -> tuple[list[mtrag.MtragConversation], dict[str, str], dict[str, list[str]]]:
    qrels = mtrag.parse_qrels(
        mtrag.file_path(data_dir, "qrels", domain).read_text(encoding="utf-8").splitlines()
    )
    conversations = mtrag.load_conversations(
        mtrag.file_path(data_dir, "reference", domain), qrels, domain
    )
    rewrites = mtrag.load_queries(mtrag.file_path(data_dir, "rewrite", domain))
    return conversations, rewrites, qrels


# --- `eval mtrag-retrieval` ------------------------------------------------------------------


def run_mtrag_retrieval(
    queries: str,
    out: Path,
    *,
    settings: Any,
    database_url: str,
    data_dir: Path,
    max_spend_usd: float,
    chat_path: Path = Path("configs/chat.yaml"),
    domain: mtrag.MtragDomain = mtrag.CLAPNQ,
) -> dict[str, Any]:
    if queries not in QUERY_SOURCES:
        raise MtragRunError("unknown_queries", queries)
    conversations, rewrites, _ = _load_inputs(data_dir, domain)
    lastturn = mtrag.load_queries(mtrag.file_path(data_dir, "lastturn", domain))
    turns = [(c, t) for c in conversations for t in c.turns if t.relevant]
    created = datetime.now(UTC)
    prefix = "mtrag" if domain is mtrag.CLAPNQ else f"mtrag-{domain.name}"
    run_id = f"{prefix}-retrieval-{queries}-{created:%Y%m%dT%H%M%SZ}"
    before = gpu_before_run()
    stack = build_stack(
        settings=settings,
        database_url=database_url,
        data_dir=data_dir,
        chat_path=chat_path,
        judge_key=None,
        judge_prompt=None,
        release_id=domain.release,
    )
    search = _paper_search(stack)
    rows: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    try:
        for conversation, turn in turns:
            if queries == "lastturn":
                query = lastturn[turn.task_id]
            elif queries == "rewrite":
                query = rewrites[turn.task_id]
            else:
                history = _history_items(turn.history, stack.chat.history_turns)
                query, _, _ = stack.condenser.standalone(history, turn.question, run_id=run_id)
                if stack.ledger.run_spend(run_id) > max_spend_usd:
                    raise MtragRunError("run_budget_exceeded", f"after {len(rows)} queries")
            hits = search(query)
            passages = mtrag.passage_ids_for(stack.engine, hits.paper_ids)
            ranked = [passages[p] for p in hits.paper_ids if p in passages]
            rows.append(
                {
                    "conversation_id": conversation.conversation_id,
                    "task_id": turn.task_id,
                    "turn": turn.turn,
                    "split": conversation.split,
                    "degraded": bool(hits.warnings),
                    "warnings": list(hits.warnings),
                    **mtrag.passage_metrics(ranked, turn.relevant),
                }
            )
            texts.append({"task_id": turn.task_id, "query": query})
    finally:
        stack.close()
    keys = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
    by_conversation: dict[str, list[float]] = {}
    for row in rows:
        by_conversation.setdefault(row["conversation_id"], []).append(row["ndcg@10"])
    metrics = {
        "queries": queries,
        "turns": len(rows),
        "degraded": sum(1 for row in rows if row["degraded"]),
        **{key: float(np.mean([row[key] for row in rows])) for key in keys},
        "ndcg@10_interval": list(
            bootstrap_interval(
                np.array([float(np.mean(v)) for _, v in sorted(by_conversation.items())]),
                resamples=1000,
                seed=42,
                confidence=0.95,
            )
        ),
        "cost_usd": stack.ledger.run_spend(run_id),
    }
    directory = out / queries
    _write_parquet(directory / "cases.parquet", rows)
    _write_json(directory / "metrics.json", metrics)
    _write_json(
        directory / "manifest.json",
        {
            "run_id": run_id,
            "created_at": created.isoformat(),
            "queries": queries,
            "identity": stack.identity,
            "source": json.loads((domain.fixtures / "source.json").read_text()),
            "gpu_before_run": before,
            "code": code_state(),
            "hardware": hardware(),
        },
    )
    text_path = Path(data_dir) / "eval" / "mtrag" / run_id / "queries.jsonl"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("".join(json.dumps(t) + "\n" for t in texts), encoding="utf-8")
    return {"run_id": run_id, "out": str(directory), **metrics}


# --- `eval mtrag` ------------------------------------------------------------------------------


@dataclass(frozen=True)
class MtragExperiment:
    path: Path
    sha256: str
    name: str
    release: str
    chat: Path
    judge: str
    judge_prompt: Path
    modes: tuple[str, ...]
    systems: tuple[dict[str, Any], ...]
    resamples: int
    seed: int
    confidence: float
    decision: dict[str, Any]
    budget_usd: float
    domain: mtrag.MtragDomain = mtrag.CLAPNQ
    # The faithfulness score for every system (2026-10-29): {model, prompt}; empty means off.
    faithfulness: Mapping[str, Any] = field(default_factory=dict)

    def pairs(self) -> dict[str, tuple[str, str]]:
        """The registered comparisons as {label: (treated, baseline)}: the decision's under
        "decision" and, when registered, a secondary one under "secondary"."""

        out = {
            "decision": (
                str(self.decision.get("treated", "agent")),
                str(self.decision.get("baseline", "pipeline")),
            )
        }
        secondary = self.decision.get("secondary")
        if secondary:
            out["secondary"] = (str(secondary["treated"]), str(secondary["baseline"]))
        return out


def is_agent(spec: Mapping[str, Any]) -> bool:
    """A system is the research agent when it carries agent settings; otherwise the pipeline."""

    return "agent" in spec


def load_mtrag_experiment(path: str | Path) -> MtragExperiment:
    target = Path(path)
    raw_bytes = target.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    try:
        bootstrap = raw.get("bootstrap") or {}
        experiment = MtragExperiment(
            path=target,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            name=str(raw["name"]),
            release=str(raw["corpus"]["release"]),
            chat=Path(raw["chat"]),
            judge=str(raw["judge"]),
            judge_prompt=Path(raw["prompts"]["judge"]),
            modes=tuple(str(m) for m in raw["modes"]),
            systems=tuple(dict(s) for s in raw["systems"]),
            resamples=int(bootstrap.get("resamples", 1000)),
            seed=int(bootstrap.get("seed", 42)),
            confidence=float(bootstrap.get("confidence", 0.95)),
            decision=dict(raw.get("decision") or {}),
            budget_usd=float((raw.get("budget") or {}).get("max_spend_usd", 0.0)),
            domain=mtrag.domain_for_release(str(raw["corpus"]["release"])),
            faithfulness=dict(raw.get("faithfulness") or {}),
        )
    except mtrag.MtragError as error:
        raise MtragRunError("experiment_invalid", "corpus.release") from error
    except (KeyError, TypeError, ValueError) as error:
        raise MtragRunError("experiment_invalid", str(error)) from error
    if not experiment.modes or set(experiment.modes) - set(MODES):
        raise MtragRunError("experiment_invalid", "modes")
    names = [str(s.get("name")) for s in experiment.systems]
    if not names or not all(s.get("name") for s in experiment.systems):
        raise MtragRunError("experiment_invalid", "systems")
    if len(names) != len(set(names)):
        raise MtragRunError("experiment_invalid", "systems")
    for treated, baseline in experiment.pairs().values():
        if treated not in names or baseline not in names:
            raise MtragRunError("experiment_invalid", "decision")
    return experiment


def select_conversations(
    conversations: Sequence[mtrag.MtragConversation], split: str, *, frozen: bool
) -> list[mtrag.MtragConversation]:
    """`development` and `test` as split; `all` reads every conversation as a development read
    (2026-10-28: ClapNQ's test conversations, spent on 2026-10-27, tune research-v3)."""

    if split not in SPLITS:
        raise MtragRunError("unknown_split", split)
    if split == "test" and not frozen:
        raise MtragRunError("frozen_flag_required", "the test split is scored once")
    if split == "all":
        if frozen:
            raise MtragRunError("frozen_not_allowed", "the all split is a development read")
        return list(conversations)
    return [c for c in conversations if c.split == split]


def mtrag_preflight_usd(
    experiment: MtragExperiment, *, turns: int, evidence_tokens: int, chat_model: str
) -> float:
    """Every turn at its worst, per mode and system: the condense call, two generations and a
    judge call for the pipeline; the agent's steps, two generations and a judge call for it."""

    from ..models.spend import estimate_usd, load_llm_config
    from .qa import (
        GENERATION_OVERHEAD_CHARS,
        JUDGE_PROMPT_CHARS,
        PROMPT_CHARS_PER_TOKEN,
        agent_limits,
        agent_turn_worst_usd,
    )

    config = load_llm_config("configs/llm.yaml")
    model = config.models[chat_model]
    judge = config.models[experiment.judge]
    generation = estimate_usd(
        model.price,
        evidence_tokens * PROMPT_CHARS_PER_TOKEN + GENERATION_OVERHEAD_CHARS,
        model.max_output_tokens,
    )
    judging = estimate_usd(judge.price, JUDGE_PROMPT_CHARS, judge.max_output_tokens)
    condense = estimate_usd(model.price, 8_000, model.max_output_tokens)
    worst = 0.0
    for system in experiment.systems:
        if is_agent(system):
            steps = agent_turn_worst_usd(
                model.price,
                model.max_output_tokens,
                evidence_tokens,
                agent_limits(system.get("agent") or {}),
                planned=bool((system.get("agent") or {}).get("plan", False)),
                checked=bool((system.get("agent") or {}).get("check", False)),
            )
        else:
            steps = condense
        worst += turns * len(experiment.modes) * (steps + 2 * generation + judging)
    if experiment.faithfulness:
        scorer = config.models[str(experiment.faithfulness["model"])]
        worst += (
            turns
            * len(experiment.modes)
            * len(experiment.systems)
            * estimate_usd(
                scorer.price,
                evidence_tokens * PROMPT_CHARS_PER_TOKEN + GENERATION_OVERHEAD_CHARS,
                scorer.max_output_tokens,
            )
        )
    return worst


def _system_for(spec: Mapping[str, Any], stack: Stack, settings: Any) -> MtragSystem:
    from ..chat.pipeline import TurnPipeline
    from ..chat.retriever import CorpusRetriever
    from .qa import build_agent_turn

    if is_agent(spec):
        return AgentSystem(
            build_agent_turn(
                spec=spec.get("agent") or {},
                model_key=stack.chat.model,
                settings=settings,
                ledger=stack.ledger,
                service=stack.service,
                release=stack.release,
                papers=stack.chat.papers,
                retriever=stack.retriever,
                generator=stack.generator,
                evidence_config=stack.evidence_config,
            ),
            name=str(spec["name"]),
        )
    retriever = CorpusRetriever(search=_paper_search(stack), evidence=stack.retriever.retrieve)
    return PipelineSystem(
        TurnPipeline(
            condenser=stack.condenser,
            retriever=retriever,
            generator=stack.generator,
            condense=stack.chat.condense,
            history_turns=stack.chat.history_turns,
        ),
        name=str(spec["name"]),
    )


def record_test_scoring(out: Path, run_id: str, keys: Sequence[str]) -> dict[str, int]:
    path = out / "frozen-scorings.json"
    registry: dict[str, list[str]] = {}
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        registry.setdefault(key, []).append(run_id)
    _write_json(path, registry)
    return {key: len(runs) for key, runs in registry.items()}


def run_mtrag(
    config: str | Path,
    split: str,
    out: Path,
    *,
    settings: Any,
    database_url: str,
    data_dir: Path,
    max_spend_usd: float,
    frozen: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    experiment = load_mtrag_experiment(config)
    conversations, rewrites, _ = _load_inputs(data_dir, experiment.domain)
    chosen = select_conversations(conversations, split, frozen=frozen)
    if limit is not None:
        chosen = chosen[:limit]
    turns = sum(len(c.turns) for c in chosen)
    from ..chat.config import load_chat_config
    from ..evidence.retrieve import load_evidence_config

    chat = load_chat_config(experiment.chat)
    worst = mtrag_preflight_usd(
        experiment,
        turns=turns,
        evidence_tokens=load_evidence_config(chat.evidence).token_budget,
        chat_model=chat.model,
    )
    if worst > max_spend_usd:
        raise MtragRunError(
            "run_budget_exceeded", f"worst case ${worst:.4f} > budget ${max_spend_usd:.4f}"
        )
    created = datetime.now(UTC)
    run_id = f"{experiment.name}-{split}-{created:%Y%m%dT%H%M%SZ}"
    before = gpu_before_run()
    stack = build_stack(
        settings=settings,
        database_url=database_url,
        data_dir=data_dir,
        chat_path=experiment.chat,
        judge_key=experiment.judge,
        judge_prompt=experiment.judge_prompt,
        release_id=experiment.release,
    )
    results: dict[str, list[dict[str, Any]]] = {}
    ledger_runs: dict[str, str] = {}
    stopped: dict[str, str] = {}
    identities: dict[str, Any] = {}
    from .qa import build_faithfulness_scorer

    scorer = build_faithfulness_scorer(experiment.faithfulness, settings, stack.ledger)
    try:
        for mode in experiment.modes:
            for spec in experiment.systems:
                system = _system_for(spec, stack, settings)
                key = f"{mode}.{system.name}"
                identities[system.name] = getattr(
                    getattr(system, "_turn", None), "identity", "pipeline"
                )
                ledger_run = f"{run_id}.{key}.{uuid4().hex[:8]}"
                ledger_runs[key] = ledger_run
                context = TurnContext(
                    judge=stack.judge,
                    run_id=ledger_run,
                    rewrites=rewrites,
                    passages=lambda ids: mtrag.passage_ids_for(stack.engine, ids),
                    history_turns=stack.chat.history_turns,
                    faithfulness=scorer,
                )
                runner = run_live if mode == "live" else run_gold_prefix
                rows: list[dict[str, Any]] = []
                for conversation in chosen:
                    rows.extend(runner(conversation, system, context))
                    spent = sum(stack.ledger.run_spend(run) for run in ledger_runs.values())
                    if spent > max_spend_usd:
                        stopped[key] = f"budget: ${spent:.4f} > ${max_spend_usd:.4f}"
                        break
                results[key] = rows
                if key in stopped:
                    break
            if stopped:
                break
    finally:
        stack.close()

    scorings = record_test_scoring(out, run_id, list(results)) if split == "test" else {}
    manifest = {
        "run_id": run_id,
        "created_at": created.isoformat(),
        "experiment": {"path": str(experiment.path), "sha256": experiment.sha256},
        "split": split,
        "conversations": len(chosen),
        "turns": turns,
        "limit": limit,
        "identity": stack.identity,
        "systems": identities,
        "faithfulness": None if scorer is None else scorer.identity,
        "source": json.loads((experiment.domain.fixtures / "source.json").read_text()),
        "budget_usd": max_spend_usd,
        "preflight_worst_usd": worst,
        "ledger_runs": ledger_runs,
        "stopped": stopped,
        "test_scorings": scorings,
        "gpu_before_run": before,
        "code": code_state(),
        "hardware": hardware(),
    }
    summary: dict[str, Any] = {}
    text_root = Path(data_dir) / "eval" / "mtrag" / run_id
    for key, rows in results.items():
        metrics = summarize_mtrag(
            rows,
            resamples=experiment.resamples,
            seed=experiment.seed,
            confidence=experiment.confidence,
        )
        metrics["ledger_cost_usd"] = stack.ledger.run_spend(ledger_runs[key])
        directory = out / split / f"{run_id}.{key}"
        _write_parquet(
            directory / "cases.parquet", [{k: v for k, v in r.items() if k != "text"} for r in rows]
        )
        _write_json(directory / "metrics.json", metrics)
        _write_json(directory / "manifest.json", {**manifest, "key": key})
        text_root.mkdir(parents=True, exist_ok=True)
        (text_root / f"{key}.jsonl").write_text(
            "".join(json.dumps({"task_id": r["task_id"], **r["text"]}) + "\n" for r in rows),
            encoding="utf-8",
        )
        summary[key] = {
            "accuracy": metrics["accuracy"],
            "judged": metrics["judged"],
            "skip_rate": metrics["followups"]["skip_rate"],
            "failures": metrics["failures"],
            "cost_usd": round(metrics["ledger_cost_usd"], 6),
        }
    pairs: dict[str, Any] = {}
    for label, (treated_name, baseline_name) in experiment.pairs().items():
        for mode in experiment.modes:
            treated = results.get(f"{mode}.{treated_name}")
            baseline = results.get(f"{mode}.{baseline_name}")
            if not (treated and baseline):
                continue
            pair = pair_systems(
                treated,
                baseline,
                resamples=experiment.resamples,
                seed=experiment.seed,
                confidence=experiment.confidence,
            )
            pair.update(treated=treated_name, baseline=baseline_name)
            # The decision's pairs stay keyed by mode, as the 2026-10-27 scoring wrote them.
            if label == "decision":
                pairs[mode] = pair
            else:
                pairs.setdefault(label, {})[mode] = pair
    if pairs:
        _write_json(out / split / f"{run_id}.pair.json", pairs)
    return {"run_id": run_id, "metrics": summary, "pairs": pairs, "stopped": stopped}
