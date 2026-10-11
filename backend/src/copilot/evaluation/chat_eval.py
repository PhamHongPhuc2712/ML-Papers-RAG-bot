"""Follow-up answering on ORB conversations under three conditions (chat plan C7)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ..chat.turns import INSUFFICIENT_TEXT, HistoryItem, answer_text
from .judge import JudgeError
from .qa import JUDGE_PROMPT_CHARS, cited_map

CONDITIONS = ("standalone", "raw", "condensed")


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    paper_id: str
    first_id: str
    first: str
    second_id: str
    second: str
    followup: str
    reference: str


def condition_inputs(
    conversation: Conversation, condition: str, first_answer: str
) -> tuple[str, list[HistoryItem], bool]:
    """(message, history, condense) for turn 2 under one condition."""

    history = [HistoryItem("user", conversation.first), HistoryItem("assistant", first_answer)]
    if condition == "standalone":
        return conversation.second, [], False
    if condition == "raw":
        return conversation.followup, history, False
    if condition == "condensed":
        return conversation.followup, history, True
    raise ValueError(f"unknown condition: {condition}")


def _invoke(graph: Any, message: str, history: list[HistoryItem], run_id: str) -> dict[str, Any]:
    return dict(
        graph.invoke(
            {
                "message": message,
                "history": history,
                "run_id": run_id,
                "warnings": [],
                "timings": {},
                "cost_usd": 0.0,
            }
        )
    )


def run_conversation(
    conversation: Conversation,
    *,
    graphs: Mapping[bool, Any],
    judge: Any,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    """Turn 1 once, then turn 2 under each condition, each judged against B's reference.

    ``graphs[True]`` condenses follow-ups and ``graphs[False]`` does not; both are the chat's
    graph over the same retriever and generator.
    """

    first = _invoke(graphs[False], conversation.first, [], run_id)
    first_answer = answer_text(first["outcome"].answer) or INSUFFICIENT_TEXT
    rows: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        message, history, condense = condition_inputs(conversation, condition, first_answer)
        state = _invoke(graphs[condense], message, history, run_id)
        outcome = state["outcome"]
        evidence = {item.id: item for item in state.get("evidence", [])}
        cited = sorted(
            {
                str(evidence[label].source_id)
                for claim in outcome.answer.claims
                for label in claim.evidence_ids
                if label in evidence
            }
        )
        abstained = outcome.answer.insufficient_evidence or not outcome.answer.claims
        verdict = None
        judge_cost = 0.0
        judge_error = None
        if not abstained:
            try:
                judged = judge.judge(
                    conversation.second,
                    conversation.reference,
                    answer_text(outcome.answer),
                    run_id=run_id,
                )
                verdict = judged.verdict.equivalent
                judge_cost = judged.cost_usd
            except JudgeError as error:
                # As in the QA runner: unjudged, so out of the accuracy's denominator.
                judge_error = error.code
        rows[condition] = {
            # `eval pair` keys cases by query_id: the conversation is the case.
            "query_id": conversation.conversation_id,
            "family_id": conversation.paper_id,
            "type": "followup",
            "condition": condition,
            "query": state["standalone"],
            "gold_in_papers": conversation.paper_id in state.get("paper_ids", []),
            "cited_map": cited_map(cited, conversation.paper_id),
            "provenance_valid": outcome.provenance_valid,
            "abstained": abstained,
            # The QA runner's column and rule: an abstention is wrong (B has an answer).
            "equivalent": False if abstained else verdict,
            "judge_error": judge_error,
            "warnings": list(state.get("warnings", [])),
            "generation_cost_usd": float(state.get("cost_usd", 0.0)),
            "judge_cost_usd": judge_cost,
            # Text, for the copy under DATA_DIR only.
            "answer": answer_text(outcome.answer),
        }
    return rows


# --- The run: every condition over the same conversations, under one budget ---------------

RUNNABLE_SPLITS = ("development", "frozen")
GENERATION_CHARS_PER_TOKEN = 4
GENERATION_OVERHEAD_CHARS = 2_000
CONDENSE_OVERHEAD_CHARS = 1_500


class ChatEvalError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class ChatExperiment:
    path: Path
    sha256: str
    name: str
    fixture: Path
    texts: str
    release: str
    chat: Path
    judge: str
    judge_prompt: Path
    conditions: tuple[str, ...]
    resamples: int
    seed: int
    confidence: float
    decision: dict[str, Any]
    max_spend_usd: float


def load_chat_experiment(path: str | Path) -> ChatExperiment:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    try:
        bootstrap = raw["bootstrap"]
        experiment = ChatExperiment(
            path=Path(path),
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            name=str(raw["name"]),
            fixture=Path(raw["dataset"]["path"]),
            texts=str(raw["dataset"]["texts"]),
            release=str(raw["corpus"]["release"]),
            chat=Path(raw["chat"]),
            judge=str(raw["judge"]),
            judge_prompt=Path(raw["prompts"]["judge"]),
            conditions=tuple(str(c) for c in raw["conditions"]),
            resamples=int(bootstrap["resamples"]),
            seed=int(bootstrap["seed"]),
            confidence=float(bootstrap["confidence"]),
            decision=dict(raw["decision"]),
            max_spend_usd=float(raw["budget"]["max_spend_usd"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ChatEvalError("experiment_invalid", str(error)) from error
    if not experiment.conditions or set(experiment.conditions) - set(CONDITIONS):
        raise ChatEvalError("experiment_invalid", "conditions")
    return experiment


def chat_preflight_usd(
    conversations: int, chat: Any, judge_key: str, *, evidence_tokens: int
) -> float:
    """The plan's worst case: (1 condense + 4 generations) × 2 + 3 judge calls a conversation,
    each at the full output budget."""

    from ..models.spend import estimate_usd, load_llm_config

    models = load_llm_config("configs/llm.yaml").models
    model, judge = models[chat.model], models[judge_key]
    generation = estimate_usd(
        model.price,
        evidence_tokens * GENERATION_CHARS_PER_TOKEN + GENERATION_OVERHEAD_CHARS,
        model.max_output_tokens,
    )
    condense = estimate_usd(
        model.price, chat.history_max_chars + CONDENSE_OVERHEAD_CHARS, model.max_output_tokens
    )
    judged = estimate_usd(judge.price, JUDGE_PROMPT_CHARS, judge.max_output_tokens)
    return conversations * ((condense + 4 * generation) * 2 + 3 * judged)


def configuration_digest(experiment: ChatExperiment, chat: Any) -> str:
    """What one frozen scoring is of: the registration, the chat file and both prompts."""

    parts = [
        experiment.sha256,
        chat.sha256,
        hashlib.sha256(Path(chat.condense_prompt).read_bytes()).hexdigest(),
        hashlib.sha256(Path(chat.generate_prompt).read_bytes()).hexdigest(),
        hashlib.sha256(experiment.judge_prompt.read_bytes()).hexdigest(),
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _scorings(out: Path) -> dict[str, list[str]]:
    path = out / "frozen-scorings.json"
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _build_live(
    experiment: ChatExperiment, chat: Any, settings: Any, engine: Any, data_dir: Path
) -> tuple[dict[bool, Any], Any, Any, dict[str, Any], list[Any]]:
    """Both graphs over the experiment's release, as ``build_default_chat_graph`` builds the
    served one, but with purpose ``evaluation``; then the judge, the ledger and identities."""

    from ..chat.condense import Condenser, load_condense_prompt
    from ..chat.config import chat_search_config
    from ..chat.graph import build_chat_graph
    from ..chat.retriever import CorpusRetriever, ServicePaperSearch
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

    client = qdrant_client(settings.qdrant_url, timeout=60)
    release = load_release(engine, experiment.release)
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
    evidence = EvidenceRetriever(
        engine=engine,
        client=client,
        embedder=embedder,
        reranker=reranker,
        config=evidence_config,
        data_dir=data_dir,
        max_tokens=max_tokens,
    )
    retriever = CorpusRetriever(
        search=ServicePaperSearch(service, lambda: release, chat.papers),
        evidence=evidence.retrieve,
    )
    # The ledger of record is the main database's, so one daily cap covers every run.
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
    graphs = {
        flag: build_chat_graph(
            condenser=condenser, retriever=retriever, generator=generator, condense_enabled=flag
        )
        for flag in (True, False)
    }
    judge = AnswerEquivalenceJudge(
        build_client(experiment.judge, settings, ledger, cache=cache),
        load_judge_prompt(experiment.judge_prompt),
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
    }
    return graphs, judge, ledger, identity, [generator, judge, service, client]


def run_chat_eval(
    config: str | Path,
    split: str,
    out: str | Path,
    *,
    max_spend_usd: float,
    frozen: bool = False,
    limit: int | None = None,
    database_url: str | None = None,
    data_dir: Path | None = None,
    text_dir: Path | None = None,
    graphs: Mapping[bool, Any] | None = None,
    judge: Any = None,
    ledger: Any = None,
) -> dict[str, Any]:
    """``eval chat``: turn 1 once per conversation, turn 2 under each condition, judged.

    Refuses the frozen split without ``frozen``, a second frozen scoring of the same
    configuration, and a worst case over ``max_spend_usd``; stops once the run's ledger spend
    passes it. ``graphs``, ``judge`` and ``ledger`` are injected by tests; otherwise the
    chat's graph is built over the experiment's release.
    """

    from ..chat.config import load_chat_config
    from ..evidence.retrieve import load_evidence_config
    from ..models.llm import LlmError
    from .conversations import read_conversations
    from .qa import summarize_qa
    from .retrieval import code_state, gpu_before_run, hardware

    if split not in RUNNABLE_SPLITS:
        raise ChatEvalError("unknown_split", split)
    if split == "frozen" and not frozen:
        raise ChatEvalError("frozen_flag_required", "the frozen split is scored once")
    experiment = load_chat_experiment(config)
    chat = load_chat_config(experiment.chat)
    out_dir = Path(out)
    digest = configuration_digest(experiment, chat)
    if split == "frozen" and _scorings(out_dir).get(digest):
        raise ChatEvalError("frozen_already_scored", digest)
    settings: Any = None
    if data_dir is None:
        from ..config import Settings

        settings = Settings()  # type: ignore[call-arg]
        data_dir = Path(settings.data_dir)
    text_dir = text_dir or data_dir / "eval" / "chat"
    conversations = [
        Conversation(
            conversation_id=row["conversation_id"],
            paper_id=row["paper_id"],
            first_id=row["first_id"],
            first=row["first"],
            second_id=row["second_id"],
            second=row["second"],
            followup=row["followup"],
            reference=row["reference"],
        )
        for row in read_conversations(experiment.fixture, data_dir / experiment.texts, split)
    ]
    if limit is not None:
        conversations = conversations[:limit]
    if not conversations:
        raise ChatEvalError("no_conversations", split)
    worst = chat_preflight_usd(
        len(conversations),
        chat,
        experiment.judge,
        evidence_tokens=load_evidence_config(chat.evidence).token_budget,
    )
    if worst > max_spend_usd:
        raise ChatEvalError(
            "run_budget_exceeded",
            f"worst case ${worst:.4f} for {len(conversations)} conversations"
            f" > budget ${max_spend_usd:.4f}",
        )
    created = datetime.now(UTC)
    run_id = f"{experiment.name}-{split}-{created:%Y%m%dT%H%M%SZ}"
    ledger_run = f"{run_id}.{uuid4().hex[:8]}"
    before: dict[str, Any] = {}
    identity: dict[str, Any] = {}
    closing: list[Any] = []
    if graphs is None or judge is None or ledger is None:
        from ..db.session import make_engine

        if settings is None:
            from ..config import Settings

            settings = Settings()  # type: ignore[call-arg]
        before = gpu_before_run()
        engine = make_engine(database_url or settings.database_url)
        graphs, judge, ledger, identity, closing = _build_live(
            experiment, chat, settings, engine, data_dir
        )

    results: dict[str, list[dict[str, Any]]] = {c: [] for c in experiment.conditions}
    failures: list[dict[str, str]] = []
    stopped: str | None = None
    try:
        for conversation in conversations:
            try:
                by_condition = run_conversation(
                    conversation, graphs=graphs, judge=judge, run_id=ledger_run
                )
            except LlmError as error:
                if error.code == "llm_spend_cap":
                    stopped = "daily spend cap"
                    break
                # A conversation that failed outright is left out of every condition, so the
                # conditions stay paired case for case.
                failures.append(
                    {"conversation_id": conversation.conversation_id, "error": error.code}
                )
                continue
            for condition in experiment.conditions:
                results[condition].append(by_condition[condition])
            spent = float(ledger.run_spend(ledger_run))
            if spent > max_spend_usd:
                stopped = f"budget: ${spent:.4f} > ${max_spend_usd:.4f}"
                break
    finally:
        for part in closing:
            close = getattr(part, "close", None)
            if callable(close):
                close()

    metrics: dict[str, Any] = {}
    for condition, rows in results.items():
        metrics[condition] = summarize_qa(
            rows,
            resamples=experiment.resamples,
            seed=experiment.seed,
            confidence=experiment.confidence,
        )
        metrics[condition]["condense_failed"] = sum(
            1 for row in rows if "condense_failed" in row["warnings"]
        )
    ledger_cost = float(ledger.run_spend(ledger_run))
    scorings: dict[str, Any] = {}
    if split == "frozen":
        registry = _scorings(out_dir)
        registry.setdefault(digest, []).append(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frozen-scorings.json").write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        scorings = {key: len(runs) for key, runs in registry.items()}
    manifest = {
        "run_id": run_id,
        "created_at": created.isoformat(),
        "experiment": {
            "path": str(experiment.path),
            "sha256": experiment.sha256,
            "name": experiment.name,
        },
        "configuration_sha256": digest,
        "chat": {"path": str(experiment.chat), "sha256": chat.sha256},
        "split": split,
        "conversations": len(conversations),
        "limit": limit,
        "dataset": {
            "path": str(experiment.fixture),
            "sha256": hashlib.sha256(experiment.fixture.read_bytes()).hexdigest(),
        },
        "judge": {"key": experiment.judge, "identity": str(getattr(judge, "identity", ""))},
        "identity": identity,
        "ledger_run_id": ledger_run,
        "ledger_cost_usd": ledger_cost,
        "budget_usd": max_spend_usd,
        "preflight_worst_usd": worst,
        "stopped": stopped,
        "failures": failures,
        "frozen_scorings": scorings,
        "code": code_state(),
        "hardware": hardware(),
        "gpu_before_run": before,
        "text_dir": str(text_dir),
    }
    texts = {c.conversation_id: c for c in conversations}
    written: dict[str, str] = {}
    for condition, rows in results.items():
        directory = out_dir / split / f"{run_id}.{condition}"
        directory.mkdir(parents=True, exist_ok=True)
        public = [{k: v for k, v in row.items() if k != "answer"} for row in rows]
        pq.write_table(pa.Table.from_pylist(public), directory / "cases.parquet")
        (directory / "metrics.json").write_text(
            json.dumps(metrics[condition], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / "manifest.json").write_text(
            json.dumps({**manifest, "condition": condition}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        text_path = text_dir / run_id / condition / "cases.jsonl"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(
            "".join(
                json.dumps(
                    {
                        **row,
                        "first": texts[row["query_id"]].first,
                        "followup": texts[row["query_id"]].followup,
                        "second": texts[row["query_id"]].second,
                        "reference": texts[row["query_id"]].reference,
                    },
                    sort_keys=True,
                )
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        written[condition] = str(directory)
    return {
        "run_id": run_id,
        "written": written,
        "metrics": {
            condition: {
                "accuracy": round(m["accuracy"]["mean"], 4),
                "wilson": [round(x, 4) for x in m["accuracy"]["wilson"]],
                "judged": m["judged"],
                "gold_in_papers": round(m["gold_in_papers"], 4),
                "abstention_rate": round(m["abstention_rate"], 4),
                "condense_failed": m["condense_failed"],
            }
            for condition, m in metrics.items()
        },
        "ledger_cost_usd": round(ledger_cost, 6),
        "failures": len(failures),
        "stopped": stopped,
    }
