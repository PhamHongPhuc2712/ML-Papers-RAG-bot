"""One chat turn as a LangGraph state graph: condense → retrieve → generate.

Linear on purpose: one search per turn (the developer's decision, 2026-10-11). LangGraph
gives each step a name the stream reports as it finishes, and a place to add a step later
without rewriting the turn. History is not LangGraph's: it lives in our tables and comes in
with the state, so no LangGraph checkpointer is used.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..contracts import Evidence
from ..models.generation import GenerationOutcome
from .condense import Condenser
from .retriever import CorpusRetriever, from_document
from .turns import HistoryItem


def _merge(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {**left, **right}


class ChatState(TypedDict, total=False):
    request_id: UUID
    run_id: str | None
    message: str
    history: list[HistoryItem]
    standalone: str
    documents: list[Document]
    evidence: list[Evidence]
    paper_ids: list[str]
    release_id: str | None
    outcome: GenerationOutcome
    warnings: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, float], _merge]
    cost_usd: Annotated[float, operator.add]


@dataclass(frozen=True)
class ChatGraph:
    """A compiled turn and what identifies it on every message it answers."""

    graph: Any
    identity: dict[str, Any] = field(default_factory=dict)
    condense: bool = True
    history_turns: int = 3


def build_chat_graph(
    *,
    condenser: Condenser,
    retriever: CorpusRetriever,
    generator: Any,
    condense_enabled: bool,
    clock: Callable[[], float] = time.monotonic,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def condense(state: ChatState) -> dict[str, Any]:
        started = clock()
        if not condense_enabled:
            return {"standalone": state["message"], "timings": {"condense": 0.0}}
        query, warnings, cost = condenser.standalone(
            state.get("history", []),
            state["message"],
            request_id=state.get("request_id"),
            run_id=state.get("run_id"),
        )
        return {
            "standalone": query,
            "warnings": warnings,
            "cost_usd": cost,
            "timings": {"condense": clock() - started},
        }

    def retrieve(state: ChatState) -> dict[str, Any]:
        started = clock()
        documents, hits = retriever.search_documents(state["standalone"])
        return {
            "documents": documents,
            "evidence": [from_document(document) for document in documents],
            "paper_ids": list(hits.paper_ids),
            "release_id": hits.release_id,
            "warnings": list(hits.warnings),
            "timings": {"retrieve": clock() - started},
        }

    def generate(state: ChatState) -> dict[str, Any]:
        started = clock()
        outcome: GenerationOutcome = generator.generate(
            state["standalone"],
            state.get("evidence", []),
            request_id=state.get("request_id"),
            run_id=state.get("run_id"),
        )
        warnings = []
        if outcome.reason not in (None, "no_evidence"):
            warnings.append("answer_failed_validation")
        return {
            "outcome": outcome,
            "warnings": warnings,
            "cost_usd": outcome.cost_usd,
            "timings": {"generate": clock() - started},
        }

    graph = StateGraph(ChatState)
    graph.add_node("condense", condense)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def build_default_chat_graph(
    settings: Any, engine: Any, search: Any, config_path: str = "configs/chat.yaml"
) -> ChatGraph:
    """The configured turn on the models and Qdrant client search already loaded.

    Mock mode answers with ``FakeGenerator`` and never condenses with a model, so a local
    stack without an API key still runs end to end and spends nothing. Otherwise an
    unconfigured key or cap raises ``LlmError("llm_unconfigured")`` and chat is unavailable.
    """

    from ..evidence.retrieve import EvidenceRetriever, load_evidence_config
    from ..models.generation import (
        ChatGenerator,
        FakeGenerator,
        StructuredGenerator,
        load_generation_prompt,
    )
    from ..models.lc_chat import build_ledger_chat_model
    from ..models.llm import LlmError, ResponseCache, build_client
    from ..models.spend import SpendLedger
    from ..search.api import default_search_service
    from .condense import load_condense_prompt
    from .config import chat_search_config, load_chat_config
    from .retriever import CorpusRetriever, ServicePaperSearch

    config = load_chat_config(config_path)
    client, models, runner = search.serving()
    if models.embedder is None:
        raise LlmError("models_unavailable")
    service = default_search_service(
        settings,
        engine,
        client,
        models=models,
        config=chat_search_config(config),
        runner=runner,
        listwise=False,
    )
    evidence_config = load_evidence_config(config.evidence)
    evidence = EvidenceRetriever(
        engine=engine,
        client=client,
        embedder=models.embedder,
        reranker=models.reranker,
        config=evidence_config,
        data_dir=settings.data_dir,
        max_tokens=models.max_tokens,
    )
    retriever = CorpusRetriever(
        search=ServicePaperSearch(service, search.active_release, config.papers),
        evidence=evidence.retrieve,
    )
    prompt = load_generation_prompt(config.generate_prompt)
    condense_prompt = load_condense_prompt(config.condense_prompt)
    mock = settings.model_mode == "mock" or settings.mock_mode
    generator: Any
    if mock:
        condenser = Condenser(None, condense_prompt, max_chars=config.history_max_chars)
        generator = FakeGenerator(prompt)
    else:
        ledger = SpendLedger(engine, settings.llm_daily_spend_cap_usd)
        cache = ResponseCache(settings.data_dir / "cache" / "llm")
        condenser = Condenser(
            build_ledger_chat_model(
                config.model, settings, ledger, timeout=config.condense_timeout, cache=cache
            ),
            condense_prompt,
            max_chars=config.history_max_chars,
        )
        if config.generator == "structured":
            generator = StructuredGenerator(
                build_ledger_chat_model(
                    config.model, settings, ledger, timeout=config.generate_timeout, cache=cache
                ),
                prompt,
                purpose="chat",
            )
        else:
            generator = ChatGenerator(
                build_client(config.model, settings, ledger, cache=cache),
                prompt,
                timeout=config.generate_timeout,
                purpose="chat",
            )
    graph = build_chat_graph(
        condenser=condenser,
        retriever=retriever,
        generator=generator,
        condense_enabled=config.condense,
    )
    identity = {
        "chat_config_sha256": config.sha256,
        "generator": generator.identity,
        "condense_prompt": f"{condense_prompt.name}:{condense_prompt.sha256[:12]}",
        "search": service.identity("hybrid_rerank"),
        "evidence": evidence_config.identity(),
    }
    return ChatGraph(
        graph=graph, identity=identity, condense=config.condense, history_turns=config.history_turns
    )
