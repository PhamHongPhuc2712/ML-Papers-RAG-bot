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
