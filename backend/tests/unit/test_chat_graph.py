"""The chat turn as a graph: condense once, search once, answer once, degrade honestly."""

from __future__ import annotations

from uuid import uuid4

import pytest

from copilot.chat.condense import Condenser, clean_query, format_history, load_condense_prompt
from copilot.chat.graph import build_chat_graph
from copilot.chat.retriever import CorpusRetriever, PaperHits, from_document, to_document
from copilot.chat.turns import HistoryItem
from copilot.contracts import Evidence
from copilot.models.generation import FakeGenerator
from copilot.models.llm import ChatResult, LlmError
from copilot.models.spend import Usage

PROMPT = load_condense_prompt("prompts/chat/condense-v1.yaml")
HISTORY = [
    HistoryItem("user", "What does SimCSE train with?"),
    HistoryItem("assistant", "SimCSE trains sentence embeddings with a contrastive loss."),
]


def _evidence(label: str) -> Evidence:
    return Evidence(
        id=label,
        source_kind="paper",
        source_id=uuid4(),
        version_id="v1",
        chunk_id=uuid4(),
        text=f"Passage {label} about contrastive sentence embeddings.",
        section="Method",
    )


class ScriptedModel:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, purpose, schema=None, request_id=None, run_id=None):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(reply, Usage(10, 0, 5), "m", 0.0001, False, 5)


class Searches:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.queries: list[str] = []
        self.evidence = evidence

    def search(self, query: str) -> PaperHits:
        self.queries.append(query)
        return PaperHits("rel-1", [str(e.source_id) for e in self.evidence], [])

    def retrieve(self, query: str, papers: list[str], release: str) -> list[Evidence]:
        return list(self.evidence)


def _graph(model: ScriptedModel, evidence: list[Evidence], *, condense: bool = True):
    searches = Searches(evidence)
    retriever = CorpusRetriever(search=searches.search, evidence=searches.retrieve)
    graph = build_chat_graph(
        condenser=Condenser(model, PROMPT, max_chars=6000),
        retriever=retriever,
        generator=FakeGenerator(),
        condense_enabled=condense,
    )
    return graph, searches


def _run(graph, message: str, history: list[HistoryItem]):
    return graph.invoke(
        {"message": message, "history": history, "warnings": [], "timings": {}, "cost_usd": 0.0}
    )


def test_a_first_turn_skips_the_condenser():
    model = ScriptedModel([])
    graph, searches = _graph(model, [_evidence("E1")])
    state = _run(graph, "What does SimCSE train with?", [])
    assert model.calls == [] and searches.queries == ["What does SimCSE train with?"]
    assert state["standalone"] == "What does SimCSE train with?"
    assert not state["outcome"].answer.insufficient_evidence


def test_a_follow_up_is_condensed_and_the_condensed_query_is_searched():
    model = ScriptedModel(["How large is SimCSE's training batch?"])
    graph, searches = _graph(model, [_evidence("E1")])
    state = _run(graph, "How large is its batch?", HISTORY)
    assert searches.queries == ["How large is SimCSE's training batch?"]
    assert "SimCSE trains sentence embeddings" in model.calls[0][1]["content"]
    assert state["cost_usd"] == pytest.approx(0.0001)


@pytest.mark.parametrize(
    "reply", ["", "   ", '{"query": "x"}', "x" * 2001, LlmError("llm_timeout", retryable=True)]
)
def test_a_failed_condense_falls_back_to_the_message(reply):
    graph, searches = _graph(ScriptedModel([reply]), [_evidence("E1")])
    state = _run(graph, "How large is its batch?", HISTORY)
    assert searches.queries == ["How large is its batch?"]
    assert "condense_failed" in state["warnings"]
    assert state["outcome"].answer.claims


def test_the_spend_cap_stops_the_turn():
    graph, searches = _graph(ScriptedModel([LlmError("llm_spend_cap")]), [_evidence("E1")])
    with pytest.raises(LlmError) as caught:
        _run(graph, "How large is its batch?", HISTORY)
    assert caught.value.code == "llm_spend_cap" and searches.queries == []


def test_no_evidence_is_insufficient_without_a_generation_call():
    graph, _ = _graph(ScriptedModel([]), [])
    state = _run(graph, "Anything about protein folding?", [])
    assert state["outcome"].answer.insufficient_evidence and state["outcome"].calls == 0


def test_disabled_condensing_sends_the_raw_message():
    model = ScriptedModel([])
    graph, searches = _graph(model, [_evidence("E1")], condense=False)
    _run(graph, "How large is its batch?", HISTORY)
    assert model.calls == [] and searches.queries == ["How large is its batch?"]


def test_updates_stream_in_node_order():
    graph, _ = _graph(ScriptedModel(["standalone?"]), [_evidence("E1")])
    nodes = [
        next(iter(update))
        for update in graph.stream(
            {"message": "it?", "history": HISTORY, "warnings": [], "timings": {}, "cost_usd": 0.0},
            stream_mode="updates",
        )
    ]
    assert nodes == ["condense", "retrieve", "generate"]


def test_history_keeps_the_newest_text_within_its_budget():
    long = [HistoryItem("user", "old " * 500), HistoryItem("assistant", "older " * 500)] + HISTORY
    text = format_history(long, max_chars=300)
    assert len(text) <= 300 and text.endswith(HISTORY[-1].text)
    assert format_history(HISTORY, max_chars=6000).startswith("User: What does SimCSE")


def test_clean_query_takes_one_line_and_refuses_garbage():
    assert clean_query('  "How big is the batch?"\nextra') == "How big is the batch?"
    assert clean_query("") is None and clean_query("{}") is None and clean_query("a" * 2001) is None


def test_documents_round_trip_to_evidence():
    item = _evidence("E3")
    assert from_document(to_document(item)) == item
