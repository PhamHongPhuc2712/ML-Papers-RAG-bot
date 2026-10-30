"""The chat pipeline: condense once, search once, answer once, degrade honestly."""

from __future__ import annotations

from uuid import uuid4

import pytest

from copilot.chat.condense import Condenser, clean_query, format_history, load_condense_prompt
from copilot.chat.pipeline import TurnPipeline
from copilot.chat.retriever import CorpusRetriever, PaperHits
from copilot.chat.turns import HistoryItem
from copilot.contracts import Evidence, Reference
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


def _pipeline(model: ScriptedModel, evidence: list[Evidence], *, condense: bool = True):
    searches = Searches(evidence)
    retriever = CorpusRetriever(search=searches.search, evidence=searches.retrieve)
    pipeline = TurnPipeline(
        condenser=Condenser(model, PROMPT, max_chars=6000),
        retriever=retriever,
        generator=FakeGenerator(),
        condense=condense,
    )
    return pipeline, searches


def _run(pipeline, message: str, history: list[HistoryItem]):
    return pipeline.run(message, history)


def test_a_first_turn_skips_the_condenser():
    model = ScriptedModel([])
    pipeline, searches = _pipeline(model, [_evidence("E1")])
    state = _run(pipeline, "What does SimCSE train with?", [])
    assert model.calls == [] and searches.queries == ["What does SimCSE train with?"]
    assert state.standalone == "What does SimCSE train with?"
    assert state.outcome and not state.outcome.answer.insufficient_evidence


def test_a_follow_up_is_condensed_and_the_condensed_query_is_searched():
    model = ScriptedModel(["How large is SimCSE's training batch?"])
    pipeline, searches = _pipeline(model, [_evidence("E1")])
    state = _run(pipeline, "How large is its batch?", HISTORY)
    assert searches.queries == ["How large is SimCSE's training batch?"]
    assert "SimCSE trains sentence embeddings" in model.calls[0][1]["content"]
    assert state.cost_usd == pytest.approx(0.0001)


@pytest.mark.parametrize(
    "reply", ["", "   ", '{"query": "x"}', "x" * 2001, LlmError("llm_timeout", retryable=True)]
)
def test_a_failed_condense_falls_back_to_the_message(reply):
    pipeline, searches = _pipeline(ScriptedModel([reply]), [_evidence("E1")])
    state = _run(pipeline, "How large is its batch?", HISTORY)
    assert searches.queries == ["How large is its batch?"]
    assert "condense_failed" in state.warnings
    assert state.outcome and state.outcome.answer.claims
    assert state.cost_usd == (0.0 if isinstance(reply, LlmError) else pytest.approx(0.0001))


def test_the_spend_cap_stops_the_turn():
    pipeline, searches = _pipeline(ScriptedModel([LlmError("llm_spend_cap")]), [_evidence("E1")])
    with pytest.raises(LlmError) as caught:
        _run(pipeline, "How large is its batch?", HISTORY)
    assert caught.value.code == "llm_spend_cap" and searches.queries == []


def test_no_evidence_is_insufficient_without_a_generation_call():
    pipeline, _ = _pipeline(ScriptedModel([]), [])
    state = _run(pipeline, "Anything about protein folding?", [])
    assert state.outcome and state.outcome.answer.insufficient_evidence and state.outcome.calls == 0


def test_disabled_condensing_sends_the_raw_message():
    model = ScriptedModel([])
    pipeline, searches = _pipeline(model, [_evidence("E1")], condense=False)
    _run(pipeline, "How large is its batch?", HISTORY)
    assert model.calls == [] and searches.queries == ["How large is its batch?"]


def test_updates_stream_in_step_order():
    pipeline, searches = _pipeline(ScriptedModel(["standalone?"]), [_evidence("E1")])
    updates = list(pipeline.iter_steps("it?", HISTORY))
    assert [update.stage for update in updates] == ["condense", "retrieve", "generate"]
    assert updates[0].state.evidence == [] and updates[0].state.outcome is None
    assert updates[1].state.evidence and updates[1].state.outcome is None
    assert searches.queries == ["standalone?"]
    assert set(updates[-1].state.timings) == {"condense", "retrieve", "generate"}


def test_retrieval_preserves_the_evidence_and_generation_warning():
    evidence = [_evidence("E1")]
    pipeline, searches = _pipeline(ScriptedModel([]), evidence)
    updates = list(pipeline.iter_steps("FABRICATE TWICE", []))
    assert updates[1].state.evidence == evidence
    assert searches.queries == ["FABRICATE TWICE"]
    assert updates[-1].warnings == ["answer_failed_validation"]
    assert updates[-1].state.warnings == ["answer_failed_validation"]
    assert updates[-1].state.outcome and updates[-1].state.outcome.answer.insufficient_evidence


def test_history_keeps_the_newest_text_within_its_budget():
    long = [HistoryItem("user", "old " * 500), HistoryItem("assistant", "older " * 500)] + HISTORY
    text = format_history(long, max_chars=300)
    assert len(text) <= 300 and text.endswith(HISTORY[-1].text)
    assert format_history(HISTORY, max_chars=6000).startswith("User: What does SimCSE")


def test_clean_query_takes_one_line_and_refuses_garbage():
    assert clean_query('  "How big is the batch?"\nextra') == "How big is the batch?"
    assert clean_query("") is None and clean_query("{}") is None and clean_query("a" * 2001) is None


def test_the_shipped_chat_condenses_follow_ups():
    """The developer's decision (2026-10-14), over C7's underpowered rule: follow-ups are
    rewritten into a standalone query before the search."""

    from copilot.chat.config import load_chat_config

    config = load_chat_config("configs/chat.yaml")
    assert config.condense is True and config.condense_prompt.is_file()


def test_search_and_chat_share_one_first_stage():
    """RAG only (plan U1, 2026-10-15): the chat's measured first stage is the shipped search
    file itself, the reranker reading title, abstract and chunk (since 2026-10-21) at a 2.5 s
    deadline; the chat has no overrides of its own."""

    from copilot.chat.config import chat_search_config, load_chat_config
    from copilot.search.service import load_search_config

    shipped = load_search_config("configs/search.yaml")
    assert shipped.rerank_text == "title_abstract_chunk" and shipped.rerank_seconds == 2.5
    chat = load_chat_config("configs/chat.yaml")
    assert not hasattr(chat, "search_overrides")
    assert chat_search_config(chat) == shipped
    assert chat.papers == 10 and chat.references == 20


def test_a_chat_config_with_search_overrides_is_refused(tmp_path):
    from copilot.chat.config import load_chat_config

    text = open("configs/chat.yaml", encoding="utf-8").read()
    path = tmp_path / "chat.yaml"
    overrides = "  papers: 10\n  overrides:\n    rerank: {text: abstract}\n"
    path.write_text(text.replace("  papers: 10\n", overrides), encoding="utf-8")
    with pytest.raises(ValueError, match="chat_config_invalid"):
        load_chat_config(path)
    path.write_text(text.replace("references: 20", "references: 5"), encoding="utf-8")
    with pytest.raises(ValueError, match="chat_config_invalid:bounds"):
        load_chat_config(path)


def test_the_turn_carries_the_references_its_search_ranked():
    evidence = [_evidence("E1")]
    references = [
        Reference(paper_id=evidence[0].source_id, rank=1, read=True, title="SimCSE"),
        Reference(paper_id=uuid4(), rank=2, read=False, title="Another", venue="ACL", year=2023),
    ]
    retriever = CorpusRetriever(
        search=lambda query: PaperHits("rel-1", [str(evidence[0].source_id)], [], references),
        evidence=lambda query, papers, release: list(evidence),
    )
    pipeline = TurnPipeline(
        condenser=Condenser(ScriptedModel([]), PROMPT, max_chars=6000),
        retriever=retriever,
        generator=FakeGenerator(),
        condense=True,
    )
    state = _run(pipeline, "What does SimCSE train with?", [])
    assert state.references == references
