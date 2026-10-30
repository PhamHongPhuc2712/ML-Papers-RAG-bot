"""One agent turn: the loop gathers evidence, the existing generator writes the answer."""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from copilot.agent.tools import AgentLimits
from copilot.agent.turn import AgentTurn, load_agent_prompt
from copilot.chat.retriever import CorpusRetriever, PaperHits
from copilot.chat.turns import HistoryItem
from copilot.contracts import Evidence
from copilot.models.generation import FakeGenerator
from copilot.models.llm import LlmError

PROMPT = load_agent_prompt("prompts/agent/research-v1.yaml")


def _evidence(label: str) -> Evidence:
    return Evidence(
        id=label,
        source_kind="paper",
        source_id=uuid4(),
        version_id=str(uuid4()),
        chunk_id=uuid4(),
        text=f"Passage {label}: dropout as augmentation.",
        section="Method",
    )


def _call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class Scripted:
    identity = "scripted"

    def __init__(self, replies: list[AIMessage | Exception]) -> None:
        self.replies = list(replies)
        self.seen: list[list] = []

    def step(self, messages, tools, *, request_id=None, run_id=None):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 0.001


class Corpus:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> PaperHits:
        self.queries.append(query)
        return PaperHits("rel-1", ["p1"], [])

    def evidence(self, query, papers, release):
        return [_evidence("E1"), _evidence("E2")]


def _turn(model, **limits) -> tuple[AgentTurn, Corpus]:
    corpus = Corpus()
    turn = AgentTurn(
        model,
        CorpusRetriever(search=corpus.search, evidence=corpus.evidence),
        lambda item: [],
        FakeGenerator(),
        PROMPT,
        limits=AgentLimits(**limits),
        max_chars=6000,
        token_budget=6000,
        count_tokens=lambda text: len(text.split()),
    )
    return turn, corpus


def test_a_first_question_searches_then_answers_from_the_chosen_passages():
    model = Scripted(
        [
            _call("search_papers", {"query": "dropout augmentation SimCSE"}),
            _call(
                "answer", {"question": "Does SimCSE use dropout?", "evidence_ids": ["S1E2"]}, "c2"
            ),
        ]
    )
    turn, corpus = _turn(model)
    result = turn.run("Does SimCSE use dropout?", [], [])
    assert corpus.queries == ["dropout augmentation SimCSE"]
    assert result.chosen_ids == ["S1E2"] and [e.id for e in result.evidence] == ["E1"]
    assert "Passage E2" in result.evidence[0].text
    assert [c.evidence_ids for c in result.outcome.answer.claims] == [["E1"]]
    assert result.stop == "answered" and result.searches == 1 and result.model_calls == 2
    assert result.standalone == "Does SimCSE use dropout?"
    assert result.cost_usd == pytest.approx(0.002 + result.outcome.cost_usd)
    assert set(result.timings) == {"agent", "generate"}
    assert result.paper_ids == ["p1"] and result.release_id == "rel-1"


def test_a_follow_up_can_answer_from_carried_passages_without_searching():
    model = Scripted([_call("answer", {"question": "Which uses dropout?", "evidence_ids": ["H1"]})])
    turn, corpus = _turn(model)
    history = [
        HistoryItem("user", "Papers on sentence embeddings?"),
        HistoryItem("assistant", "SimCSE."),
    ]
    carried = _evidence("E4")
    result = turn.run("Which of them uses dropout?", history, [carried])
    assert corpus.queries == [] and result.searches == 0
    assert result.chosen_ids == ["H1"] and result.evidence[0].chunk_id == carried.chunk_id
    prompt = model.seen[0]
    assert prompt[1].content == "Papers on sentence embeddings?" and prompt[2].content == "SimCSE."
    assert "[H1]" in prompt[-1].content and "Which of them uses dropout?" in prompt[-1].content


def test_the_system_prompt_states_the_search_budget():
    model = Scripted([_call("answer", {"question": "q", "evidence_ids": []})])
    turn, _ = _turn(model, max_searches=2)
    turn.run("q", [], [])
    assert "at most 2 searches" in model.seen[0][0].content


def test_an_answer_citing_unknown_ids_drops_them_and_warns():
    model = Scripted([_call("answer", {"question": "q", "evidence_ids": ["H1", "S9E9"]})])
    turn, _ = _turn(model)
    result = turn.run("Which of them?", [], [])
    assert result.evidence == [] and "agent_unknown_evidence" in result.warnings
    assert result.outcome.answer.insufficient_evidence and result.outcome.reason == "no_evidence"


def test_an_empty_answer_is_an_abstention_without_a_generation_call():
    model = Scripted(
        [
            _call("search_papers", {"query": "q"}),
            _call("answer", {"question": "q", "evidence_ids": []}, "c2"),
        ]
    )
    turn, _ = _turn(model)
    result = turn.run("q", [], [])
    assert result.evidence == [] and result.outcome.calls == 0
    assert result.outcome.answer.insufficient_evidence and result.warnings == []


def test_no_answer_call_writes_from_everything_seen():
    model = Scripted([_call("search_papers", {"query": "q"}), AIMessage(content="Done.")])
    turn, _ = _turn(model)
    result = turn.run("q", [], [])
    assert result.stop == "no_answer_call" and "agent_no_answer_call" in result.warnings
    assert result.chosen_ids == ["S1E1", "S1E2"]


def test_the_model_call_cap_ends_the_loop():
    model = Scripted([_call("search_papers", {"query": f"q{i}"}, f"c{i}") for i in range(5)])
    turn, _ = _turn(model, max_model_calls=2)
    result = turn.run("q", [], [])
    assert result.model_calls == 2 and result.stop == "model_call_limit"
    assert "agent_model_call_limit" in result.warnings


def test_the_search_budget_is_enforced_inside_the_loop():
    calls = [_call("search_papers", {"query": f"q{i}"}, f"c{i}") for i in range(4)]
    model = Scripted([*calls, _call("answer", {"question": "q", "evidence_ids": ["S1E1"]}, "c9")])
    turn, corpus = _turn(model, max_searches=3)
    result = turn.run("q", [], [])
    assert len(corpus.queries) == 3 and result.searches == 3
    assert "Search limit reached" in model.seen[4][-1].content


def test_a_malformed_tool_call_is_answered_with_an_error_and_the_turn_goes_on():
    broken = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "search_papers", "args": "{not json", "id": "bad", "error": "x"}
        ],
    )
    model = Scripted(
        [
            broken,
            _call("search_papers", {"query": "q"}),
            _call("answer", {"question": "q", "evidence_ids": ["S1E1"]}, "c3"),
        ]
    )
    turn, _ = _turn(model)
    result = turn.run("q", [], [])
    assert "Malformed call" in model.seen[1][-1].content and result.stop == "answered"


def test_an_unknown_tool_is_answered_and_the_turn_goes_on():
    model = Scripted(
        [
            _call("execute_code", {"code": "1"}),
            _call("answer", {"question": "q", "evidence_ids": []}, "c2"),
        ]
    )
    turn, _ = _turn(model)
    result = turn.run("q", [], [])
    assert "Unknown tool execute_code" in model.seen[1][-1].content and result.stop == "answered"


def test_a_spend_cap_mid_turn_propagates():
    model = Scripted([_call("search_papers", {"query": "q"}), LlmError("llm_spend_cap")])
    turn, _ = _turn(model)
    with pytest.raises(LlmError) as raised:
        turn.run("q", [], [])
    assert raised.value.code == "llm_spend_cap"


def test_identity_names_the_prompt_limits_model_and_generator():
    turn, _ = _turn(Scripted([]))
    identity = turn.identity
    assert identity["agent_prompt"].startswith("research-v1:")
    assert identity["limits"] == {"max_searches": 3, "max_model_calls": 8, "max_sections": 4}
    assert identity["model"] == "scripted" and identity["generator"] == "fixture/grounded-claims"


def test_v2_puts_the_current_question_before_the_carried_passages():
    model = Scripted([_call("answer", {"question": "q", "evidence_ids": []})])
    corpus = Corpus()
    turn = AgentTurn(
        model,
        CorpusRetriever(search=corpus.search, evidence=corpus.evidence),
        lambda item: [],
        FakeGenerator(),
        load_agent_prompt("prompts/agent/research-v2.yaml"),
        limits=AgentLimits(),
        max_chars=6000,
        token_budget=6000,
        count_tokens=lambda text: len(text.split()),
    )
    turn.run("What do they eat?", [], [_evidence("E4")])
    last = model.seen[0][-1].content
    assert last.startswith("Current question: What do they eat?")
    assert last.index("What do they eat?") < last.index("[H1]")


def test_answering_from_seen_puts_the_named_passages_first_then_the_rest():
    model = Scripted(
        [
            _call("search_papers", {"query": "q"}),
            _call("answer", {"question": "q", "evidence_ids": ["S1E2"]}, "c2"),
        ]
    )
    turn, _ = _turn(model)
    turn.answer_from = "seen"
    result = turn.run("q", [], [_evidence("E9")])
    assert result.chosen_ids == ["S1E2", "H1", "S1E1"]
    assert [e.id for e in result.evidence] == ["E1", "E2", "E3"]
    assert turn.identity["answer_from"] == "seen"


def test_verbatim_first_searches_the_question_as_written_and_answers_it_as_written():
    model = Scripted([_call("answer", {"question": "rephrased", "evidence_ids": ["S1E1"]})])
    turn, corpus = _turn(model)
    turn.verbatim_first = True
    result = turn.run("Exact ORB question?", [], [])
    assert corpus.queries == ["Exact ORB question?"] and result.searches == 1
    assert result.standalone == "Exact ORB question?"
    assert "A first search was run with the question as written" in model.seen[0][-1].content
    assert "[S1E1]" in model.seen[0][-1].content


def test_verbatim_first_leaves_a_follow_up_to_the_agent():
    model = Scripted(
        [_call("answer", {"question": "standalone follow-up", "evidence_ids": ["H1"]})]
    )
    turn, corpus = _turn(model)
    turn.verbatim_first = True
    history = [HistoryItem("user", "first"), HistoryItem("assistant", "answer")]
    result = turn.run("And them?", history, [_evidence("E4")])
    assert corpus.queries == [] and result.standalone == "standalone follow-up"


def test_research_v3_tells_the_agent_to_search_unless_carried_passages_state_the_answer():
    v2 = load_agent_prompt("prompts/agent/research-v2.yaml")
    v3 = load_agent_prompt("prompts/agent/research-v3.yaml")
    assert v3.name == "research-v3"
    assert "When unsure, search." in v3.system and "When unsure" not in v2.system
    assert v3.user == v2.user


class VerdictChat:
    """The checker's chat: one scripted verdict list per call."""

    model = type("M", (), {"key": "scripted-checker"})()

    def __init__(self, verdicts: list[list[bool]]) -> None:
        self.verdicts = list(verdicts)

    def complete(self, messages, *, purpose, schema=None, request_id=None, run_id=None):
        import json

        from copilot.models.llm import ChatResult
        from copilot.models.spend import Usage

        body = json.dumps(
            {
                "claims": [
                    {"index": i, "reason": "r", "supported": s}
                    for i, s in enumerate(self.verdicts.pop(0))
                ]
            }
        )
        return ChatResult(body, Usage(1, 1, 2), "m", 0.0005, False, 5)


def _checked(verdicts: list[list[bool]]):
    from copilot.agent.check import LedgerChecker, load_check_prompt
    from copilot.models.generation import load_rewrite_prompt

    model = Scripted(
        [
            _call("search_papers", {"query": "q"}),
            _call("answer", {"question": "q", "evidence_ids": ["S1E1", "S1E2"]}, "c2"),
        ]
    )
    turn, _ = _turn(model)
    turn.checker = LedgerChecker(
        VerdictChat(verdicts), load_check_prompt("prompts/agent/check-v1.yaml")
    )
    turn.rewrite_prompt = load_rewrite_prompt("prompts/agent/rewrite-v1.yaml")
    turn.verbatim_first = False
    return turn


def test_the_checker_runs_after_the_answer_and_its_cost_is_counted():
    turn = _checked([[True, False], [True, True]])
    result = turn.run("q", [], [])
    assert result.check is not None and result.check.result == "rewritten"
    assert result.check.calls == 3 and "check" in result.timings
    assert result.cost_usd == pytest.approx(0.002 + result.outcome.cost_usd + 0.001)
    assert turn.identity["check"]["checker"].startswith("scripted-checker#check-v1:")
    assert turn.identity["check"]["rewrite"].startswith("rewrite-v1:")


def test_a_failed_check_warns_and_keeps_the_answer():
    turn = _checked([])
    turn.checker.chat.complete = lambda *a, **k: (_ for _ in ()).throw(LlmError("llm_timeout"))
    result = turn.run("q", [], [])
    assert result.check.result == "failed" and "agent_check_failed" in result.warnings
    assert len(result.outcome.answer.claims) == 2


def test_no_checker_means_no_check_and_no_timing():
    turn, _ = _turn(Scripted([_call("answer", {"question": "q", "evidence_ids": []})]))
    result = turn.run("q", [], [])
    assert result.check is None and "check" not in result.timings
    assert turn.identity["check"] is None


def test_the_turn_streams_each_search_then_the_evidence_then_the_answer():
    model = Scripted(
        [
            _call("search_papers", {"query": "first"}),
            _call("search_papers", {"query": "second"}, "c2"),
            _call("answer", {"question": "q", "evidence_ids": ["S1E1"]}, "c3"),
        ]
    )
    turn, _ = _turn(model)
    updates = list(turn.iter_steps("q", [], []))
    assert [(u.stage, u.query) for u in updates] == [
        ("search", "first"),
        ("search", "second"),
        ("retrieve", None),
        ("generate", None),
    ]
    assert updates[2].queries == ["first", "second"] and updates[2].evidence
    assert updates[-1].result is not None and updates[-1].result.queries == ["first", "second"]


def test_a_verbatim_first_search_is_streamed_first():
    model = Scripted([_call("answer", {"question": "q", "evidence_ids": ["S1E1"]})])
    turn, _ = _turn(model)
    turn.verbatim_first = True
    updates = list(turn.iter_steps("Does SimCSE use dropout?", [], []))
    assert (updates[0].stage, updates[0].query) == ("search", "Does SimCSE use dropout?")


def test_references_merge_across_searches_in_the_order_first_seen():
    from copilot.agent.tools import AgentTools
    from copilot.contracts import Reference

    a, b, c = uuid4(), uuid4(), uuid4()
    rankings = iter(
        [
            [
                Reference(paper_id=a, rank=1, read=True, title="A"),
                Reference(paper_id=b, rank=2, read=False, title="B"),
            ],
            [
                Reference(paper_id=b, rank=1, read=True, title="B"),
                Reference(paper_id=c, rank=2, read=False, title="C"),
            ],
        ]
    )
    tools = AgentTools(
        CorpusRetriever(
            search=lambda query: PaperHits("rel", [], [], next(rankings)),
            evidence=lambda query, papers, release: [],
        ),
        lambda item: [],
        AgentLimits(),
        6000,
    )
    tools.search_papers("one")
    tools.search_papers("two")
    assert [(str(r.paper_id), r.rank, r.read) for r in tools.references] == [
        (str(a), 1, True),
        (str(b), 2, True),
        (str(c), 3, False),
    ]
