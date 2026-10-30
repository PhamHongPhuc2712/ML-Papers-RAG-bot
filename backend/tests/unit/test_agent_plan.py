"""The research agent's optional planning step (agent follow-ups, Task 4)."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from copilot.agent.plan import LedgerPlanner, Plan, PlanError, load_plan_prompt, parse_plan
from copilot.agent.tools import AgentLimits
from copilot.agent.turn import AgentTurn, load_agent_prompt
from copilot.chat.retriever import CorpusRetriever, PaperHits
from copilot.contracts import Evidence
from copilot.models.generation import FakeGenerator
from copilot.models.llm import ChatResult, LlmError
from copilot.models.spend import Usage


class Scripted:
    identity = "scripted"

    def __init__(self, replies: list[AIMessage]) -> None:
        self.replies = list(replies)
        self.seen: list[list] = []

    def step(self, messages, tools, *, request_id=None, run_id=None):
        self.seen.append(list(messages))
        return self.replies.pop(0), 0.001


def _evidence() -> Evidence:
    return Evidence(
        id="E1",
        source_kind="paper",
        source_id=uuid4(),
        version_id=str(uuid4()),
        chunk_id=uuid4(),
        text="A passage.",
        section="Body",
    )


def _answer() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "answer", "args": {"question": "q", "evidence_ids": []}, "id": "c1"}],
    )


class ScriptedPlanner:
    identity = "scripted-plan"

    def __init__(self, parts: list[str] | Exception, cost: float = 0.002) -> None:
        self.parts = parts
        self.cost = cost
        self.calls = 0

    def plan(self, question, *, max_subquestions, request_id, run_id):
        self.calls += 1
        if isinstance(self.parts, Exception):
            raise self.parts
        return Plan(tuple(self.parts)[:max_subquestions], self.cost)


def _planned(planner, **fields):
    model = Scripted([_answer()])
    turn = AgentTurn(
        model,
        CorpusRetriever(
            search=lambda query: PaperHits("rel-1", ["p1"], []),
            evidence=lambda query, papers, release: [_evidence()],
        ),
        lambda item: [],
        FakeGenerator(),
        load_agent_prompt("prompts/agent/research-v3.yaml"),
        limits=AgentLimits(max_searches=4),
        max_chars=6000,
        token_budget=6000,
        count_tokens=lambda text: len(text.split()),
        planner=planner,
        **fields,
    )
    return turn, model, None


def test_the_plan_is_shown_to_the_agent_as_parts_to_cover():
    turn, model, _ = _planned(ScriptedPlanner(["X's size", "Y's size"]))
    result = turn.run("Is X bigger than Y?", [], [])
    assert "Parts to cover:\n1. X's size\n2. Y's size" in model.seen[0][-1].content
    assert result.plan == ("X's size", "Y's size")
    assert result.cost_usd == pytest.approx(0.002 + 0.001)
    assert turn.identity["plan"] == {"prompt": "scripted-plan", "max_subquestions": 3}


def test_a_single_part_plan_changes_nothing_the_agent_sees():
    turn, model, _ = _planned(ScriptedPlanner(["Is X big?"]))
    result = turn.run("Is X big?", [], [])
    assert "Parts to cover" not in model.seen[0][-1].content and result.plan == ("Is X big?",)


def test_a_malformed_plan_is_dropped_with_a_warning():
    for error in (LlmError("schema_violation"), PlanError("plan_malformed")):
        turn, model, _ = _planned(ScriptedPlanner(error))
        result = turn.run("Is X bigger than Y?", [], [])
        assert result.plan == () and "agent_plan_failed" in result.warnings
        assert "Parts to cover" not in model.seen[0][-1].content


def test_extra_subquestions_are_cut_to_the_limit():
    turn, _, _ = _planned(ScriptedPlanner(["a", "b", "c"]), max_subquestions=2)
    assert turn.run("q", [], []).plan == ("a", "b")


def test_no_planner_makes_no_plan():
    turn, _, _ = _planned(None)
    result = turn.run("q", [], [])
    assert result.plan == () and turn.identity["plan"] is None


def test_parse_plan_keeps_order_drops_blanks_and_refuses_bad_json():
    text = json.dumps({"subquestions": [" a  b ", "", "c", "d"]})
    assert parse_plan(text, max_subquestions=2) == ("a b", "c")
    for bad in ("not json", "{}", '{"subquestions": "a"}'):
        with pytest.raises(PlanError):
            parse_plan(bad, max_subquestions=3)


class FakeChat:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def complete(self, messages, *, purpose, schema=None, request_id=None, run_id=None):
        self.calls.append({"messages": messages, "purpose": purpose, "schema": schema})
        return ChatResult(self.text, Usage(1, 1, 2), "m", 0.0004, False, 5)


def test_the_ledger_planner_asks_once_with_the_schema_and_the_limit():
    chat = FakeChat(json.dumps({"subquestions": ["a", "b"]}))
    planner = LedgerPlanner(chat, load_plan_prompt("prompts/agent/plan-v1.yaml"))
    plan = planner.plan("Is A older than B?", max_subquestions=3, request_id=None, run_id="r")
    assert plan == Plan(("a", "b"), 0.0004)
    (call,) = chat.calls
    assert call["purpose"] == "evaluation" and call["schema"]["required"] == ["subquestions"]
    assert "at most 3" in call["messages"][0]["content"]
    assert call["messages"][1]["content"] == "Question: Is A older than B?"
    assert planner.identity.startswith("plan-v1:")


def test_the_worst_case_counts_the_planning_call():
    from copilot.agent.tools import AgentLimits
    from copilot.evaluation.qa import agent_turn_worst_usd
    from copilot.models.spend import load_llm_config

    model = load_llm_config("configs/llm.yaml").models["openai-gpt-6-luna"]
    args = (model.price, model.max_output_tokens, 6000, AgentLimits(max_searches=4))
    assert agent_turn_worst_usd(*args, planned=True) > agent_turn_worst_usd(*args)


def test_the_plan_schema_is_accepted_for_structured_output():
    from langchain_openai import ChatOpenAI

    from copilot.agent.plan import PLAN_SCHEMA

    ChatOpenAI(model="m", api_key="sk-test").with_structured_output(
        dict(PLAN_SCHEMA), method="json_schema", strict=True, include_raw=True
    )
