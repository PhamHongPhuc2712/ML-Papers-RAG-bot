"""The research agent's self-check: verify each claim, rewrite once, drop what still fails."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from copilot.agent.check import (
    SUPPORT_SCHEMA,
    CheckError,
    LedgerChecker,
    load_check_prompt,
    parse_support,
    self_check,
)
from copilot.contracts import Claim, Evidence, GroundedAnswer
from copilot.models.generation import (
    GenerationOutcome,
    _Attempt,
    _Grounded,
    load_generation_prompt,
    load_rewrite_prompt,
)
from copilot.models.llm import ChatResult, LlmError
from copilot.models.spend import Usage

CHECK = load_check_prompt("prompts/agent/check-v1.yaml")
REWRITE = load_rewrite_prompt("prompts/agent/rewrite-v1.yaml")


def _evidence(label: str, *, abstract: bool = False) -> Evidence:
    return Evidence(
        id=label,
        source_kind="paper",
        source_id=uuid4(),
        version_id="v1",
        chunk_id=None if abstract else uuid4(),
        text=f"Passage {label}.",
        section="Abstract" if abstract else "Method",
        abstract_only=abstract,
    )


EVIDENCE = [_evidence("E1"), _evidence("E2")]


def _answer(*texts: str) -> GroundedAnswer:
    return GroundedAnswer(
        claims=[Claim(text=t, evidence_ids=[f"E{i % 2 + 1}"]) for i, t in enumerate(texts)],
        insufficient_evidence=not texts,
    )


def _verdicts(*supported: bool) -> str:
    return json.dumps(
        {"claims": [{"index": i, "reason": "r", "supported": s} for i, s in enumerate(supported)]}
    )


class ScriptedChat:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, messages, *, purpose, schema=None, request_id=None, run_id=None):
        self.calls.append({"messages": messages, "schema": schema, "purpose": purpose})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(reply, Usage(1, 1, 2), "m", 0.001, False, 5)


class ScriptedGenerator(_Grounded):
    """Answers each rewrite with the next scripted answer (a GroundedAnswer or an error)."""

    def __init__(self, rewrites: list[GroundedAnswer | Exception]) -> None:
        super().__init__(load_generation_prompt())
        self.rewrites = list(rewrites)
        self.messages: list[list[dict[str, str]]] = []

    def _attempt(self, messages, *, request_id, run_id):
        self.messages.append(messages)
        reply = self.rewrites.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Attempt(reply.model_dump_json(), 0.002, "m", False)


def _run(first, verdicts, *, rewrites=(), evidence=EVIDENCE):
    chat = ScriptedChat(list(verdicts))
    generator = ScriptedGenerator(list(rewrites))
    outcome, record = self_check(
        "q",
        evidence,
        GenerationOutcome(first, calls=1, cost_usd=0.003),
        checker=LedgerChecker(chat, CHECK),
        generator=generator,
        rewrite=REWRITE,
    )
    return outcome, record, chat, generator


def _texts(outcome: GenerationOutcome) -> list[str]:
    return [c.text for c in outcome.answer.claims]


def test_all_supported_leaves_the_answer_alone():
    outcome, record, chat, _ = _run(_answer("A", "B"), [_verdicts(True, True)])
    assert record.result == "passed" and record.calls == 1 and _texts(outcome) == ["A", "B"]
    assert chat.calls[0]["schema"] is SUPPORT_SCHEMA


def test_an_unsupported_claim_triggers_one_rewrite_and_a_second_check():
    outcome, record, chat, generator = _run(
        _answer("A", "B"),
        [_verdicts(True, False), _verdicts(True, True)],
        rewrites=[_answer("A", "B fixed")],
    )
    assert record.result == "rewritten" and record.calls == 3
    assert _texts(outcome) == ["A", "B fixed"] and record.first_unsupported == 1
    assert record.cost_usd == pytest.approx(0.001 + 0.002 + 0.001)
    assert outcome.cost_usd == 0.003 and outcome.calls == 1
    rewrite_message = generator.messages[0][-1]["content"]
    assert "- B (why: r)" in rewrite_message and "E1, E2" in rewrite_message


def test_still_unsupported_after_the_rewrite_is_dropped():
    outcome, record, _, _ = _run(
        _answer("A", "B"),
        [_verdicts(True, False), _verdicts(True, False)],
        rewrites=[_answer("A", "B2")],
    )
    assert _texts(outcome) == ["A"] and record.final_unsupported_dropped == 1
    assert record.result == "rewritten" and not outcome.answer.insufficient_evidence


def test_all_claims_dropped_becomes_insufficiency():
    outcome, record, _, _ = _run(
        _answer("B"), [_verdicts(False), _verdicts(False)], rewrites=[_answer("B2")]
    )
    assert outcome.answer.insufficient_evidence and outcome.answer.claims == []
    assert record.result == "emptied"


def test_a_rewrite_that_gives_up_becomes_insufficiency():
    outcome, record, chat, _ = _run(_answer("B"), [_verdicts(False)], rewrites=[_answer()])
    assert outcome.answer.insufficient_evidence and record.result == "emptied"
    assert len(chat.calls) == 1


def test_missing_verdicts_count_as_unsupported():
    text = '{"claims": [{"index": 0, "reason": "", "supported": true}]}'
    assert [v.supported for v in parse_support(text, claims=2)] == [True, False]


def test_repeated_and_out_of_range_verdicts_are_ignored():
    text = json.dumps(
        {
            "claims": [
                {"index": 0, "reason": "", "supported": False},
                {"index": 0, "reason": "", "supported": True},
                {"index": 7, "reason": "", "supported": True},
            ]
        }
    )
    assert [v.supported for v in parse_support(text, claims=1)] == [False]
    with pytest.raises(CheckError):
        parse_support("not json", claims=1)


def test_a_checker_failure_keeps_the_answer():
    for error in (LlmError("llm_timeout"), "not json"):
        outcome, record, _, _ = _run(_answer("A", "B"), [error])
        assert record.result == "failed" and _texts(outcome) == ["A", "B"]


def test_a_failed_rewrite_drops_the_unsupported_claims():
    outcome, record, _, _ = _run(
        _answer("A", "B"), [_verdicts(True, False)], rewrites=[LlmError("llm_timeout")]
    )
    assert record.result == "dropped" and _texts(outcome) == ["A"]


def test_a_rewrite_citing_an_unknown_id_twice_drops_the_unsupported_claims():
    bad = GroundedAnswer(claims=[Claim(text="A", evidence_ids=["E9"])], insufficient_evidence=False)
    outcome, record, _, generator = _run(
        _answer("A", "B"), [_verdicts(True, False)], rewrites=[bad, bad]
    )
    assert record.result == "dropped" and _texts(outcome) == ["A"]
    assert len(generator.messages) == 2


def test_insufficiency_is_never_checked():
    outcome, record, chat, _ = _run(_answer(), [])
    assert record.result == "skipped" and record.calls == 0 and chat.calls == []


def test_abstract_evidence_is_shown_to_the_checker():
    evidence = [_evidence("E1", abstract=True), _evidence("E2")]
    _, _, chat, _ = _run(_answer("A"), [_verdicts(True)], evidence=evidence)
    user = chat.calls[0]["messages"][1]["content"]
    assert "(abstract only)" in user and "Claim 0: A" in user and "Passage E1." in user


def test_the_support_schema_has_a_title():
    assert SUPPORT_SCHEMA["title"] == "claim_support"
