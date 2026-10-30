"""The faithfulness score: claims checked against their cited passages, for every variant."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from copilot.agent.check import load_check_prompt
from copilot.contracts import Claim, Evidence, GroundedAnswer
from copilot.evaluation.faithfulness import (
    FaithfulnessScorer,
    faithfulness_input,
    faithfulness_metrics,
    paired_rate_difference,
    rejudge_agreement,
    run_faithfulness_rejudge,
)
from copilot.models.llm import ChatResult, LlmError
from copilot.models.spend import Usage

PROMPT = load_check_prompt("prompts/judge/claim-support-v1.yaml")


def _evidence(label: str) -> Evidence:
    return Evidence(
        id=label,
        source_kind="paper",
        source_id=uuid4(),
        version_id="v1",
        chunk_id=uuid4(),
        text=f"Passage {label}.",
        section="Method",
    )


def _answer(n: int) -> GroundedAnswer:
    return GroundedAnswer(
        claims=[Claim(text=f"C{i}", evidence_ids=["E1"]) for i in range(n)],
        insufficient_evidence=n == 0,
    )


class Chat:
    model = type("M", (), {"key": "scorer"})()

    def __init__(self, replies: list[list[bool] | Exception]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages, *, purpose, schema=None, request_id=None, run_id=None):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        body = json.dumps(
            {"claims": [{"index": i, "reason": "", "supported": v} for i, v in enumerate(reply)]}
        )
        return ChatResult(body, Usage(1, 1, 2), "m", 0.0002, False, 5)


def test_a_score_counts_supported_claims():
    score = FaithfulnessScorer(Chat([[True, False, True]]), PROMPT).score(
        "q", _answer(3), [_evidence("E1")], request_id=None, run_id=None
    )
    assert (score.claims, score.supported) == (3, 2)
    assert score.rate == pytest.approx(2 / 3) and score.verdicts == (True, False, True)
    assert score.fields()["supported_rate"] == pytest.approx(2 / 3)


def test_an_insufficient_answer_makes_no_scorer_call():
    chat = Chat([])
    score = FaithfulnessScorer(chat, PROMPT).score(
        "q", _answer(0), [], request_id=None, run_id=None
    )
    assert score.claims == 0 and score.rate is None and chat.calls == 0


def test_a_scorer_failure_is_recorded_not_scored():
    score = FaithfulnessScorer(Chat([LlmError("llm_timeout")]), PROMPT).score(
        "q", _answer(2), [_evidence("E1")], request_id=None, run_id=None
    )
    assert score.error == "faithfulness:llm_timeout" and score.rate is None
    assert score.fields()["supported_claims"] is None


def test_supported_rate_is_the_mean_over_answered_cases_only():
    rows = [
        {"claims": 2, "supported_rate": 0.5},
        {"claims": 0, "supported_rate": None},
        {"claims": 4, "supported_rate": 1.0},
        {"claims": 1, "supported_rate": None, "faithfulness_error": "faithfulness:x"},
    ]
    metrics = faithfulness_metrics(rows)
    assert metrics["supported_rate"] == pytest.approx(0.75)
    assert metrics["fully_supported"] == pytest.approx(0.5)
    assert metrics["scored"] == 2 and metrics["errors"] == 1 and metrics["claims"] == 7


def test_the_supported_rate_pair_bootstraps_over_conversations():
    treated = [
        {"task_id": f"t{i}", "conversation_id": f"c{i % 3}", "supported_rate": 1.0}
        for i in range(9)
    ]
    baseline = [
        {"task_id": f"t{i}", "conversation_id": f"c{i % 3}", "supported_rate": 0.5 + 0.1 * (i % 3)}
        for i in range(9)
    ]
    baseline[0]["supported_rate"] = None
    pair = paired_rate_difference(treated, baseline, key="task_id", by="conversation_id")
    assert pair["items"] == 8 and pair["groups"] == 3
    assert pair["interval"][0] <= pair["difference"] <= pair["interval"][1]
    assert pair["interval"][0] > 0


def test_a_pair_with_nothing_scored_says_so():
    pair = paired_rate_difference([{"query_id": "a", "supported_rate": None}], [], key="query_id")
    assert pair["items"] == 0 and pair["difference"] is None


def test_rejudge_reports_kappa():
    agreement = rejudge_agreement([True, True, False, False], [True, False, False, False])
    assert agreement["agreement"] == 0.75 and agreement["kappa"] == pytest.approx(0.5)


def test_the_input_keeps_only_the_cited_passages():
    answer = GroundedAnswer(
        claims=[Claim(text="A", evidence_ids=["E2"])], insufficient_evidence=False
    )
    data = faithfulness_input(answer, {"E1": _evidence("E1"), "E2": _evidence("E2")})
    assert [e["id"] for e in data["evidence"]] == ["E2"] and data["claims"][0]["text"] == "A"


def test_the_rejudge_rescores_from_the_text_rows(tmp_path):
    run = tmp_path / "run.agent-check"
    run.mkdir()
    text_dir = tmp_path / "text"
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "variant": "agent-check",
                "text_dir": str(text_dir),
                "faithfulness": {"key": "openai-gpt-6-luna-low", "identity": "x"},
            }
        )
    )
    answer = GroundedAnswer(
        claims=[Claim(text="A", evidence_ids=["E1"]), Claim(text="B", evidence_ids=["E1"])],
        insufficient_evidence=False,
    )
    row = {
        "query_id": "q1",
        "question": "q",
        "supported_verdicts": [True, True],
        "faithfulness_input": faithfulness_input(answer, {"E1": _evidence("E1")}),
    }
    (text_dir / "r1" / "agent-check").mkdir(parents=True)
    (text_dir / "r1" / "agent-check" / "cases.jsonl").write_text(json.dumps(row) + "\n")
    result = run_faithfulness_rejudge(
        run,
        scorer_key="openai-gpt-6-sol",
        out=tmp_path / "sheet.jsonl",
        max_spend_usd=1.0,
        scorer=FaithfulnessScorer(Chat([[True, False]]), PROMPT),
    )
    assert result["answers"] == 1 and result["claims"] == 2 and result["agreement"] == 0.5


def test_a_scored_case_row_carries_the_faithfulness_columns_and_its_text_the_input():
    from copilot.evaluation.judge import RecordedJudge
    from copilot.evaluation.qa import QaCase, QaVariant, _empty_result, _score_outcome
    from copilot.models.generation import GenerationOutcome

    case = QaCase("q1", "f1", "abstractive", "q", "ref", None)
    result = _empty_result(case, QaVariant("flat"), [])
    scored = _score_outcome(
        result,
        case,
        GenerationOutcome(_answer(2)),
        {"E1": _evidence("E1")},
        RecordedJudge({"q": '{"reason": "", "equivalent": true}'}),
        run_id="r",
        started=0.0,
        clock=lambda: 1.0,
        faithfulness=FaithfulnessScorer(Chat([[True, False]]), PROMPT),
    )
    row = scored.row()
    assert row["supported_rate"] == 0.5 and row["supported_verdicts"] == [True, False]
    assert "faithfulness_input" not in row and "faithfulness_input" in scored.text_row()


def test_without_a_scorer_the_case_row_is_unchanged():
    from copilot.evaluation.judge import RecordedJudge
    from copilot.evaluation.qa import QaCase, QaVariant, _empty_result, _score_outcome
    from copilot.models.generation import GenerationOutcome

    case = QaCase("q1", "f1", "abstractive", "q", "ref", None)
    scored = _score_outcome(
        _empty_result(case, QaVariant("flat"), []),
        case,
        GenerationOutcome(_answer(1)),
        {"E1": _evidence("E1")},
        RecordedJudge({"q": '{"reason": "", "equivalent": true}'}),
        run_id="r",
        started=0.0,
        clock=lambda: 1.0,
    )
    assert "supported_rate" not in scored.row()
