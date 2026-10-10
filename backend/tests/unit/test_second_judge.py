"""A second judge over a run's answers: agreement, abstentions skipped, no text in the report."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import SecretStr

from copilot.evaluation.judge import JUDGE_PROMPT_PATH, AnswerEquivalenceJudge, load_judge_prompt
from copilot.evaluation.qa import QaError, run_second_judge
from copilot.models.llm import ChatClient
from copilot.models.spend import MemoryLedger, load_llm_config

CONFIG = load_llm_config("configs/llm.yaml")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"
KEY = SecretStr("sk-test-must-never-appear")
ROOT_PROMPTS = Path(__file__).resolve().parents[3] / "prompts" / "judge"
YES = json.dumps({"reason": "Same claim.", "equivalent": True})
NO = json.dumps({"reason": "Different claim.", "equivalent": False})


def _run(tmp_path: Path) -> Path:
    """A recorded run: three judged answers, one abstention, text under a DATA_DIR-like dir."""

    text_dir = tmp_path / "text"
    run_dir = tmp_path / "reports" / "run.flat"
    run_dir.mkdir(parents=True)
    (text_dir / "run" / "flat").mkdir(parents=True)
    rows = [
        {
            "query_id": "q1",
            "question": "Q1?",
            "reference": "R1",
            "answer": "A1",
            "equivalent": True,
            "abstained": False,
        },
        {
            "query_id": "q2",
            "question": "Q2?",
            "reference": "R2",
            "answer": "A2",
            "equivalent": False,
            "abstained": False,
        },
        {
            "query_id": "q3",
            "question": "Q3?",
            "reference": "R3",
            "answer": "A3",
            "equivalent": True,
            "abstained": False,
        },
        {
            "query_id": "q4",
            "question": "Q4?",
            "reference": "R4",
            "answer": None,
            "equivalent": False,
            "abstained": True,
        },
    ]
    (text_dir / "run" / "flat" / "cases.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "query_id": [r["query_id"] for r in rows],
                "equivalent": [r["equivalent"] for r in rows],
            }
        ),
        run_dir / "cases.parquet",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "variant": "flat",
                "text_dir": str(text_dir),
                "judge": {"key": "openai-gpt-6-luna-low", "identity": "luna"},
                "prompts": {"judge": {"path": str(JUDGE_PROMPT_PATH)}},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _judge(replies: list[str]) -> tuple[AnswerEquivalenceJudge, MemoryLedger, list[dict]]:
    queue = list(replies)
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        body = json.loads((FIXTURES / "openai_chat.json").read_text())
        body["choices"][0]["message"]["content"] = queue.pop(0)
        return httpx.Response(200, json=body)

    model = CONFIG.models["openai-gpt-6-sol"]
    ledger = MemoryLedger(daily_cap_usd=Decimal("1.00"))
    client = ChatClient(
        model,
        CONFIG.providers[model.provider],
        KEY,
        ledger,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    return AnswerEquivalenceJudge(client, load_judge_prompt(JUDGE_PROMPT_PATH)), ledger, calls


def test_a_second_judge_measures_agreement_and_skips_abstentions(tmp_path):
    run_dir = _run(tmp_path)
    # Agrees on q1 and q2, disagrees on q3; q4 (abstained) is never sent.
    judge, ledger, calls = _judge([YES, NO, NO])
    out = tmp_path / "text" / "labels" / "run.flat.openai-gpt-6-sol.jsonl"
    summary = run_second_judge(
        run_dir,
        judge_key="openai-gpt-6-sol",
        out=out,
        max_spend_usd=1.0,
        judge=judge,
        ledger=ledger,
    )
    assert len(calls) == 3 and all("Q4?" not in json.dumps(call) for call in calls)
    assert summary["cases"] == 3 and summary["judged"] == 3 and summary["judge_errors"] == 0
    assert summary["agreement"]["judged"] == 3
    assert summary["agreement"]["agreement"] == pytest.approx(2 / 3)
    assert summary["disagreements"] == ["q3"]
    assert summary["cost_usd"] > 0 and summary["stopped"] is None
    # The sheet has the verdicts beside the text, under the text directory.
    sheet = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["equivalent"] for row in sheet] == [True, False, False]
    assert sheet[0]["reason"] == "Same claim."
    # The report-side summary carries no question, reference or answer text.
    written = (run_dir / "second-judge.openai-gpt-6-sol.json").read_text(encoding="utf-8")
    assert json.loads(written) == summary
    for text in ("Q1?", "R1", "A1", "Same claim."):
        assert text not in written


def test_the_same_judge_and_an_oversized_budget_are_refused(tmp_path):
    run_dir = _run(tmp_path)
    judge, ledger, _ = _judge([])
    with pytest.raises(QaError, match="same_judge"):
        run_second_judge(
            run_dir,
            judge_key="openai-gpt-6-luna-low",
            out=tmp_path / "x.jsonl",
            max_spend_usd=1.0,
            judge=judge,
            ledger=ledger,
        )
    with pytest.raises(QaError, match="run_budget_exceeded"):
        run_second_judge(
            run_dir,
            judge_key="openai-gpt-6-sol",
            out=tmp_path / "x.jsonl",
            max_spend_usd=0.001,
            judge=judge,
            ledger=ledger,
        )


def test_a_judge_error_is_recorded_not_fatal(tmp_path):
    run_dir = _run(tmp_path)
    # q1 malformed twice (retry exhausted), q2 and q3 answered.
    judge, ledger, calls = _judge(["not json", "still not json", NO, YES])
    out = tmp_path / "text" / "labels" / "sheet.jsonl"
    summary = run_second_judge(
        run_dir,
        judge_key="openai-gpt-6-sol",
        out=out,
        max_spend_usd=1.0,
        judge=judge,
        ledger=ledger,
    )
    assert summary["judge_errors"] == 1 and summary["judged"] == 2 and len(calls) == 4
    sheet = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sheet[0]["judge_error"] == "judge_error" and sheet[0]["equivalent"] is None
    assert summary["agreement"]["judged"] == 2 and summary["disagreements"] == []


def test_another_rubric_lets_the_same_judge_re_read_its_own_verdicts(tmp_path):
    run_dir = _run(tmp_path)
    rubric = ROOT_PROMPTS / "answer-equivalence-haiku-v1.yaml"
    judge, ledger, calls = _judge([YES, YES, YES])
    summary = run_second_judge(
        run_dir,
        judge_key="openai-gpt-6-luna-low",
        out=tmp_path / "text" / "labels" / "rubric.jsonl",
        max_spend_usd=1.0,
        judge=judge,
        ledger=ledger,
        prompt=rubric,
    )
    assert summary["second_judge"]["prompt"] == str(rubric)
    assert summary["agreement"]["judged"] == 3 and summary["disagreements"] == ["q2"]
    assert (
        run_dir / "second-judge.openai-gpt-6-luna-low.answer-equivalence-haiku-v1.json"
    ).exists()
