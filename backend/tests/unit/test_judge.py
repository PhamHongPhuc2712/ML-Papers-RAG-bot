"""The answer-equivalence judge: reason before verdict, one retry, then a typed error."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from copilot.evaluation.judge import (
    JUDGE_PROMPT_PATH,
    AnswerEquivalenceJudge,
    JudgeError,
    JudgeVerdict,
    RecordedJudge,
    build_judge_messages,
    load_judge_prompt,
    parse_verdict,
)
from copilot.models.llm import ChatClient
from copilot.models.spend import MemoryLedger, load_llm_config

CONFIG = load_llm_config("configs/llm.yaml")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"
KEY = SecretStr("sk-test-must-never-appear")


def _body(content: str) -> dict:
    body = json.loads((FIXTURES / "openai_chat.json").read_text())
    body["choices"][0]["message"]["content"] = content
    return body


def _judge(replies: list[str]) -> tuple[AnswerEquivalenceJudge, list[dict]]:
    queue = list(replies)
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=_body(queue.pop(0)))

    model = CONFIG.models["openai-gpt-6-luna-low"]
    client = ChatClient(
        model,
        CONFIG.providers[model.provider],
        KEY,
        MemoryLedger(daily_cap_usd=Decimal("1.00")),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )
    return AnswerEquivalenceJudge(client, load_judge_prompt(JUDGE_PROMPT_PATH)), calls


YES = json.dumps({"reason": "Both say the loss is contrastive.", "equivalent": True})
NO = json.dumps({"reason": "The answer names a different objective.", "equivalent": False})


def test_the_schema_puts_the_reason_before_the_verdict():
    assert list(JudgeVerdict.__dataclass_fields__) == ["reason", "equivalent"]
    prompt = load_judge_prompt(JUDGE_PROMPT_PATH)
    assert prompt.name == "answer-equivalence-v1" and len(prompt.sha256) == 64
    # The output shape in the rubric names `reason` before `equivalent`, on purpose.
    assert prompt.system.index('"reason"') < prompt.system.index('"equivalent"')
    assert "haiku.rag" in prompt.attribution
    messages = build_judge_messages(prompt, "Q?", "the reference", "the answer")
    user = messages[1]["content"]
    assert user.index("Q?") < user.index("the reference") < user.index("the answer")


def test_verdicts_parse_from_plain_or_fenced_json_and_reject_anything_else():
    assert parse_verdict(YES) == JudgeVerdict("Both say the loss is contrastive.", True)
    assert parse_verdict("```json\n" + NO + "\n```").equivalent is False
    for bad in (
        "yes",
        json.dumps({"equivalent": True}),
        json.dumps({"reason": "r", "equivalent": "yes"}),
    ):
        with pytest.raises(JudgeError, match="verdict_invalid|verdict_unparseable"):
            parse_verdict(bad)


def test_a_judgement_is_one_call_with_its_cost_and_the_rubric_digest_in_its_identity():
    judge, calls = _judge([YES])
    outcome = judge.judge("Q?", "reference", "answer")
    assert outcome.verdict == JudgeVerdict("Both say the loss is contrastive.", True)
    assert outcome.calls == 1 and outcome.cost_usd > 0 and outcome.served_model == "gpt-6-luna"
    assert judge.identity.startswith("openai-gpt-6-luna-low=gpt-6-luna#answer-equivalence-v1:")
    assert calls[0]["reasoning_effort"] == "low"


def test_a_malformed_verdict_is_retried_once_then_a_judge_error():
    judge, calls = _judge(["I think it is fine.", YES])
    outcome = judge.judge("Q?", "reference", "answer")
    assert outcome.verdict.equivalent is True and outcome.calls == 2 and outcome.retried
    assert len(calls) == 2
    judge, _ = _judge(["nope", "still nope"])
    with pytest.raises(JudgeError, match="judge_error"):
        judge.judge("Q?", "reference", "answer")


def test_the_recorded_judge_answers_by_question_through_the_same_parser():
    recorded = RecordedJudge({"Q?": YES, "Other?": "garbage"})
    assert recorded.judge("Q?", "r", "a").verdict.equivalent is True
    with pytest.raises(JudgeError, match="judge_error"):
        recorded.judge("Other?", "r", "a")
    with pytest.raises(JudgeError, match="judge_unrecorded"):
        recorded.judge("Unknown?", "r", "a")


def test_a_verdict_with_unescaped_latex_backslashes_still_parses():
    text = '{"reason": "Both give \\(\\phi_{AB} = e \\bar{\\Phi}(t_f)/t_f\\).", "equivalent": true}'
    verdict = parse_verdict(text)
    assert verdict.equivalent is True and "\\phi" in verdict.reason
    # A real escape is left alone.
    assert parse_verdict('{"reason": "line\\nbreak", "equivalent": false}').reason == "line\nbreak"


def test_latex_escapes_that_happen_to_be_json_escapes_are_kept_literal():
    text = (
        '{"reason": "The angle is \\theta and the rate \\nu, not \\upsilon.", "equivalent": false}'
    )
    verdict = parse_verdict(text)
    assert "\\theta" in verdict.reason and "\\nu" in verdict.reason and verdict.equivalent is False
