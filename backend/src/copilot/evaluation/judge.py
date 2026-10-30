"""The answer-equivalence judge (ORB plan O5; benchmarks plan E6).

One hosted-model call compares a candidate answer with the reference answer under a
pinned rubric and returns a reason first and a verdict second, as JSON. A malformed
verdict is retried once, with a reminder that changes the request so a cached
malformed answer is not served again; a second failure is a typed ``judge_error``,
never a verdict coerced from prose. The rubric's digest is part of the judge's
identity and of every call's cache key.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import yaml

from ..models.generation import loads_lenient
from ..models.llm import ChatClient

JUDGE_PROMPT_PATH = Path("prompts/judge/answer-equivalence-v1.yaml")
_PLACEHOLDER = re.compile(r"\{(question|reference|answer)\}")
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
RETRY_REMINDER = (
    "Your previous reply was not a JSON object with a reason and a verdict. Reply with the"
    ' JSON object only: {"reason": "...", "equivalent": true or false}.'
)


class JudgeError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class JudgeVerdict:
    """Reason first, on purpose: the verdict follows from it."""

    reason: str
    equivalent: bool


@dataclass(frozen=True)
class JudgePrompt:
    name: str
    sha256: str
    system: str
    user: str
    attribution: str


def load_judge_prompt(path: str | Path = JUDGE_PROMPT_PATH) -> JudgePrompt:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    try:
        prompt = JudgePrompt(
            name=str(raw["name"]),
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            system=str(raw["system"]),
            user=str(raw["user"]),
            attribution=str(raw.get("attribution", "")),
        )
    except KeyError as error:
        raise ValueError(f"prompt_invalid:{error.args[0]}") from error
    for placeholder in ("{question}", "{reference}", "{answer}"):
        if placeholder not in prompt.user:
            raise ValueError(f"prompt_invalid:user.{placeholder}")
    return prompt


def build_judge_messages(
    prompt: JudgePrompt, question: str, reference: str, answer: str
) -> list[dict[str, str]]:
    values = {"question": question, "reference": reference, "answer": answer}
    user = _PLACEHOLDER.sub(lambda match: values[match.group(1)], prompt.user)
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]


def parse_verdict(text: str) -> JudgeVerdict:
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        payload = loads_lenient(body)
    except json.JSONDecodeError as error:
        raise JudgeError("verdict_unparseable", str(error)[:60]) from error
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("reason"), str)
        or not isinstance(payload.get("equivalent"), bool)
    ):
        raise JudgeError("verdict_invalid", "reason or equivalent")
    return JudgeVerdict(reason=payload["reason"], equivalent=payload["equivalent"])


@dataclass
class JudgeOutcome:
    verdict: JudgeVerdict
    calls: int = 0
    retried: bool = False
    cost_usd: float = 0.0
    served_model: str = ""
    cached: bool = False


class AnswerEquivalenceJudge:
    """The rubric over a configured hosted model, every call under the spend ledger."""

    def __init__(
        self,
        client: ChatClient,
        prompt: JudgePrompt,
        *,
        timeout: float = 60.0,
        purpose: str = "evaluation",
    ) -> None:
        self._client = client
        self.prompt = prompt
        self.model = client.model
        self.timeout = timeout
        self.purpose = purpose
        self.identity = f"{self.model.key}={self.model.model}#{prompt.name}:{prompt.sha256[:12]}"

    def close(self) -> None:
        self._client.close()

    def judge(
        self,
        question: str,
        reference: str,
        answer: str,
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> JudgeOutcome:
        messages = build_judge_messages(self.prompt, question, reference, answer)
        outcome = JudgeOutcome(JudgeVerdict("", False))
        last: JudgeError | None = None
        for attempt in range(2):
            if attempt == 1:
                messages = [*messages, {"role": "user", "content": RETRY_REMINDER}]
                outcome.retried = True
            result = self._client.complete(
                messages,
                timeout=self.timeout,
                purpose=self.purpose,
                request_id=request_id,
                run_id=run_id,
            )
            outcome.calls += 1
            outcome.cost_usd += result.cost_usd
            outcome.served_model = result.served_model
            outcome.cached = outcome.cached or result.cached
            try:
                outcome.verdict = parse_verdict(result.text)
                return outcome
            except JudgeError as error:
                last = error
        raise JudgeError("judge_error", last.code if last else "")


class RecordedJudge:
    """Verdicts recorded per question, through the same parser: the smoke set's judge."""

    identity = "fixture/answer-equivalence"

    def __init__(self, responses: Mapping[str, str]) -> None:
        self._responses = dict(responses)

    def judge(
        self,
        question: str,
        reference: str,
        answer: str,
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> JudgeOutcome:
        text = self._responses.get(question)
        if text is None:
            raise JudgeError("judge_unrecorded", question[:40])
        outcome = JudgeOutcome(JudgeVerdict("", False), served_model="fixture")
        last: JudgeError | None = None
        for _ in range(2):
            outcome.calls += 1
            try:
                outcome.verdict = parse_verdict(text)
                return outcome
            except JudgeError as error:
                last = error
                outcome.retried = True
        raise JudgeError("judge_error", last.code if last else "")
