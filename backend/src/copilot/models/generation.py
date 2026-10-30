"""The grounded generator: structured claims with evidence ids, validated before they count.

A hosted chat model is asked for JSON claims, each naming the evidence ids it rests
on. The answer is parsed and checked structurally (``evidence.citations``): an
unknown id, an uncited claim, a duplicate reference, a quote not in its source or a
malformed answer earns exactly one repair call that names what was wrong, and a
second failure is insufficiency, never a silent drop and never an answer with an
invented citation. A question with no evidence is insufficiency without a call.

Prompts are pinned by digest; evidence is data to the model, said once in the system
message. The deterministic ``FakeGenerator`` runs the same validation path in CI,
and on a fixture question invents an id so the repair is exercised.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from ..contracts import Claim, Evidence, GroundedAnswer
from ..evidence.citations import check_claims
from .lc_chat import GROUNDED_ANSWER_SCHEMA, LedgerChatModel
from .llm import ChatClient

PROMPT_PATH = Path("prompts/generate/grounded-claims-v1.yaml")
_PLACEHOLDER = re.compile(r"\{(question|evidence|count|problems|allowed)\}")
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
# Any backslash except one escaping a quote: LaTeX in an answer, written unescaped.
_LONE_BACKSLASH = re.compile(r'\\(?!")')


def loads_lenient(body: str) -> Any:
    """``json.loads``, then once more with every backslash but ``\\"`` escaped.

    Models answering about papers write LaTeX (``\\phi``, ``\\theta``, ``\\nu``) inside
    JSON strings without escaping it. Some of it is invalid JSON (``\\p``), some of it
    is a valid escape that corrupts the text (``\\t``, ``\\n``, ``\\u``). When the strict
    parse fails, every backslash is taken literally, which is what was meant.
    """

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return json.loads(_LONE_BACKSLASH.sub(r"\\\\", body))


class GenerationError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class GenerationPrompt:
    name: str
    sha256: str
    system: str
    user: str
    repair: str


def load_generation_prompt(path: str | Path = PROMPT_PATH) -> GenerationPrompt:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    try:
        prompt = GenerationPrompt(
            name=str(raw["name"]),
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            system=str(raw["system"]),
            user=str(raw["user"]),
            repair=str(raw["repair"]),
        )
    except KeyError as error:
        raise ValueError(f"prompt_invalid:{error.args[0]}") from error
    for placeholder in ("{question}", "{evidence}", "{count}"):
        if placeholder not in prompt.user:
            raise ValueError(f"prompt_invalid:user.{placeholder}")
    for placeholder in ("{problems}", "{allowed}"):
        if placeholder not in prompt.repair:
            raise ValueError(f"prompt_invalid:repair.{placeholder}")
    return prompt


_REWRITE_PLACEHOLDER = re.compile(r"\{(unsupported|allowed)\}")


@dataclass(frozen=True)
class RewritePrompt:
    """The self-check's follow-up to the generator (agent self-check, 2026-10-29)."""

    name: str
    sha256: str
    text: str


def load_rewrite_prompt(path: str | Path) -> RewritePrompt:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    prompt = RewritePrompt(
        name=str(raw["name"]), sha256=hashlib.sha256(raw_bytes).hexdigest(), text=str(raw["text"])
    )
    for placeholder in ("{unsupported}", "{allowed}"):
        if placeholder not in prompt.text:
            raise ValueError(f"prompt_invalid:text.{placeholder}")
    return prompt


def _fill(template: str, values: dict[str, str]) -> str:
    """One pass, so text inserted for one placeholder is never read as another."""

    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def evidence_block(item: Evidence) -> str:
    where = item.section
    if item.page_start is not None:
        pages = (
            f"p. {item.page_start}"
            if item.page_end in (None, item.page_start)
            else f"pp. {item.page_start}-{item.page_end}"
        )
        where = f"{where}, {pages}"
    if item.abstract_only:
        where = f"{where} (abstract only)"
    return f"[{item.id}] ({where})\n{item.text}"


def build_messages(
    prompt: GenerationPrompt, question: str, evidence: Sequence[Evidence]
) -> list[dict[str, str]]:
    block = "\n\n".join(evidence_block(item) for item in evidence)
    user = _fill(
        prompt.user, {"question": question, "evidence": block, "count": str(len(evidence))}
    )
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]


def parse_answer(text: str) -> tuple[list[Claim], bool]:
    """Claims and the insufficiency flag from the model's JSON, fenced or not."""

    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        payload = loads_lenient(body)
    except json.JSONDecodeError as error:
        raise GenerationError("answer_unparseable", str(error)[:60]) from error
    if not isinstance(payload, dict):
        raise GenerationError("answer_invalid", "not an object")
    claims = payload.get("claims")
    insufficient = payload.get("insufficient_evidence")
    if not isinstance(claims, list) or not isinstance(insufficient, bool):
        raise GenerationError("answer_invalid", "claims or insufficient_evidence")
    parsed: list[Claim] = []
    for item in claims:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("text"), str)
            or not isinstance(item.get("evidence_ids"), list)
            or not all(isinstance(value, str) for value in item["evidence_ids"])
        ):
            raise GenerationError("answer_invalid", "claim shape")
        parsed.append(Claim(text=item["text"], evidence_ids=list(item["evidence_ids"])))
    return parsed, insufficient


@dataclass
class GenerationOutcome:
    """An answer and how it was reached; the answer alone is the ``Generator`` contract."""

    answer: GroundedAnswer
    calls: int = 0
    repair_attempted: bool = False
    # Why the answer is insufficiency, when it is one: the last problem seen.
    reason: str | None = None
    # Every problem the first answer had; empty means provenance was valid outright.
    problems: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    served_models: list[str] = field(default_factory=list)
    cached: bool = False

    @property
    def provenance_valid(self) -> bool:
        return not self.problems and not self.repair_attempted


@dataclass(frozen=True)
class _Attempt:
    text: str
    cost_usd: float
    served_model: str
    cached: bool


class _Grounded:
    """The validation loop both generators share: one answer, one repair, then insufficiency."""

    def __init__(self, prompt: GenerationPrompt) -> None:
        self.prompt = prompt

    def _attempt(
        self, messages: list[dict[str, str]], *, request_id: UUID | None, run_id: str | None
    ) -> _Attempt:
        raise NotImplementedError

    def generate(
        self,
        question: str,
        evidence: Sequence[Evidence],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> GenerationOutcome:
        if not evidence:
            return GenerationOutcome(
                GroundedAnswer(claims=[], insufficient_evidence=True), reason="no_evidence"
            )
        return self._settle(
            build_messages(self.prompt, question, evidence),
            {item.id: item for item in evidence},
            request_id=request_id,
            run_id=run_id,
        )

    def rewrite(
        self,
        question: str,
        evidence: Sequence[Evidence],
        previous: GroundedAnswer,
        unsupported: Sequence[tuple[str, str]],
        prompt: RewritePrompt,
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> GenerationOutcome:
        """The answer again, told which of its claims (with the checker's reason) the cited
        evidence does not support; validated and repaired exactly as ``generate``'s."""

        allowed = {item.id: item for item in evidence}
        listed = "\n".join(f"- {text} (why: {reason})" for text, reason in unsupported)
        follow_up = _REWRITE_PLACEHOLDER.sub(
            lambda match: {"unsupported": listed, "allowed": ", ".join(allowed)}[match.group(1)],
            prompt.text,
        )
        messages = [
            *build_messages(self.prompt, question, evidence),
            {"role": "assistant", "content": previous.model_dump_json()},
            {"role": "user", "content": follow_up},
        ]
        return self._settle(messages, allowed, request_id=request_id, run_id=run_id)

    def _settle(
        self,
        messages: list[dict[str, str]],
        allowed: dict[str, Evidence],
        *,
        request_id: UUID | None,
        run_id: str | None,
    ) -> GenerationOutcome:
        outcome = GenerationOutcome(GroundedAnswer(claims=[], insufficient_evidence=True))
        first = self._attempt(messages, request_id=request_id, run_id=run_id)
        self._account(outcome, first)
        answer, problems = self._check(first.text, allowed)
        if answer is not None and not problems:
            outcome.answer = answer
            return outcome
        outcome.problems = problems
        outcome.repair_attempted = True
        repair = _fill(
            self.prompt.repair,
            {"problems": "; ".join(problems), "allowed": ", ".join(allowed)},
        )
        second = self._attempt(
            [
                *messages,
                {"role": "assistant", "content": first.text},
                {"role": "user", "content": repair},
            ],
            request_id=request_id,
            run_id=run_id,
        )
        self._account(outcome, second)
        answer, problems = self._check(second.text, allowed)
        if answer is not None and not problems:
            outcome.answer = answer
            return outcome
        outcome.reason = problems[0]
        return outcome

    def answer(self, question: str, evidence: list[Evidence]) -> GroundedAnswer:
        return self.generate(question, evidence).answer

    @staticmethod
    def _account(outcome: GenerationOutcome, attempt: _Attempt) -> None:
        outcome.calls += 1
        outcome.cost_usd += attempt.cost_usd
        outcome.served_models.append(attempt.served_model)
        outcome.cached = outcome.cached or attempt.cached

    @staticmethod
    def _check(text: str, allowed: dict[str, Evidence]) -> tuple[GroundedAnswer | None, list[str]]:
        try:
            claims, insufficient = parse_answer(text)
        except GenerationError as error:
            return None, [error.code]
        problems = [str(problem) for problem in check_claims(claims, allowed)]
        if problems:
            return None, problems
        return GroundedAnswer(claims=claims, insufficient_evidence=insufficient), []


class ChatGenerator(_Grounded):
    """Grounded answers from a configured hosted model, every call under the spend ledger."""

    def __init__(
        self,
        client: ChatClient,
        prompt: GenerationPrompt,
        *,
        timeout: float = 60.0,
        purpose: str = "evaluation",
    ) -> None:
        super().__init__(prompt)
        self._client = client
        self.model = client.model
        self.timeout = timeout
        self.purpose = purpose
        self.identity = f"{self.model.key}={self.model.model}#{prompt.name}:{prompt.sha256[:12]}"

    def close(self) -> None:
        self._client.close()

    def _attempt(
        self, messages: list[dict[str, str]], *, request_id: UUID | None, run_id: str | None
    ) -> _Attempt:
        result = self._client.complete(
            messages,
            timeout=self.timeout,
            purpose=self.purpose,
            request_id=request_id,
            run_id=run_id,
        )
        return _Attempt(result.text, result.cost_usd, result.served_model, result.cached)


class StructuredGenerator(_Grounded):
    """The same validation loop, the model constrained to the answer's JSON schema.

    Constrained decoding removes malformed JSON and code fences; it does not check that a
    cited id exists or supports its claim, so the answer is still parsed and validated by
    ``_check`` exactly as ``ChatGenerator``'s is, with the same single repair.
    """

    def __init__(
        self, model: LedgerChatModel, prompt: GenerationPrompt, *, purpose: str = "evaluation"
    ) -> None:
        super().__init__(prompt)
        self._model = model
        self.purpose = purpose
        self.identity = (
            f"{model.model.key}={model.model.model}#{prompt.name}:{prompt.sha256[:12]}+json_schema"
        )

    def close(self) -> None:
        """Nothing to release: ChatOpenAI owns and reuses its own HTTP clients."""

    def _attempt(
        self, messages: list[dict[str, str]], *, request_id: UUID | None, run_id: str | None
    ) -> _Attempt:
        result = self._model.complete(
            messages,
            purpose=self.purpose,
            schema=GROUNDED_ANSWER_SCHEMA,
            request_id=request_id,
            run_id=run_id,
        )
        return _Attempt(result.text, result.cost_usd, result.served_model, result.cached)


class FakeGenerator(_Grounded):
    """Deterministic: one claim per piece of evidence, citing it; invents an id on request."""

    FABRICATE = "FABRICATE"
    FABRICATE_TWICE = "FABRICATE TWICE"
    identity = "fixture/grounded-claims"

    def __init__(self, prompt: GenerationPrompt | None = None) -> None:
        super().__init__(prompt or load_generation_prompt(PROMPT_PATH))
        self._calls = 0

    def _attempt(
        self, messages: list[dict[str, str]], *, request_id: UUID | None, run_id: str | None
    ) -> _Attempt:
        self._calls += 1
        user = messages[1]["content"]
        question = user.split("\n", 1)[0]
        ids = re.findall(r"^\[(E\d+)\]", user, flags=re.MULTILINE)
        repairing = messages[-1]["role"] == "user" and len(messages) > 2
        fabricate = self.FABRICATE_TWICE in question or (
            self.FABRICATE in question and not repairing
        )
        if fabricate:
            claims: list[dict[str, Any]] = [
                {"text": "Something the evidence never said.", "evidence_ids": ["E999"]}
            ]
        else:
            claims = [{"text": f"Claim from {label}.", "evidence_ids": [label]} for label in ids]
        body = json.dumps({"claims": claims, "insufficient_evidence": not claims})
        return _Attempt(body, 0.0, "fixture", False)
