"""The research agent's optional self-check (agent self-check, 2026-10-29).

After the answer is written, one structured call under the spend ledger checks every claim
against the full text of the passages it cites. When any claim is unsupported, the generator
rewrites the answer once, told which claims and why; the rewrite is checked again and any claim
still unsupported is dropped. A checker failure keeps the answer it was given, with a warning.

The citation check in ``evidence.citations`` is structural (the ids exist, quotes are in their
source); this one asks whether the cited text says what the claim says.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from ..contracts import Claim, Evidence, GroundedAnswer
from ..models.generation import (
    GenerationError,
    GenerationOutcome,
    RewritePrompt,
    evidence_block,
    loads_lenient,
)
from ..models.llm import LlmError

SUPPORT_SCHEMA: dict[str, Any] = {
    "title": "claim_support",
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["index", "reason", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class CheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckPrompt:
    name: str
    sha256: str
    system: str
    user: str


def load_check_prompt(path: str | Path) -> CheckPrompt:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    prompt = CheckPrompt(
        name=str(data["name"]),
        sha256=hashlib.sha256(raw).hexdigest(),
        system=str(data["system"]),
        user=str(data["user"]),
    )
    for placeholder in ("{question}", "{claims}"):
        if placeholder not in prompt.user:
            raise ValueError(f"prompt_invalid:user.{placeholder}")
    return prompt


@dataclass(frozen=True)
class ClaimVerdict:
    index: int
    supported: bool
    reason: str


def parse_support(text: str, *, claims: int) -> list[ClaimVerdict]:
    """One verdict per claim, in order. A missing verdict counts as unsupported, a repeated
    index keeps its first verdict, an index out of range is ignored."""

    try:
        items = loads_lenient(text.strip())["claims"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise CheckError("check_unparseable") from error
    if not isinstance(items, list):
        raise CheckError("check_unparseable")
    found: dict[int, ClaimVerdict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < claims or index in found:
            continue
        found[index] = ClaimVerdict(
            index, item.get("supported") is True, " ".join(str(item.get("reason", "")).split())
        )
    return [found.get(i, ClaimVerdict(i, False, "no verdict")) for i in range(claims)]


def claims_block(answer: GroundedAnswer, evidence: Sequence[Evidence]) -> str:
    """Each claim under its index, followed by the full text of the passages it cites, as the
    generator saw them."""

    by_id = {item.id: item for item in evidence}
    parts = []
    for index, claim in enumerate(answer.claims):
        cited = [
            evidence_block(by_id[label]) if label in by_id else f"[{label}] (not in the evidence)"
            for label in claim.evidence_ids
        ]
        passages = "\n\n".join(cited) if cited else "(no passage cited)"
        parts.append(f"Claim {index}: {claim.text}\nCited passages:\n{passages}")
    return "\n\n---\n\n".join(parts)


@dataclass
class LedgerChecker:
    """The verdicts from a chat model through ``LedgerChatModel.complete``."""

    chat: Any
    prompt: CheckPrompt
    purpose: str = "evaluation"

    @property
    def identity(self) -> str:
        key = getattr(getattr(self.chat, "model", None), "key", "unknown")
        return f"{key}#{self.prompt.name}:{self.prompt.sha256[:12]}"

    def check(
        self,
        question: str,
        answer: GroundedAnswer,
        evidence: Sequence[Evidence],
        *,
        request_id: UUID | None,
        run_id: str | None,
    ) -> tuple[list[ClaimVerdict], float]:
        user = self.prompt.user.replace("{claims}", claims_block(answer, evidence))
        result = self.chat.complete(
            [
                {"role": "system", "content": self.prompt.system},
                {"role": "user", "content": user.replace("{question}", question)},
            ],
            purpose=self.purpose,
            schema=SUPPORT_SCHEMA,
            request_id=request_id,
            run_id=run_id,
        )
        return parse_support(result.text, claims=len(answer.claims)), result.cost_usd


@dataclass
class SelfCheck:
    # passed | rewritten | dropped | emptied | failed | skipped
    result: str
    first_claims: int = 0
    first_unsupported: int = 0
    final_unsupported_dropped: int = 0
    calls: int = 0
    cost_usd: float = 0.0


def _without(answer: GroundedAnswer, verdicts: Sequence[ClaimVerdict]) -> GroundedAnswer:
    kept: list[Claim] = [c for c, v in zip(answer.claims, verdicts, strict=True) if v.supported]
    return GroundedAnswer(
        claims=kept, insufficient_evidence=answer.insufficient_evidence or not kept
    )


def self_check(
    question: str,
    evidence: Sequence[Evidence],
    outcome: GenerationOutcome,
    *,
    checker: LedgerChecker,
    generator: Any,
    rewrite: RewritePrompt,
    request_id: UUID | None = None,
    run_id: str | None = None,
) -> tuple[GenerationOutcome, SelfCheck]:
    """The answer checked claim by claim, rewritten once if needed. The returned outcome is the
    one given with its answer replaced; the check's own calls and cost are in ``SelfCheck``."""

    answer = outcome.answer
    record = SelfCheck("skipped", first_claims=len(answer.claims))
    if not answer.claims:
        return outcome, record

    def replaced(new: GroundedAnswer) -> GenerationOutcome:
        return GenerationOutcome(
            new,
            calls=outcome.calls,
            repair_attempted=outcome.repair_attempted,
            reason=outcome.reason,
            problems=list(outcome.problems),
            cost_usd=outcome.cost_usd,
            served_models=list(outcome.served_models),
            cached=outcome.cached,
        )

    def verdicts_for(given: GroundedAnswer) -> list[ClaimVerdict]:
        found, cost = checker.check(question, given, evidence, request_id=request_id, run_id=run_id)
        record.calls += 1
        record.cost_usd += cost
        return found

    try:
        first = verdicts_for(answer)
    except (LlmError, CheckError):
        record.result = "failed"
        return outcome, record
    unsupported = [v for v in first if not v.supported]
    record.first_unsupported = len(unsupported)
    if not unsupported:
        record.result = "passed"
        return outcome, record
    try:
        rewritten = generator.rewrite(
            question,
            evidence,
            answer,
            [(answer.claims[v.index].text, v.reason) for v in unsupported],
            rewrite,
            request_id=request_id,
            run_id=run_id,
        )
    except (LlmError, GenerationError):
        rewritten = None
    if rewritten is not None:
        record.calls += rewritten.calls
        record.cost_usd += rewritten.cost_usd
    if rewritten is None or rewritten.reason is not None:
        record.result = "emptied" if len(unsupported) == len(first) else "dropped"
        record.final_unsupported_dropped = len(unsupported)
        return replaced(_without(answer, first)), record
    second_answer = rewritten.answer
    if not second_answer.claims:
        record.result = "emptied"
        return replaced(GroundedAnswer(claims=[], insufficient_evidence=True)), record
    try:
        second = verdicts_for(second_answer)
    except (LlmError, CheckError):
        record.result = "failed"
        return replaced(second_answer), record
    dropped = [v for v in second if not v.supported]
    record.final_unsupported_dropped = len(dropped)
    final = _without(second_answer, second)
    record.result = "rewritten" if final.claims else "emptied"
    return replaced(final), record
