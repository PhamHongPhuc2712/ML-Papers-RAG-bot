"""The research agent's optional planning step (agent follow-ups, 2026-10-28).

One structured-output call under the spend ledger splits a question into at most
``max_subquestions`` standalone sub-questions before the tool loop; the turn shows them to the
agent as parts to cover. A failed or malformed plan is dropped with a warning: the turn goes on
without one, as haiku.rag's agent always does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import yaml

PLAN_SCHEMA = {
    "title": "plan",
    "type": "object",
    "properties": {"subquestions": {"type": "array", "items": {"type": "string"}}},
    "required": ["subquestions"],
    "additionalProperties": False,
}


class PlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanPrompt:
    name: str
    sha256: str
    system: str
    user: str


def load_plan_prompt(path: str | Path) -> PlanPrompt:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    return PlanPrompt(
        name=str(data["name"]),
        sha256=hashlib.sha256(raw).hexdigest(),
        system=str(data["system"]),
        user=str(data["user"]),
    )


@dataclass(frozen=True)
class Plan:
    subquestions: tuple[str, ...]
    cost_usd: float


class Planner(Protocol):
    @property
    def identity(self) -> str: ...

    def plan(
        self,
        question: str,
        *,
        max_subquestions: int,
        request_id: UUID | None,
        run_id: str | None,
    ) -> Plan: ...


def parse_plan(text: str, *, max_subquestions: int) -> tuple[str, ...]:
    """The non-empty sub-questions, in order, cut to the limit; PlanError when malformed."""

    try:
        data = json.loads(text)
        items = data["subquestions"]
    except (ValueError, KeyError, TypeError) as error:
        raise PlanError("plan_malformed") from error
    if not isinstance(items, list):
        raise PlanError("plan_malformed")
    parts = tuple(" ".join(str(item).split()) for item in items)
    return tuple(part for part in parts if part)[:max_subquestions]


@dataclass
class LedgerPlanner:
    """The plan from the agent's own model through ``LedgerChatModel.complete``."""

    chat: Any
    prompt: PlanPrompt
    purpose: str = "evaluation"

    @property
    def identity(self) -> str:
        return f"{self.prompt.name}:{self.prompt.sha256[:12]}"

    def plan(
        self,
        question: str,
        *,
        max_subquestions: int,
        request_id: UUID | None,
        run_id: str | None,
    ) -> Plan:
        system = self.prompt.system.replace("{max_subquestions}", str(max_subquestions))
        result = self.chat.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": self.prompt.user.replace("{question}", question)},
            ],
            purpose=self.purpose,
            schema=PLAN_SCHEMA,
            request_id=request_id,
            run_id=run_id,
        )
        return Plan(parse_plan(result.text, max_subquestions=max_subquestions), result.cost_usd)
