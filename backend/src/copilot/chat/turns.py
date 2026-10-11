"""Plain data a chat turn is made of; no I/O."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..contracts import Evidence, GroundedAnswer

TITLE_CHARS = 80
# What history shows the condenser for an answered turn that had no answer.
INSUFFICIENT_TEXT = "The corpus did not contain enough evidence to answer this."


@dataclass(frozen=True)
class HistoryItem:
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class TurnResult:
    answer: GroundedAnswer
    evidence: Sequence[Evidence]
    standalone_query: str
    corpus_release_id: str | None
    warnings: Sequence[str]
    identity: Mapping[str, Any]
    trace: Mapping[str, float]
    cost_usd: float


def chat_title(message: str) -> str:
    """The first message, on one line, cut at a word boundary."""

    line = " ".join(message.split())
    if len(line) <= TITLE_CHARS:
        return line
    cut = line[: TITLE_CHARS - 1].rsplit(" ", 1)[0]
    return f"{cut}…"


def answer_text(answer: GroundedAnswer) -> str:
    return " ".join(claim.text for claim in answer.claims)
