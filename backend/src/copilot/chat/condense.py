"""A follow-up becomes one standalone search query, read from the conversation.

The condenser is the only place history reaches a model. It is bounded twice: by turns
(configs/chat.yaml) before it gets here, and by characters here, newest kept, so a long chat
cannot grow the reserved cost without bound. Anything but the spend cap that goes wrong —
a timeout, an empty or structured reply, an essay — falls back to the user's own message
with the warning ``condense_failed``: the turn still answers, perhaps less well.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import yaml

from ..models.llm import ChatResult, LlmError
from .turns import HistoryItem

MAX_QUERY_CHARS = 2000
_PLACEHOLDER = re.compile(r"\{(history|message)\}")
_LABEL = {"user": "User", "assistant": "Assistant"}


class Completer(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        purpose: str,
        schema: Any = None,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> ChatResult: ...


@dataclass(frozen=True)
class CondensePrompt:
    name: str
    sha256: str
    system: str
    user: str


def load_condense_prompt(path: str | Path) -> CondensePrompt:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    prompt = CondensePrompt(
        name=str(raw["name"]),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        system=str(raw["system"]),
        user=str(raw["user"]),
    )
    for placeholder in ("{history}", "{message}"):
        if placeholder not in prompt.user:
            raise ValueError(f"prompt_invalid:user.{placeholder}")
    return prompt


def format_history(history: Sequence[HistoryItem], *, max_chars: int) -> str:
    """``User: …`` / ``Assistant: …`` lines, oldest first, cut from the front to the budget."""

    lines: list[str] = []
    used = 0
    for item in reversed(history):
        line = f"{_LABEL[item.role]}: {' '.join(item.text.split())}"
        cost = len(line) + (1 if lines else 0)
        if used + cost > max_chars:
            room = max_chars - used - (1 if lines else 0)
            if room > 0:
                lines.append(line[-room:])
            break
        lines.append(line)
        used += cost
    return "\n".join(reversed(lines))


def clean_query(text: str) -> str | None:
    lines = text.strip().splitlines()
    if not lines:
        return None
    line = " ".join(lines[0].split()).strip("\"'“”` ")
    if not line or line.startswith(("{", "[")) or len(line) > MAX_QUERY_CHARS:
        return None
    return line


class Condenser:
    def __init__(
        self,
        model: Completer | None,
        prompt: CondensePrompt,
        *,
        max_chars: int,
        purpose: str = "chat",
    ) -> None:
        self._model = model
        self.purpose = purpose
        self.prompt = prompt
        self._max_chars = max_chars

    def standalone(
        self,
        history: Sequence[HistoryItem],
        message: str,
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> tuple[str, list[str], float]:
        """The query, warnings, and what the call cost."""

        if not history or self._model is None:
            return message, [], 0.0
        values = {"history": format_history(history, max_chars=self._max_chars), "message": message}
        user = _PLACEHOLDER.sub(lambda match: values[match.group(1)], self.prompt.user)
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": user},
        ]
        try:
            result = self._model.complete(
                messages, purpose=self.purpose, request_id=request_id, run_id=run_id
            )
        except LlmError as error:
            if error.code == "llm_spend_cap":
                raise
            return message, ["condense_failed"], 0.0
        query = clean_query(result.text)
        if query is None:
            return message, ["condense_failed"], result.cost_usd
        return query, [], result.cost_usd
