"""Listwise reranking by a hosted LLM, as LitSearch's best published system does.

One prompt shows the whole reranked head — each candidate's title and abstract,
cut to a word budget, under a bracketed identifier — and asks for the order
"[3] > [1] > [2]". The answer is parsed into a strict permutation: identifiers
out of range or repeated are dropped, candidates the model left out follow in
their incoming order, and an answer that names no candidate is an error. The
model can therefore reorder the head but never add to it or drop from it.

The prompt file's sha256 is part of the reranker's identity, so an edited
prompt is a different ranker in every manifest. Failures propagate: the search
service decides the fallback.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import yaml

from ..contracts import ListwiseResult
from ..models.llm import ChatClient
from .rerank import RerankError

_PLACEHOLDER = re.compile(r"\{(query|candidates|count)\}")
_BRACKETED = re.compile(r"\[(\d+)\]")
# Without brackets, only a chain such as "2 > 1" or "2, 1" is a ranking; a lone
# number in prose ("these 4 papers") is not.
_BARE_CHAIN = re.compile(r"\d+(?:\s*[>,]\s*\d+)+")
_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class ListwisePrompt:
    name: str
    sha256: str
    system: str
    user: str


def load_prompt(path: str | Path) -> ListwisePrompt:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    try:
        name, system, user = str(raw["name"]), str(raw["system"]), str(raw["user"])
    except KeyError as error:
        raise ValueError(f"prompt_invalid:{error.args[0]}") from error
    for placeholder in ("{query}", "{candidates}", "{count}"):
        if placeholder not in user:
            raise ValueError(f"prompt_invalid:user.{placeholder}")
    return ListwisePrompt(
        name=name, sha256=hashlib.sha256(raw_bytes).hexdigest(), system=system, user=user
    )


def build_messages(
    prompt: ListwisePrompt, query: str, texts: Sequence[str], words: int
) -> list[dict[str, str]]:
    """The system and user messages. The template is filled in one pass, so text
    inserted for one placeholder is never read as another."""

    block = "\n\n".join(
        f"[{number}] {' '.join(text.split()[:words])}" for number, text in enumerate(texts, 1)
    )
    values = {"query": query, "candidates": block, "count": str(len(texts))}
    user = _PLACEHOLDER.sub(lambda match: values[match.group(1)], prompt.user)
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]


def parse_permutation(answer: str, count: int) -> tuple[list[int], int]:
    """Indices into the candidates, best first, and how many the model itself placed.

    Unranked candidates follow in their incoming order, as LitSearch's reranker does, so the
    result is always a permutation of exactly the candidates given.
    """

    numbers = _BRACKETED.findall(answer)
    if not numbers:
        numbers = [
            number
            for chain in _BARE_CHAIN.findall(answer)
            for number in _NUMBER.findall(chain)
        ]
    placed = list(dict.fromkeys(i for i in (int(n) - 1 for n in numbers) if 0 <= i < count))
    if not placed:
        raise RerankError("llm_rerank_unparseable")
    chosen = set(placed)
    return placed + [i for i in range(count) if i not in chosen], len(placed)


class LlmListwiseReranker:
    """One listwise call per head, through a configured chat client."""

    def __init__(self, client: ChatClient, prompt: ListwisePrompt, *, words: int = 300) -> None:
        if words < 1:
            raise ValueError("listwise_words_invalid")
        self._client = client
        self.prompt = prompt
        self.words = words
        self.model = client.model
        self.identity = (
            f"{self.model.key}={self.model.model}#{prompt.name}:{prompt.sha256[:12]}#w{words}"
        )

    def close(self) -> None:
        self._client.close()

    def order(
        self,
        query: str,
        texts: Sequence[str],
        *,
        timeout: float,
        purpose: str,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> ListwiseResult:
        messages = build_messages(self.prompt, query, texts, self.words)
        result = self._client.complete(
            messages, timeout=timeout, purpose=purpose, request_id=request_id, run_id=run_id
        )
        order, ranked = parse_permutation(result.text, len(texts))
        return ListwiseResult(
            order=order,
            ranked_by_model=ranked,
            cost_usd=result.cost_usd,
            served_model=result.served_model,
            cached=result.cached,
        )
