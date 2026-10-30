"""configs/chat.yaml, validated; its digest is part of every turn's identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..search.service import SearchConfig, load_search_config

DEFAULT_PATH = Path("configs/chat.yaml")
SEARCH_PATH = Path("configs/search.yaml")
# A search request returns at most 50 results (spec §10).
MAX_REFERENCES = 50


@dataclass(frozen=True)
class ChatConfig:
    sha256: str
    model: str
    generator: str
    generate_prompt: Path
    condense_prompt: Path
    papers: int
    references: int
    evidence: Path
    history_turns: int
    history_max_chars: int
    condense: bool
    condense_timeout: float
    generate_timeout: float
    turn_timeout: float
    daily_answers: int
    # The research agent that serves every turn (2026-10-30).
    agent_prompt: Path = Path("prompts/agent/research-v2.yaml")
    max_searches: int = 3
    max_model_calls: int = 8
    max_sections: int = 4
    answer_from: str = "seen"
    verbatim_first: bool = True
    carried: int = 12


def load_chat_config(path: str | Path = DEFAULT_PATH) -> ChatConfig:
    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    try:
        search = raw["search"] or {}
        if "overrides" in search:
            # One first stage serves every turn (RAG-only plan, U1): configs/search.yaml.
            raise ValueError("search.overrides: the chat reads configs/search.yaml unchanged")
        timeouts = raw["timeouts_seconds"]
        agent = raw["agent"]
        config = ChatConfig(
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            model=str(raw["model"]),
            generator=str(raw["generator"]),
            generate_prompt=Path(raw["prompts"]["generate"]),
            condense_prompt=Path(raw["prompts"]["condense"]),
            papers=int(search["papers"]),
            references=int(search["references"]),
            evidence=Path(raw["evidence"]),
            history_turns=int(raw["history"]["turns"]),
            history_max_chars=int(raw["history"]["max_chars"]),
            condense=bool(raw["condense"]),
            condense_timeout=float(timeouts["condense"]),
            generate_timeout=float(timeouts["generate"]),
            turn_timeout=float(timeouts["turn"]),
            daily_answers=int(raw["limits"]["daily_answers"]),
            agent_prompt=Path(agent["prompt"]),
            max_searches=int(agent["max_searches"]),
            max_model_calls=int(agent["max_model_calls"]),
            max_sections=int(agent["max_sections"]),
            answer_from=str(agent["answer_from"]),
            verbatim_first=bool(agent["verbatim_first"]),
            carried=int(agent["carried"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"chat_config_invalid:{error}") from error
    if config.generator not in ("structured", "http"):
        raise ValueError("chat_config_invalid:generator")
    if config.answer_from not in ("chosen", "seen"):
        raise ValueError("chat_config_invalid:agent.answer_from")
    if min(config.max_searches, config.max_model_calls) < 1 or config.carried < 0:
        raise ValueError("chat_config_invalid:agent bounds")
    if (
        not 1 <= config.papers <= config.references <= MAX_REFERENCES
        or config.history_turns < 0
        or config.daily_answers < 1
    ):
        raise ValueError("chat_config_invalid:bounds")
    return config


def chat_search_config(config: ChatConfig, path: str | Path = SEARCH_PATH) -> SearchConfig:
    """The shipped search file, the one first stage every turn and the search API share."""

    search = load_search_config(path)
    if search.rerank_seconds > search.total_seconds:
        raise ValueError("chat_config_invalid:rerank deadline exceeds the total")
    return search
