"""configs/chat.yaml, validated; its digest is part of every turn's identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..search.service import SearchConfig, load_search_config

DEFAULT_PATH = Path("configs/chat.yaml")
SEARCH_PATH = Path("configs/search.yaml")


@dataclass(frozen=True)
class ChatConfig:
    sha256: str
    model: str
    generator: str
    generate_prompt: Path
    condense_prompt: Path
    papers: int
    search_overrides: tuple[tuple[str, Any], ...]
    evidence: Path
    history_turns: int
    history_max_chars: int
    condense: bool
    condense_timeout: float
    generate_timeout: float
    turn_timeout: float


def load_chat_config(path: str | Path = DEFAULT_PATH) -> ChatConfig:
    from ..evaluation.retrieval import ExperimentError, _search_overrides

    raw_bytes = Path(path).read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    try:
        overrides = _search_overrides(
            "search.overrides", "hybrid_rerank", (raw["search"] or {}).get("overrides")
        )
        timeouts = raw["timeouts_seconds"]
        config = ChatConfig(
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            model=str(raw["model"]),
            generator=str(raw["generator"]),
            generate_prompt=Path(raw["prompts"]["generate"]),
            condense_prompt=Path(raw["prompts"]["condense"]),
            papers=int(raw["search"]["papers"]),
            search_overrides=overrides,
            evidence=Path(raw["evidence"]),
            history_turns=int(raw["history"]["turns"]),
            history_max_chars=int(raw["history"]["max_chars"]),
            condense=bool(raw["condense"]),
            condense_timeout=float(timeouts["condense"]),
            generate_timeout=float(timeouts["generate"]),
            turn_timeout=float(timeouts["turn"]),
        )
    except (KeyError, TypeError, ValueError, ExperimentError) as error:
        raise ValueError(f"chat_config_invalid:{error}") from error
    if config.generator not in ("structured", "http"):
        raise ValueError("chat_config_invalid:generator")
    if not 1 <= config.papers <= 50 or config.history_turns < 0:
        raise ValueError("chat_config_invalid:bounds")
    return config


def chat_search_config(config: ChatConfig, path: str | Path = SEARCH_PATH) -> SearchConfig:
    """The shipped search file with the chat's overrides, checked as the harness checks them."""

    search = replace(load_search_config(path), **dict(config.search_overrides))
    if search.rerank_seconds > search.total_seconds:
        raise ValueError("chat_config_invalid:rerank deadline exceeds the total")
    return search
