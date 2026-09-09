"""Section-aware chunking with a versioned policy and stable chunk identity."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

Tokenize = Callable[[str], Sequence[str]]
Detokenize = Callable[[Sequence[str]], str]

# Namespace for UUIDv5 chunk identifiers; configs/parsing.yaml may override it.
DEFAULT_CHUNK_NAMESPACE = uuid.UUID("9d2f5a6c-7b3e-4c1a-8e5d-2f6b9c1d3e47")
EVIDENCE_EXCLUDED_KINDS = frozenset({"references"})


@dataclass(frozen=True)
class Section:
    """One heading-delimited span of a document with the pages it touches."""

    name: str
    text: str
    page_start: int | None
    page_end: int | None
    ordinal: int
    kind: str = "body"


@dataclass(frozen=True)
class ParserConfig:
    adapter: str
    parser_version: str
    max_pdf_bytes: int
    reject_encrypted: bool


@dataclass(frozen=True)
class ChunkerConfig:
    policy: str
    chunker_version: str
    target_tokens: int
    hard_cap_tokens: int
    overlap_tokens: int
    references_in_default_evidence: bool
    uuid_namespace: uuid.UUID


@dataclass(frozen=True)
class ParsingConfig:
    schema_version: int
    parser: ParserConfig
    chunker: ChunkerConfig


def _require(mapping: Mapping[str, Any], key: str, kind: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is int:
        raise ValueError(f"parsing_config_invalid:{key}")
    return value


def load_parsing_config(path: str | Path) -> ParsingConfig:
    """Load and validate the versioned parsing/chunking policy."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("parsing_config_invalid:root")
    parser = raw.get("parser")
    chunker = raw.get("chunker")
    if not isinstance(parser, Mapping) or not isinstance(chunker, Mapping):
        raise ValueError("parsing_config_invalid:sections")
    namespace_value = chunker.get("uuid_namespace", str(DEFAULT_CHUNK_NAMESPACE))
    try:
        namespace = uuid.UUID(str(namespace_value))
    except ValueError as exc:
        raise ValueError("parsing_config_invalid:uuid_namespace") from exc
    chunker_config = ChunkerConfig(
        policy=_require(chunker, "policy", str),
        chunker_version=_require(chunker, "chunker_version", str),
        target_tokens=_require(chunker, "target_tokens", int),
        hard_cap_tokens=_require(chunker, "hard_cap_tokens", int),
        overlap_tokens=_require(chunker, "overlap_tokens", int),
        references_in_default_evidence=bool(chunker.get("references_in_default_evidence", False)),
        uuid_namespace=namespace,
    )
    _validate_window(
        chunker_config.target_tokens, chunker_config.overlap_tokens, chunker_config.hard_cap_tokens
    )
    return ParsingConfig(
        schema_version=_require(raw, "schema_version", int),
        parser=ParserConfig(
            adapter=_require(parser, "adapter", str),
            parser_version=_require(parser, "parser_version", str),
            max_pdf_bytes=_require(parser, "max_pdf_bytes", int),
            reject_encrypted=bool(parser.get("reject_encrypted", True)),
        ),
        chunker=chunker_config,
    )


def chunk_id(
    paper_id: uuid.UUID,
    content_sha256: str,
    parser_version: str,
    chunker_version: str,
    section_ordinal: int,
    chunk_ordinal: int,
    *,
    namespace: uuid.UUID = DEFAULT_CHUNK_NAMESPACE,
) -> uuid.UUID:
    """UUIDv5 over the inputs that make a chunk reproducible (spec §4)."""

    name = "|".join(
        (
            str(paper_id),
            content_sha256,
            parser_version,
            chunker_version,
            str(section_ordinal),
            str(chunk_ordinal),
        )
    )
    return uuid.uuid5(namespace, name)


def whitespace_tokenize(text: str) -> list[str]:
    """Deterministic fixture tokenizer; production injects the pinned model tokenizer."""

    return text.split()


def whitespace_detokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def _validate_window(target: int, overlap: int, hard_cap: int | None) -> None:
    if target <= 0 or overlap < 0 or overlap >= target:
        raise ValueError("invalid_chunk_window")
    if hard_cap is not None and hard_cap < target:
        raise ValueError("invalid_chunk_window")


def _chunk(
    section: Section,
    text: str,
    token_count: int,
    ordinal: int,
    *,
    fragment: str | None,
) -> dict[str, Any]:
    return {
        "section": section.name,
        "text": text,
        "token_count": token_count,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "ordinal": ordinal,
        "kind": section.kind,
        "section_ordinal": section.ordinal,
        "evidence_default": section.kind not in EVIDENCE_EXCLUDED_KINDS,
        "fragment": fragment,
    }


def chunk_sections(
    sections: Sequence[Section],
    tokenize: Tokenize,
    detokenize: Detokenize,
    target: int,
    overlap: int,
    hard_cap: int | None = None,
) -> list[dict[str, Any]]:
    """Split each section into token windows that never cross a section boundary.

    Prose uses fixed windows of ``target`` tokens with ``overlap`` tokens repeated
    from the previous window. Tables stay coherent up to ``hard_cap`` tokens and
    otherwise become labeled fragments that repeat the table heading instead of
    overlapping rows. References are chunked but flagged out of default evidence.
    """

    _validate_window(target, overlap, hard_cap)
    cap = hard_cap if hard_cap is not None else target
    chunks: list[dict[str, Any]] = []
    for section in sections:
        tokens = list(tokenize(section.text))
        if not tokens:
            continue
        if section.kind == "table":
            _chunk_table(section, tokens, detokenize, target, cap, chunks)
            continue
        step = target - overlap
        for start in range(0, len(tokens), step):
            part = tokens[start : start + target]
            chunks.append(_chunk(section, detokenize(part), len(part), len(chunks), fragment=None))
            if start + target >= len(tokens):
                break
    return chunks


def _chunk_table(
    section: Section,
    tokens: list[str],
    detokenize: Detokenize,
    target: int,
    cap: int,
    chunks: list[dict[str, Any]],
) -> None:
    if len(tokens) <= cap:
        chunks.append(_chunk(section, detokenize(tokens), len(tokens), len(chunks), fragment=None))
        return
    parts = [tokens[start : start + target] for start in range(0, len(tokens), target)]
    total = len(parts)
    for index, part in enumerate(parts, start=1):
        label = f"{section.name} (part {index}/{total})"
        chunks.append(
            _chunk(
                section,
                f"{label}\n{detokenize(part)}",
                len(part),
                len(chunks),
                fragment=f"{index}/{total}",
            )
        )
