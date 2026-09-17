"""Section-aware chunking with a versioned policy and stable chunk identity."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer

# One (start, end) character span per token, into the text that produced them.
# Windows are cut on these spans, so chunk text is always a literal slice of the
# section rather than a detokenized reconstruction of it.
TokenSpans = Callable[[str], Sequence[tuple[int, int]]]
_WORD = re.compile(r"\S+")

POLICIES = ("fixed-window", "paragraph-pack")

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


class TokenizerUnavailableError(RuntimeError):
    """The pinned tokenizer is absent or altered.

    Never degrade to whitespace counting here: that is precisely the defect
    this boundary exists to prevent, and it would only reappear at corpus
    scale where re-chunking costs a re-download.
    """

    def __init__(self, code: str, path: Path, detail: str = "") -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code}:{path}" + (f":{detail}" if detail else ""))


@dataclass(frozen=True)
class TokenizerSpec:
    """Which tokenizer defines the window, pinned by revision and checksum."""

    kind: str
    repo: str = ""
    revision: str = ""
    file: str = "tokenizer.json"
    sha256: str = ""


def whitespace_spans(text: str) -> list[tuple[int, int]]:
    """Deterministic fixture tokenizer: one token per whitespace-delimited word."""

    return [match.span() for match in _WORD.finditer(text)]


def tokenizer_cache_path(spec: TokenizerSpec, data_dir: str | Path) -> Path:
    """Pinned tokenizers live beside the model cache under the one data root."""

    return Path(data_dir) / "models" / "tokenizers" / spec.repo / spec.revision / spec.file


def load_token_spans(path: str | Path, *, sha256: str = "") -> TokenSpans:
    """Load a tokenizers.json artifact and expose it as character spans."""

    target = Path(path)
    if not target.is_file():
        raise TokenizerUnavailableError("tokenizer_missing", target)
    if sha256:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != sha256:
            raise TokenizerUnavailableError("tokenizer_checksum_mismatch", target, digest)
    tokenizer = Tokenizer.from_file(str(target))
    # A tokenizer that truncates would silently drop the tail of every long
    # section instead of chunking it, so both limits are cleared explicitly.
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def spans(text: str) -> list[tuple[int, int]]:
        return list(tokenizer.encode(text, add_special_tokens=False).offsets)

    return spans


def token_spans_for(spec: TokenizerSpec, data_dir: str | Path) -> TokenSpans:
    """Resolve a configured tokenizer, failing loudly when it is unavailable."""

    if spec.kind == "whitespace":
        return whitespace_spans
    return load_token_spans(tokenizer_cache_path(spec, data_dir), sha256=spec.sha256)


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
    tokenizer: TokenizerSpec
    # paragraph-pack policy only; ignored by fixed-window.
    max_tokens: int = 900
    paragraph_max_tokens: int = 1200
    overlap_sentences: int = 2


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


def _tokenizer_spec(value: object) -> TokenizerSpec:
    """Validate the configured tokenizer; a model one must be fully pinned."""

    # Absent is an error, not whitespace: a config that forgets the tokenizer
    # would otherwise silently reinstate word-counted windows.
    if not isinstance(value, Mapping):
        raise ValueError("parsing_config_invalid:tokenizer")
    kind = str(value.get("kind", "model"))
    if kind not in {"whitespace", "model"}:
        raise ValueError("parsing_config_invalid:tokenizer_kind")
    if kind == "whitespace":
        return TokenizerSpec(kind=kind)
    spec = TokenizerSpec(
        kind=kind,
        repo=str(value.get("repo", "")),
        revision=str(value.get("revision", "")),
        file=str(value.get("file", "tokenizer.json")),
        sha256=str(value.get("sha256", "")),
    )
    if not (spec.repo and spec.revision and spec.sha256):
        raise ValueError("parsing_config_invalid:tokenizer_pin")
    return spec


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
        tokenizer=_tokenizer_spec(chunker.get("tokenizer")),
        policy=_require(chunker, "policy", str),
        chunker_version=_require(chunker, "chunker_version", str),
        target_tokens=_require(chunker, "target_tokens", int),
        hard_cap_tokens=_require(chunker, "hard_cap_tokens", int),
        overlap_tokens=_require(chunker, "overlap_tokens", int),
        references_in_default_evidence=bool(chunker.get("references_in_default_evidence", False)),
        uuid_namespace=namespace,
        max_tokens=int(chunker.get("max_tokens", 900)),
        paragraph_max_tokens=int(chunker.get("paragraph_max_tokens", 1200)),
        overlap_sentences=int(chunker.get("overlap_sentences", 2)),
    )
    if chunker_config.policy not in POLICIES:
        raise ValueError(f"parsing_config_invalid:policy:{chunker_config.policy}")
    if chunker_config.policy == "fixed-window":
        _validate_window(
            chunker_config.target_tokens,
            chunker_config.overlap_tokens,
            chunker_config.hard_cap_tokens,
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


def _slice(section: Section, window: Sequence[tuple[int, int]]) -> str:
    """The literal source text a token window covers, spacing and all."""

    return section.text[window[0][0] : window[-1][1]]


def chunk_sections(
    sections: Sequence[Section],
    spans: TokenSpans,
    target: int,
    overlap: int,
    hard_cap: int | None = None,
) -> list[dict[str, Any]]:
    """Split each section into token windows that never cross a section boundary.

    ``target``, ``overlap`` and ``hard_cap`` are counted in the tokens of the
    supplied tokenizer — the pinned embedding model's in production — because a
    window measured in words is a different window for every language and
    vocabulary. Prose uses fixed windows with ``overlap`` tokens repeated from
    the previous one. Tables stay coherent up to ``hard_cap`` and otherwise
    become labeled fragments that repeat the table heading instead of
    overlapping rows. References are chunked but flagged out of default
    evidence.
    """

    _validate_window(target, overlap, hard_cap)
    cap = hard_cap if hard_cap is not None else target
    chunks: list[dict[str, Any]] = []
    for section in sections:
        tokens = list(spans(section.text))
        if not tokens:
            continue
        if section.kind == "table":
            _chunk_table(section, tokens, target, cap, chunks)
            continue
        step = target - overlap
        for start in range(0, len(tokens), step):
            window = tokens[start : start + target]
            chunks.append(
                _chunk(section, _slice(section, window), len(window), len(chunks), fragment=None)
            )
            if start + target >= len(tokens):
                break
    return chunks


def _chunk_table(
    section: Section,
    tokens: list[tuple[int, int]],
    target: int,
    cap: int,
    chunks: list[dict[str, Any]],
) -> None:
    if len(tokens) <= cap:
        chunks.append(
            _chunk(section, _slice(section, tokens), len(tokens), len(chunks), fragment=None)
        )
        return
    parts = [tokens[start : start + target] for start in range(0, len(tokens), target)]
    total = len(parts)
    for index, part in enumerate(parts, start=1):
        label = f"{section.name} (part {index}/{total})"
        chunks.append(
            _chunk(
                section,
                f"{label}\n{_slice(section, part)}",
                len(part),
                len(chunks),
                fragment=f"{index}/{total}",
            )
        )


def chunk_document(
    sections: Sequence[Section], spans: TokenSpans, chunker: ChunkerConfig
) -> list[dict[str, Any]]:
    """Chunk one document under the configured policy.

    Both policies return the same chunk shape and both keep windows inside a
    section, so the persistence layer and chunk identity are unchanged by the
    choice. The policy name is part of ``chunker_version``, so switching it
    changes every chunk ID — which is the intended provenance signal.
    """

    if chunker.policy == "paragraph-pack":
        from .chunk_paragraph import pack_paragraphs

        return pack_paragraphs(
            sections,
            spans,
            target_tokens=chunker.target_tokens,
            max_tokens=chunker.max_tokens,
            paragraph_max_tokens=chunker.paragraph_max_tokens,
            overlap_sentences=chunker.overlap_sentences,
        )
    return chunk_sections(
        sections,
        spans,
        target=chunker.target_tokens,
        overlap=chunker.overlap_tokens,
        hard_cap=chunker.hard_cap_tokens,
    )
