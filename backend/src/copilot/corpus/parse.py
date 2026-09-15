"""PDF parsing into traceable sections behind a page-adapter boundary.

The first adapter extracts plain text per page with pypdf and is labeled low
quality (spec §4). Docling or another structured converter can replace it by
implementing :class:`PageAdapter` and bumping ``PARSER_VERSION``; chunk
identities change with the version, which is the intended provenance signal.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Protocol

from pypdf import PasswordType, PdfReader

from .chunk import Section

ADAPTER = "pypdf-text"
PARSER_VERSION = "pypdf-text-v2"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
FRONT_MATTER = "Front matter"


class ParseStatus(StrEnum):
    PARSED = "parsed"
    FAILED = "failed"


class ParseErrorCode(StrEnum):
    MISSING = "missing"
    NOT_PDF = "not_pdf"
    OVERSIZED = "oversized"
    ENCRYPTED = "encrypted"
    CORRUPT = "corrupt"
    EMPTY_TEXT = "empty_text"


class PDFParseError(ValueError):
    """Typed parse failure; ``code`` is stored as the version's parse status."""

    def __init__(self, code: ParseErrorCode, path: Path | None = None) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code.value}:{path}" if path is not None else code.value)


@dataclass(frozen=True)
class ParsedPage:
    number: int | None
    text: str


@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    error_code: ParseErrorCode | None
    sections: tuple[Section, ...]
    content_sha256: str
    parser_version: str
    page_count: int
    quality: str


class PageAdapter(Protocol):
    def __call__(self, payload: bytes) -> Sequence[ParsedPage]:
        """Return page texts in reading order; raise PDFParseError for typed failures."""


_SURROGATE = re.compile(r"[\ud800-\udfff]")
_NUMBERED = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][^\n]{0,80})$")
_CAPTION = re.compile(r"^(?P<kind>Table|Figure)\s+\d+[A-Za-z]?\s*[:.]", re.IGNORECASE)
_APPENDIX = re.compile(r"^Appendix(\s+[A-Z](\b|\.|:))?\s*.*$")
_NAMED = {
    "abstract": ("Abstract", "abstract"),
    "references": ("References", "references"),
    "bibliography": ("References", "references"),
    "acknowledgments": ("Acknowledgments", "acknowledgments"),
    "acknowledgements": ("Acknowledgments", "acknowledgments"),
}

# A caption is a bounded block, not a section boundary: without these limits the
# prose that follows a figure is absorbed into the caption's section and labeled
# as figure content (audit 2026-09-14: 28% of body text).
CAPTION_MAX_CHARS = 400
CAPTION_MAX_LINES = 6
# Sections below this survive only by merging into the previous one of the same
# kind, so a stray heading match cannot strand a handful of tokens as a chunk.
MIN_SECTION_CHARS = 60
_MERGEABLE_KINDS = frozenset({"body", "appendix"})
_CAPTION_KINDS = frozenset({"figure", "table"})
_SENTENCE_END = (".", "!", "?")

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_LIGATURE = re.compile("[" + "".join(_LIGATURES) + "]")
# "1 I NTRODUCTION" and "M- PATTERN": small caps render the first letter at full
# size, and pypdf separates it. A lone capital followed by a single capital (an
# article before a small-caps word) does not match, so "A B ETTER W AY" is safe.
_SMALL_CAPS = re.compile(r"(?<![A-Za-z])([A-Z]-?)\s+([A-Z]{2,})")
_HYPHEN_BREAK = re.compile(r"(\w+)-\n(?=[a-z])")
_COMPOUND = re.compile(r"\w+(?:-\w+)+")
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
# A section title carrying a free-standing number is a table row or list item,
# not a heading. Digits bound to a letter ("GPT-4") stay acceptable.
_LOOSE_NUMBER = re.compile(r"(?<![A-Za-z-])\d")


def _normalise_small_caps(line: str) -> str:
    """Rejoin the capital that small-caps rendering separates from its word."""

    return _SMALL_CAPS.sub(lambda match: f"{match.group(1)}{match.group(2)}", line)


def _classify_heading(line: str) -> tuple[str, str] | None:
    if len(line) > 90:
        return None
    line = _normalise_small_caps(line)
    named = _NAMED.get(line.rstrip(".:").lower())
    if named is not None:
        return named
    caption = _CAPTION.match(line)
    if caption is not None:
        return line, caption.group("kind").lower()
    if _APPENDIX.match(line):
        return line, "appendix"
    numbered = _NUMBERED.match(line)
    if numbered is not None:
        title = numbered.group("title").strip()
        if not title.endswith(".") and len(title) >= 3 and not _LOOSE_NUMBER.search(title):
            return title, "body"
    return None


def _storable_text(text: str) -> str:
    """Repair adapter output that UTF-8 and PostgreSQL text columns reject.

    pypdf emits NUL characters from damaged font encodings and maps some astral
    characters onto raw UTF-16 surrogate halves. Both abort the chunk insert
    rather than degrade, so they are repaired at the adapter boundary for every
    adapter. Re-encoding through UTF-16 rejoins genuine pairs into the character
    they stand for, unpaired halves become U+FFFD, and NUL carries no content.
    """

    if "\x00" in text:
        text = text.replace("\x00", "")
    if _SURROGATE.search(text):
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return text


def _normalise_glyphs(text: str) -> str:
    if _LIGATURE.search(text):
        return _LIGATURE.sub(lambda match: _LIGATURES[match.group()], text)
    return text


def _hyphenated_pairs(text: str) -> frozenset[str]:
    """Adjacent halves of every hyphenated word written inline in this document.

    The document is its own dictionary: a line-break hyphen whose halves appear
    joined by a hyphen elsewhere is a real compound, not a wrapped word.
    """

    pairs: set[str] = set()
    for word in _COMPOUND.findall(text):
        parts = word.lower().split("-")
        pairs.update(f"{left}-{right}" for left, right in zip(parts, parts[1:], strict=False))
    return frozenset(pairs)


def _rejoin_hyphenation(text: str, compounds: frozenset[str]) -> str:
    def join(match: re.Match[str]) -> str:
        left = match.group(1)
        right = re.match(r"[a-z]+", text[match.end() :])
        if right is not None and f"{left}-{right.group()}".lower() in compounds:
            return f"{left}-"
        return left

    return _HYPHEN_BREAK.sub(join, text)


def _furniture_key(line: str) -> str:
    key = re.sub(r"\s+", " ", re.sub(r"\d+", "", line)).strip().lower()
    return key if len(key) >= 12 else ""


def _page_furniture(pages: Sequence[ParsedPage]) -> frozenset[str]:
    """Lines repeating at the edge of most pages are headers or footers, not text."""

    if len(pages) < 3:
        return frozenset()
    seen: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        for line in lines[:2] + lines[-2:]:
            key = _furniture_key(line)
            if key:
                seen[key] += 1
    threshold = max(3, len(pages) // 2)
    return frozenset(key for key, count in seen.items() if count >= threshold)


def _normalise_pages(pages: Sequence[ParsedPage]) -> list[ParsedPage]:
    """Repair adapter output before any structure is inferred from it."""

    cleaned = [
        ParsedPage(number=page.number, text=_normalise_glyphs(_storable_text(page.text)))
        for page in pages
    ]
    furniture = _page_furniture(cleaned)
    compounds = _hyphenated_pairs("\n".join(page.text for page in cleaned))
    repaired: list[ParsedPage] = []
    for page in cleaned:
        kept = [
            line
            for line in page.text.splitlines()
            if not (
                _PAGE_NUMBER.match(line.strip()) or (_furniture_key(line.strip()) in furniture)
            )
        ]
        # Rejoin after dropping furniture: a header between the halves of a
        # wrapped word would otherwise keep them apart.
        repaired.append(
            ParsedPage(number=page.number, text=_rejoin_hyphenation("\n".join(kept), compounds))
        )
    return repaired


def _rejoin_target(merged: list[Section], section: Section) -> int | None:
    """Index of the earlier section this one continues, or None to keep it separate.

    A figure between two halves of a section is an interruption in the page
    stream, not a boundary in the argument, so the halves are looked up past any
    intervening captions and become one section again.
    """

    if not merged or section.kind not in _MERGEABLE_KINDS:
        return None
    index = len(merged) - 1
    while index >= 0 and merged[index].kind in _CAPTION_KINDS:
        index -= 1
    if index < 0 or merged[index].kind != section.kind:
        return None
    if merged[index].name == section.name:
        return index
    if len(section.text) < MIN_SECTION_CHARS:
        return index
    return None


def _merge_small_sections(sections: list[Section]) -> list[Section]:
    """Rejoin interrupted prose, fold undersized fragments, and renumber."""

    merged: list[Section] = []
    for section in sections:
        target = _rejoin_target(merged, section)
        if target is not None:
            previous = merged[target]
            merged[target] = Section(
                previous.name,
                f"{previous.text}\n{section.text}",
                previous.page_start,
                section.page_end if section.page_end is not None else previous.page_end,
                previous.ordinal,
                previous.kind,
            )
            continue
        merged.append(
            Section(
                section.name,
                section.text,
                section.page_start,
                section.page_end,
                len(merged),
                section.kind,
            )
        )
    return merged


def sections_from_pages(pages: Iterable[ParsedPage]) -> list[Section]:
    """Assign every non-empty line to the most recent heading, tracking page spans."""

    sections: list[Section] = []
    name, kind = FRONT_MATTER, "front"
    lines: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    in_appendix = False
    # Set while inside a caption block; holds the section to resume afterwards.
    resume: tuple[str, str] | None = None
    caption_chars = 0
    caption_lines = 0

    def flush() -> None:
        if lines:
            sections.append(
                Section(name, "\n".join(lines), page_start, page_end, len(sections), kind)
            )

    for page in pages:
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading = _classify_heading(line)
            if heading is not None:
                flush()
                new_name, new_kind = heading
                lines = []
                page_start = page.number
                page_end = page.number
                if new_kind in {"figure", "table"}:
                    # Remember the section this caption interrupts, before it is replaced.
                    if resume is None:
                        resume = (name, kind)
                    name, kind = new_name, new_kind
                    # The caption line is content as well as name, so a caption that
                    # completes on one line closes here instead of swallowing the
                    # prose that follows it.
                    lines = [new_name]
                    caption_chars, caption_lines = len(new_name), 1
                    if new_name.endswith(_SENTENCE_END):
                        flush()
                        name, kind = resume
                        resume = None
                        caption_chars = caption_lines = 0
                        lines = []
                    continue
                resume = None
                if new_kind == "appendix":
                    in_appendix = True
                elif new_kind == "body" and in_appendix:
                    new_kind = "appendix"
                name, kind = new_name, new_kind
                continue
            lines.append(line)
            if page.number is not None:
                if page_start is None:
                    page_start = page.number
                page_end = page.number
            if resume is not None:
                caption_chars += len(line)
                caption_lines += 1
                if (
                    line.endswith(_SENTENCE_END)
                    or caption_chars >= CAPTION_MAX_CHARS
                    or caption_lines >= CAPTION_MAX_LINES
                ):
                    flush()
                    name, kind = resume
                    resume = None
                    caption_chars = caption_lines = 0
                    lines = []
                    page_start = page.number
                    page_end = page.number
    flush()
    return _merge_small_sections(sections)


def _pypdf_pages(payload: bytes) -> list[ParsedPage]:
    try:
        reader = PdfReader(BytesIO(payload), strict=False)
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:  # cryptography missing or unsupported filter
                raise PDFParseError(ParseErrorCode.ENCRYPTED) from exc
            if decrypted == PasswordType.NOT_DECRYPTED:
                raise PDFParseError(ParseErrorCode.ENCRYPTED)
        pages = [
            ParsedPage(number=index, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages, start=1)
        ]
    except PDFParseError:
        raise
    except Exception as exc:  # pypdf raises a wide family of errors on damaged files
        raise PDFParseError(ParseErrorCode.CORRUPT) from exc
    if not pages:
        raise PDFParseError(ParseErrorCode.CORRUPT)
    return pages


def _read_pdf(path: Path, max_bytes: int) -> bytes:
    if not path.is_file():
        raise PDFParseError(ParseErrorCode.MISSING, path)
    if path.stat().st_size > max_bytes:
        raise PDFParseError(ParseErrorCode.OVERSIZED, path)
    payload = path.read_bytes()
    if b"%PDF-" not in payload[:1024]:
        raise PDFParseError(ParseErrorCode.NOT_PDF, path)
    return payload


def _failed(code: ParseErrorCode, checksum: str, page_count: int = 0) -> ParseResult:
    return ParseResult(
        status=ParseStatus.FAILED,
        error_code=code,
        sections=(),
        content_sha256=checksum,
        parser_version=PARSER_VERSION,
        page_count=page_count,
        quality="low",
    )


def parse_pdf_result(
    path: str | Path,
    *,
    adapter: PageAdapter = _pypdf_pages,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ParseResult:
    """Parse without raising: failures come back as typed statuses for persistence."""

    target = Path(path)
    try:
        payload = _read_pdf(target, max_bytes)
    except PDFParseError as error:
        checksum = ""
        if error.code not in {ParseErrorCode.MISSING, ParseErrorCode.OVERSIZED}:
            checksum = hashlib.sha256(target.read_bytes()).hexdigest()
        return _failed(error.code, checksum)
    checksum = hashlib.sha256(payload).hexdigest()
    try:
        pages = list(adapter(payload))
    except PDFParseError as error:
        return _failed(error.code, checksum)
    pages = _normalise_pages(pages)
    sections = sections_from_pages(pages)
    if not any(section.text.strip() for section in sections):
        return _failed(ParseErrorCode.EMPTY_TEXT, checksum, page_count=len(pages))
    return ParseResult(
        status=ParseStatus.PARSED,
        error_code=None,
        sections=tuple(sections),
        content_sha256=checksum,
        parser_version=PARSER_VERSION,
        page_count=len(pages),
        quality="low",
    )


def parse_pdf(
    path: str | Path,
    *,
    adapter: PageAdapter = _pypdf_pages,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Section]:
    """Parse a PDF into sections or raise a typed PDFParseError."""

    result = parse_pdf_result(path, adapter=adapter, max_bytes=max_bytes)
    if result.status is ParseStatus.FAILED:
        assert result.error_code is not None
        raise PDFParseError(result.error_code, Path(path))
    return list(result.sections)
