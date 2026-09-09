"""PDF parsing into traceable sections behind a page-adapter boundary.

The first adapter extracts plain text per page with pypdf and is labeled low
quality (spec §4). Docling or another structured converter can replace it by
implementing :class:`PageAdapter` and bumping ``PARSER_VERSION``; chunk
identities change with the version, which is the intended provenance signal.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Protocol

from pypdf import PasswordType, PdfReader

from .chunk import Section

ADAPTER = "pypdf-text"
PARSER_VERSION = "pypdf-text-v1"
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


def _classify_heading(line: str) -> tuple[str, str] | None:
    if len(line) > 90:
        return None
    named = _NAMED.get(line.rstrip(".:").lower())
    if named is not None:
        return named
    caption = _CAPTION.match(line)
    if caption is not None:
        return line, caption.group("kind").lower()
    if _APPENDIX.match(line):
        return line, "appendix"
    numbered = _NUMBERED.match(line)
    if numbered is not None and not numbered.group("title").endswith("."):
        return numbered.group("title").strip(), "body"
    return None


def sections_from_pages(pages: Iterable[ParsedPage]) -> list[Section]:
    """Assign every non-empty line to the most recent heading, tracking page spans."""

    sections: list[Section] = []
    name, kind = FRONT_MATTER, "front"
    lines: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    in_appendix = False

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
                name, kind = heading
                if kind == "appendix":
                    in_appendix = True
                elif kind == "body" and in_appendix:
                    kind = "appendix"
                lines = []
                page_start = page.number
                page_end = page.number
                continue
            lines.append(line)
            if page.number is not None:
                if page_start is None:
                    page_start = page.number
                page_end = page.number
    flush()
    return sections


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
