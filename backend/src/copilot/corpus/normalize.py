"""Deterministic normalization for publication identifiers and metadata matching."""

from __future__ import annotations

import re
import unicodedata


def normalize_doi(value: str) -> str:
    """Normalize and validate a DOI supplied by a source provider."""

    if not isinstance(value, str):
        raise ValueError("invalid_doi")
    value = value.strip().lower()
    value = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value)
    if not re.fullmatch(r"10\.\d{4,9}/\S+", value):
        raise ValueError("invalid_doi")
    return value


def normalize_arxiv(value: str) -> tuple[str, str | None]:
    """Return a versionless arXiv identifier and an optional ``vN`` suffix."""

    if not isinstance(value, str):
        raise ValueError("invalid_arxiv_id")
    value = re.sub(r"^https?://arxiv.org/(abs|pdf)/", "", value.strip())
    value = value.removesuffix(".pdf")
    match = re.fullmatch(
        r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", value
    )
    if not match:
        raise ValueError("invalid_arxiv_id")
    return match.group(1), match.group(2)


def normalize_title(value: str) -> str:
    """Normalize Unicode spelling and whitespace for equality checks."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("title_required")
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def normalize_author_name(value: str) -> str:
    """Normalize an author name conservatively for identity comparisons."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("author_name_required")
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip().casefold()
