"""Corpus identity and provenance helpers."""

from .dedupe import (
    IdentityConflictError,
    QuarantineError,
    RecordValidationError,
    merge_paper_identities,
    merge_papers,
    resolve_paper,
)
from .normalize import normalize_arxiv, normalize_doi, normalize_title

__all__ = [
    "IdentityConflictError",
    "QuarantineError",
    "RecordValidationError",
    "merge_paper_identities",
    "merge_papers",
    "normalize_arxiv",
    "normalize_doi",
    "normalize_title",
    "resolve_paper",
]
