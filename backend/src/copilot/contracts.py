"""Stable API and model adapter contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    """Reject accidental provider-specific fields at the domain boundary."""

    model_config = ConfigDict(extra="forbid")


class Principal(ContractModel):
    user_id: UUID


class PaperFilters(ContractModel):
    year_from: int | None = None
    year_to: int | None = None
    venues: list[str] = Field(default_factory=list)
    fulltext_only: bool = False

    @field_validator("venues")
    @classmethod
    def strip_venues(cls, value: list[str]) -> list[str]:
        return [venue.strip() for venue in value if venue.strip()]

    @model_validator(mode="after")
    def ordered_years(self) -> PaperFilters:
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("invalid_year_interval")
        return self


class SearchRequest(ContractModel):
    # Spec §10: 1-2,000 characters and 1-50 results, checked before anything is served.
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
    filters: PaperFilters
    limit: int = Field(default=20, ge=1, le=50)
    cursor: str | None = None

    @field_validator("query")
    @classmethod
    def require_terms(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query_blank")
        return value


class RankedPaper(ContractModel):
    paper_id: UUID
    score: float
    scores: dict[str, float]
    rank: int = Field(ge=1)


class PaperSummary(ContractModel):
    title: str
    authors: list[str]
    venue: str
    year: int
    abstract: str | None = None
    pdf_url: str | None = None
    fulltext_indexed: bool


class SearchResponse(ContractModel):
    request_id: UUID
    corpus_release_id: str
    items: list[RankedPaper]
    papers: dict[str, PaperSummary]
    next_cursor: str | None = None
    degraded: bool
    warnings: list[str]


class FeedbackEvent(ContractModel):
    id: UUID
    paper_id: UUID
    impression_id: UUID | None = None
    event_type: Literal["open", "save", "unsave", "like", "dislike", "read", "ask"]
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at_must_be_timezone_aware")
        return value


class RecommendationRequest(ContractModel):
    limit: int = Field(default=20, ge=1, le=100)
    refresh: bool = False


class FeedResponse(ContractModel):
    request_id: UUID
    run_id: UUID
    items: list[RankedPaper]
    reasons: dict[str, dict[str, Any]]
    papers: dict[str, PaperSummary]
    impressions: dict[str, UUID]
    stale: bool
    profile_version: str
    corpus_release_id: str


class Evidence(ContractModel):
    id: str
    source_kind: Literal["paper", "upload"]
    source_id: UUID
    version_id: str
    chunk_id: UUID
    text: str
    section: str
    page_start: int | None = None
    page_end: int | None = None


class Claim(ContractModel):
    text: str
    evidence_ids: list[str]


class GroundedAnswer(ContractModel):
    claims: list[Claim]
    insufficient_evidence: bool


class ChatRequest(ContractModel):
    request_id: UUID
    chat_id: UUID | None = None
    message: str = Field(min_length=1)
    mode: Literal["auto", "general", "research", "document"]
    document_ids: list[UUID]


class JobStatus(ContractModel):
    id: UUID
    status: str
    progress_done: int = Field(ge=0)
    progress_total: int | None = Field(default=None, ge=0)
    error_code: str | None = None


class EmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts while preserving input order."""


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score texts for a query while preserving input order."""


class Generator(Protocol):
    def answer(self, question: str, evidence: list[Evidence]) -> GroundedAnswer:
        """Produce an answer grounded in the supplied evidence."""
