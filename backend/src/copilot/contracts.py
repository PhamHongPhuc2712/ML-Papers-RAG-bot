"""Stable API and model adapter contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol
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
    # hybrid_rerank_llm is opt-in deep search: a hosted LLM reorders the reranked head.
    mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_llm"]
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


class ListwiseResult(ContractModel):
    """A hosted LLM's order over a reranked head: indices into it, best first.

    ``order`` is always a permutation of every candidate given; ``ranked_by_model``
    counts how many of them the model placed itself before the rest were appended
    in their incoming order.
    """

    order: list[int]
    ranked_by_model: int = Field(ge=1)
    cost_usd: float = Field(ge=0)
    served_model: str
    cached: bool


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


class Evidence(ContractModel):
    id: str
    source_kind: Literal["paper", "upload"]
    source_id: UUID
    version_id: str
    # None only for abstract-only evidence: a paper with no parsed text (P4.1).
    chunk_id: UUID | None
    text: str
    section: str
    page_start: int | None = None
    page_end: int | None = None
    # The hit chunk's text was widened to its section (ORB plan, O4).
    expanded: bool = False
    # The paper's abstract stands in for a body it does not have; labelled, never hidden.
    abstract_only: bool = False


class Claim(ContractModel):
    text: str
    evidence_ids: list[str]


class GroundedAnswer(ContractModel):
    claims: list[Claim]
    insufficient_evidence: bool


class ChatRequest(ContractModel):
    """One user turn. ``request_id`` is the client's idempotency key for the turn."""

    request_id: UUID
    chat_id: UUID | None = None
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message_blank")
        return value


ChatStatus = Literal["pending", "complete", "insufficient", "failed", "interrupted"]
Feedback = Literal["helpful", "not_helpful"]


class StoredEvidence(ContractModel):
    """A passage an answer saw, as it was then: chunks can be re-chunked, history cannot."""

    id: str
    paper_id: UUID
    title: str
    venue: str | None = None
    year: int | None = None
    section: str
    page_start: int | None = None
    page_end: int | None = None
    text: str
    abstract_only: bool = False
    cited: bool


class Reference(ContractModel):
    """A paper the turn's search ranked: the answer's References (RAG-only plan, U1).

    ``read`` marks the papers the turn looked inside for evidence, the first of the ranking.
    Stored as the id, rank and flag only; title, venue and year are the corpus's own.
    """

    paper_id: UUID
    rank: int = Field(ge=1)
    read: bool
    title: str
    venue: str | None = None
    year: int | None = None


class ChatMessage(ContractModel):
    id: UUID
    ordinal: int
    role: Literal["user", "assistant"]
    status: ChatStatus
    text: str
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[StoredEvidence] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    standalone_query: str | None = None
    # Every search the answer ran, in order; empty for answers before 2026-10-30.
    search_queries: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    corpus_release_id: str | None = None
    created_at: datetime
    feedback: Feedback | None = None


class FeedbackRequest(ContractModel):
    """A rating for an answer; null clears it (sign-up plan, A5)."""

    rating: Feedback | None


class ChatSummary(ContractModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ChatList(ContractModel):
    request_id: UUID
    items: list[ChatSummary]


class ChatDetail(ContractModel):
    request_id: UUID
    chat: ChatSummary
    messages: list[ChatMessage]


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
