"""Semantic Scholar Graph API adapter for optional identifier/citation enrichment.

Enrichment never admits a paper; it attaches DOI/arXiv aliases and a citation
observation to a work that membership sources already admitted. Failures leave
the base record searchable (spec §4).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from ..normalize import normalize_title
from .base import Transport, check_response, utc_now

DEFAULT_API_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,externalIds,citationCount"


class SemanticScholarSource:
    source = "semantic_scholar"

    def __init__(
        self,
        config: Mapping[str, Any],
        transport: Transport,
        *,
        venue: str,
        year: int,
        track: str,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self.api_base = str(config.get("api_base", DEFAULT_API_BASE)).rstrip("/")
        self.api_key = config.get("api_key")
        self.paper_ids = [str(item) for item in config.get("paper_ids", [])]
        self.transport = transport
        self.venue = venue
        self.year = year
        self.track = track
        self.now = now

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": str(self.api_key)} if self.api_key else {}

    def lookup(
        self, *, title: str, doi: str | None = None, arxiv: str | None = None
    ) -> dict[str, Any] | None:
        """Find the S2 record by strong ID when known, else by exact normalized title."""

        if doi or arxiv:
            key = f"DOI:{doi}" if doi else f"arXiv:{arxiv}"
            response = self.transport.get(
                f"{self.api_base}/paper/{key}", params={"fields": FIELDS}, headers=self._headers()
            )
            if response.status_code == 404:
                return None
            check_response(response)
            data = response.json()
            return dict(data) if isinstance(data, Mapping) else None
        response = self.transport.get(
            f"{self.api_base}/paper/search",
            params={"query": title, "fields": FIELDS, "limit": 5},
            headers=self._headers(),
        )
        check_response(response)
        data = response.json()
        candidates = data.get("data", []) if isinstance(data, Mapping) else []
        wanted = normalize_title(title)
        for candidate in candidates:
            if isinstance(candidate, Mapping) and candidate.get("title"):
                try:
                    if normalize_title(str(candidate["title"])) == wanted:
                        return dict(candidate)
                except ValueError:
                    continue
        return None

    def enrichment_record(
        self,
        found: Mapping[str, Any],
        *,
        title: str,
        authors: Sequence[str],
        known_ids: Mapping[str, str],
    ) -> dict[str, Any]:
        external: dict[str, str] = dict(known_ids)
        paper_id = str(found.get("paperId") or "")
        if paper_id:
            external["semantic_scholar"] = paper_id
        ids = found.get("externalIds")
        if isinstance(ids, Mapping):
            if ids.get("DOI"):
                external["doi"] = str(ids["DOI"])
            if ids.get("ArXiv"):
                external["arxiv"] = str(ids["ArXiv"])
        retrieved = self.now()
        citations = found.get("citationCount")
        return {
            "source": self.source,
            "source_item_id": paper_id or title,
            "source_revision": retrieved.date().isoformat(),
            "external_ids": external,
            "title": title,
            "authors": list(authors),
            "venue": {"name": self.venue, "track": self.track},
            "year": self.year,
            "track": self.track,
            "decision": "unknown",
            "withdrawn": False,
            "retrieved_at": retrieved.isoformat(),
            "citation_count": int(citations) if isinstance(citations, int) else None,
            "redistribution": "unknown",
        }

    def fetch_page(self, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        """Iterate configured paper IDs; used for explicit enrichment lists."""

        start = int(cursor) if cursor else 0
        if start >= len(self.paper_ids):
            return [], None
        paper_id = self.paper_ids[start]
        response = self.transport.get(
            f"{self.api_base}/paper/{paper_id}", params={"fields": FIELDS}, headers=self._headers()
        )
        check_response(response)
        found = response.json()
        records = []
        if isinstance(found, Mapping):
            records.append(
                self.enrichment_record(
                    found, title=str(found.get("title") or ""), authors=[], known_ids={}
                )
            )
        next_start = start + 1
        return records, (str(next_start) if next_start < len(self.paper_ids) else None)
