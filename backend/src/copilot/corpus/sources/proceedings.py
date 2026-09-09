"""Official-proceedings adapter over a captured JSON listing.

Official proceedings decide publication membership (spec §4). Proceedings
sites publish HTML, not a stable API, so each venue-year listing is captured
once into a JSON document of the shape ``{"papers": [{"id", "title",
"authors", "pdf_url", "url", "doi"}]}`` whose URL, as-of date and capture
method are recorded in the manifest. Per-venue HTML capture tooling belongs to
P6.1's verified venue adapters.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from .base import Transport, check_response, utc_now


class ProceedingsSource:
    source = "proceedings"

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
        self.listing_url = str(config["listing_url"])
        self.as_of = str(config.get("as_of", ""))
        self.page_size = int(config.get("page_size", 500))
        self.transport = transport
        self.venue = venue
        self.year = year
        self.track = track
        self.now = now
        self._papers: list[Mapping[str, Any]] | None = None

    def _listing(self) -> list[Mapping[str, Any]]:
        if self._papers is None:
            response = self.transport.get(self.listing_url)
            check_response(response)
            data = response.json()
            papers = data.get("papers", []) if isinstance(data, Mapping) else []
            self._papers = [paper for paper in papers if isinstance(paper, Mapping)]
        return self._papers

    def fetch_page(self, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        start = int(cursor) if cursor else 0
        papers = self._listing()
        page = papers[start : start + self.page_size]
        next_start = start + len(page)
        return [self.normalize(paper) for paper in page], (
            str(next_start) if next_start < len(papers) else None
        )

    def normalize(self, paper: Mapping[str, Any]) -> dict[str, Any]:
        item_id = str(paper.get("id") or paper.get("url") or paper.get("title"))
        external: dict[str, str] = {"proceedings": item_id}
        if paper.get("doi"):
            external["doi"] = str(paper["doi"])
        if paper.get("arxiv"):
            external["arxiv"] = str(paper["arxiv"])
        authors = paper.get("authors")
        return {
            "source": self.source,
            "source_item_id": item_id,
            "source_revision": self.as_of or "unknown",
            "external_ids": external,
            "title": str(paper.get("title") or ""),
            "authors": [str(name) for name in authors] if isinstance(authors, list) else [],
            "abstract": paper.get("abstract"),
            "venue": {"name": self.venue, "track": str(paper.get("track", self.track))},
            "year": self.year,
            "track": str(paper.get("track", self.track)),
            # Presence in the official proceedings is the acceptance signal.
            "decision": "accepted",
            "withdrawn": False,
            "acceptance_decision": str(paper.get("presentation") or "accepted"),
            "retrieved_at": self.now().isoformat(),
            "source_url": paper.get("url"),
            "pdf_url": paper.get("pdf_url"),
            "redistribution": str(paper.get("redistribution", "unknown")),
        }
