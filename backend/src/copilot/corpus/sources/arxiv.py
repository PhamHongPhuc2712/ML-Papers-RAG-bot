"""arXiv Atom API adapter supplementing accessible preprint versions."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from ..normalize import normalize_arxiv
from .base import ProviderError, Transport, check_response, utc_now

DEFAULT_API_BASE = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivSource:
    source = "arxiv"

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
        self.api_base = str(config.get("api_base", DEFAULT_API_BASE))
        self.ids = [str(item) for item in config.get("ids", [])]
        self.search_query = config.get("search_query")
        self.page_size = int(config.get("page_size", 100))
        self.transport = transport
        self.venue = venue
        self.year = year
        self.track = track
        self.now = now

    def fetch_page(self, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        start = int(cursor) if cursor else 0
        params: dict[str, Any]
        if self.ids:
            batch = self.ids[start : start + self.page_size]
            if not batch:
                return [], None
            params = {"id_list": ",".join(batch), "max_results": len(batch)}
            total = len(self.ids)
        elif self.search_query:
            params = {
                "search_query": str(self.search_query),
                "start": start,
                "max_results": self.page_size,
            }
            total = None
        else:
            return [], None
        response = self.transport.get(self.api_base, params=params)
        check_response(response)
        entries = self._entries(response.body)
        records = [self.normalize(entry) for entry in entries]
        next_start = start + len(entries)
        more = bool(entries) and (total is None or next_start < total)
        return records, (str(next_start) if more else None)

    @staticmethod
    def _entries(body: bytes) -> list[ET.Element]:
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise ProviderError("provider_error", "invalid atom feed") from exc
        return list(root.iter(f"{ATOM}entry"))

    def normalize(self, entry: ET.Element) -> dict[str, Any]:
        def text(tag: str) -> str:
            node = entry.find(tag)
            return (node.text or "").strip() if node is not None else ""

        raw_id = text(f"{ATOM}id")
        arxiv_id, version = normalize_arxiv(raw_id)
        authors = [
            (author.findtext(f"{ATOM}name") or "").strip() for author in entry.iter(f"{ATOM}author")
        ]
        pdf_url = None
        abs_url = raw_id
        for link in entry.iter(f"{ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
            elif link.get("rel") == "alternate":
                abs_url = link.get("href") or abs_url
        external: dict[str, str] = {"arxiv": arxiv_id + (version or "")}
        doi = text(f"{ARXIV}doi")
        if doi:
            external["doi"] = doi
        return {
            "source": self.source,
            "source_item_id": arxiv_id + (version or ""),
            "source_revision": text(f"{ATOM}updated") or "unknown",
            "external_ids": external,
            "title": " ".join(text(f"{ATOM}title").split()),
            "authors": [name for name in authors if name],
            "abstract": " ".join(text(f"{ATOM}summary").split()) or None,
            "venue": {"name": self.venue, "track": self.track},
            "year": self.year,
            "track": self.track,
            # arXiv has no acceptance decision; these records supplement, never admit.
            "decision": "unknown",
            "withdrawn": False,
            "retrieved_at": self.now().isoformat(),
            "source_url": abs_url,
            "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}{version or ''}",
            "first_published_at": text(f"{ATOM}published") or None,
            "redistribution": "unknown",
        }
