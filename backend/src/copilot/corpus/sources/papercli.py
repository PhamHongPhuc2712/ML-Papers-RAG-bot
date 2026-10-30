"""Local-mirror adapter over a captured venue-year index (spec §4).

`corpus mirror-index` projects one venue-year out of the papercli registry
parquet and pairs each row with its mirrored PDF, checksummed on this machine.
This adapter reads that index and touches no network: the mirror is the source,
and the dataset revision it was captured at is the source revision. That keeps
ingestion replayable and keeps Hugging Face an offline artifact store rather
than a request-time dependency.

The registry carries eleven columns, including authors, the full abstract, the
venue's own presentation label and the PDF path — the five-column `browse/`
views this adapter previously read carried none of those. Two consequences:

* Records carry real ``authors``, so P1.2's title matching keeps its
  author-compatibility signal.
* ``track`` carries the venue's own label (``ICLR 2024 poster``, ``Submitted
  to ICLR 2023``, ``Findings of the ACL: EMNLP 2024``, ``main``), so membership
  is read off each record by :func:`classify_track` rather than asserted for a
  whole listing. A record keeps the raw label in ``source_track`` beside the
  canonical track it was classified into.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..mirror import split_authors
from .base import Transport, classify_track, utc_now

_OPENREVIEW_HOSTS = frozenset({"openreview.net", "api2.openreview.net"})


def openreview_id(url: object) -> str | None:
    """The forum ID an OpenReview URL carries, or None for other venues."""

    parts = urlsplit(str(url or ""))
    if parts.hostname not in _OPENREVIEW_HOSTS:
        return None
    values = parse_qs(parts.query).get("id")
    return values[0] if values else None


class PapercliSource:
    """Page through a mirrored venue-year index, emitting normalized records."""

    source = "papercli"

    def __init__(
        self,
        config: Mapping[str, Any],
        transport: Transport | None = None,
        *,
        venue: str,
        year: int,
        track: str,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        # transport is accepted for a uniform adapter signature and never used:
        # the mirror is on local disk.
        # The mirror lives under DATA_DIR, which differs per machine, so the
        # tracked manifest names it by variable rather than by absolute path.
        records = Path(os.path.expandvars(str(config["records"])))
        root = config.get("root_dir")
        self.root_dir = Path(os.path.expandvars(str(root))) if root else records.parent
        self.records_path = records if records.is_absolute() else self.root_dir / records
        self.page_size = max(1, int(config.get("page_size", 500)))
        self.dataset = str(config.get("dataset", ""))
        self.revision = str(config.get("dataset_revision", "")) or "unknown"
        self.pdf_dataset = str(config.get("pdf_dataset", ""))
        self.pdf_revision = str(config.get("pdf_dataset_revision", "")) or "unknown"
        self.redistribution = str(config.get("redistribution", "unknown"))
        self.venue = venue
        self.year = year
        self.track = track
        self.now = now
        self._rows: list[Mapping[str, Any]] | None = None

    def _listing(self) -> list[Mapping[str, Any]]:
        if self._rows is None:
            rows: list[Mapping[str, Any]] = []
            with self.records_path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    decoded = json.loads(line)
                    if not isinstance(decoded, Mapping):
                        raise ValueError(f"mirror_record_invalid:{number}")
                    rows.append(decoded)
            self._rows = rows
        return self._rows

    def fetch_page(self, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        start = int(cursor) if cursor else 0
        rows = self._listing()
        page = rows[start : start + self.page_size]
        next_start = start + len(page)
        return (
            [self.normalize(row) for row in page],
            str(next_start) if next_start < len(rows) else None,
        )

    def _pdf_path(self, row: Mapping[str, Any]) -> str | None:
        value = row.get("pdf_path")
        if not value:
            return None
        path = Path(str(value))
        return str(path if path.is_absolute() else self.root_dir / path)

    def normalize(self, row: Mapping[str, Any]) -> dict[str, Any]:
        item_id = str(row.get("id") or row.get("forum_id") or row.get("title") or "")
        external: dict[str, str] = {}
        if item_id:
            external["papercli"] = item_id
        forum = openreview_id(row.get("forum_url"))
        if forum:
            external["openreview"] = forum
        for namespace in ("doi", "arxiv"):
            if row.get(namespace):
                external[namespace] = str(row[namespace])
        authors = row.get("authors")
        names = split_authors(authors) if isinstance(authors, str) else list(authors or [])
        year = row.get("year")
        # The venue's own label is the membership evidence; an absent or
        # unrecognized one classifies as ineligible rather than as accepted.
        source_track = row.get("track")
        track, decision = classify_track(source_track)
        return {
            "source": self.source,
            "source_item_id": item_id,
            "source_revision": self.revision,
            "external_ids": external,
            "title": str(row.get("title") or ""),
            "authors": [str(name) for name in names],
            "abstract": row.get("abstract") or row.get("snippet"),
            "venue": {"name": str(row.get("venue") or self.venue), "track": track},
            "year": int(year) if isinstance(year, int) else self.year,
            "track": track,
            # The label kept verbatim beside the track it classified into, so a
            # record carries the evidence and not only the conclusion.
            "source_track": source_track,
            "decision": decision,
            "withdrawn": False,
            "acceptance_decision": decision,
            "retrieved_at": self.now().isoformat(),
            "source_url": row.get("forum_url"),
            "pdf_url": row.get("pdf_url") or row.get("openreview_pdf_url"),
            # Local mirror copy: the worker adopts this file instead of downloading.
            "pdf_path": self._pdf_path(row),
            "pdf_sha256": row.get("sha256"),
            "pdf_bytes": row.get("bytes"),
            "mirror": {
                "dataset": self.dataset,
                "dataset_revision": self.revision,
                "pdf_dataset": self.pdf_dataset,
                "pdf_dataset_revision": self.pdf_revision,
            },
            # Per-paper rights are not verified by the dataset card; P1.5 exports
            # only records explicitly marked allowed.
            "redistribution": self.redistribution,
        }
