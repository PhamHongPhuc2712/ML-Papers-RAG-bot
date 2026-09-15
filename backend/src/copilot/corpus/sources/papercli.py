"""Local-mirror adapter over a captured paper index (spec §4).

The papercli datasets publish one metadata shard per venue-year alongside a PDF
shard. Both are mirrored under ``DATA_DIR`` before a run, so this adapter reads
a local JSONL index and touches no network: the mirror is the source, and the
dataset revision it was captured at is the source revision. That keeps ingestion
replayable and keeps Hugging Face an offline artifact store rather than a
request-time dependency.

Two properties of the index shape this adapter:

* It carries **no acceptance decision**, and a venue-year listing is not always
  the accepted set — ICLR 2023 holds 3,792 rows against roughly 1,574 accepted
  papers, while ICLR 2024 holds exactly its 2,260. Membership is therefore
  asserted per manifest through ``membership_is_acceptance``, checked against the
  venue's published total. Without that assertion every record is ``unknown`` and
  :func:`is_eligible` rejects the listing rather than admitting rejected work.
* It carries **no author list**, so records leave ``authors`` empty. Author
  coverage has to come from enrichment, and title matching in P1.2 loses the
  author-compatibility signal for these records.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import Transport, utc_now


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
        self.membership_is_acceptance = bool(config.get("membership_is_acceptance", False))
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

    def normalize(self, row: Mapping[str, Any]) -> dict[str, Any]:
        item_id = str(row.get("forum_id") or row.get("id") or row.get("title") or "")
        external: dict[str, str] = {}
        if item_id:
            external["openreview"] = item_id
        for namespace in ("doi", "arxiv"):
            if row.get(namespace):
                external[namespace] = str(row[namespace])
        year = row.get("year")
        relative = str(row.get("pdf_path") or "")
        decision = "accepted" if self.membership_is_acceptance else "unknown"
        return {
            "source": self.source,
            "source_item_id": item_id,
            "source_revision": self.revision,
            "external_ids": external,
            "title": str(row.get("title") or ""),
            # The index publishes no authors; enrichment has to supply them.
            "authors": [],
            "abstract": row.get("abstract") or row.get("snippet"),
            "venue": {"name": str(row.get("venue") or self.venue), "track": self.track},
            "year": int(year) if isinstance(year, int) else self.year,
            "track": self.track,
            "decision": decision,
            "withdrawn": False,
            "acceptance_decision": decision if self.membership_is_acceptance else None,
            "retrieved_at": self.now().isoformat(),
            "source_url": row.get("forum_url"),
            "pdf_url": row.get("openreview_pdf_url"),
            # Local mirror copy: the worker adopts this file instead of downloading.
            "pdf_path": str(self.root_dir / relative) if relative else None,
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
