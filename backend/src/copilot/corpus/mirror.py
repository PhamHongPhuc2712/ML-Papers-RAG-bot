"""Build a venue-year index from the pinned paper registry.

The papercli dataset publishes two artifacts: a registry parquet holding every
venue-year's metadata, and per-venue PDF shards. The registry is the record of
what exists; this module projects one venue-year out of it and pairs each row
with the mirrored PDF already on disk, checksumming what it finds.

The resulting JSONL is what the ingestion adapter reads, so the parquet reader
stays out of the ingest path and every downstream record carries a checksum
this module actually computed rather than one a remote index claimed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# The registry columns this project depends on. The per-venue `browse/` views
# publish only five of them, which is why the root registry is the source.
REGISTRY_COLUMNS = (
    "id",
    "title",
    "authors",
    "abstract",
    "venue",
    "year",
    "track",
    "source",
    "pdf_url",
    "forum_url",
    "hf_pdf_path",
)
_READ_BLOCK = 1 << 20


class MirrorIndexError(RuntimeError):
    """The registry cannot produce a usable index for this venue-year."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def split_authors(value: object) -> list[str]:
    """The registry stores authors as one semicolon-separated string."""

    if not value:
        return []
    return [name.strip() for name in str(value).split(";") if name.strip()]


def checksum(path: Path) -> tuple[str, int]:
    """Stream a local file to a sha256 and byte count."""

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def read_registry(registry: str | Path, *, venue: str, year: int) -> list[dict[str, Any]]:
    """Read one venue-year out of the registry parquet, sorted for replayability."""

    try:
        table = pq.read_table(
            registry,
            columns=list(REGISTRY_COLUMNS),
            filters=[("venue", "=", venue), ("year", "=", year)],
        )
    except (KeyError, ValueError) as error:  # missing column or unreadable file
        raise MirrorIndexError("registry_unreadable", str(error)) from error
    rows: list[dict[str, Any]] = table.to_pylist()
    if not rows:
        raise MirrorIndexError("no_rows", f"{venue}:{year}")
    rows.sort(key=lambda row: str(row.get("id")))
    return rows


def build_index(
    registry: str | Path,
    *,
    venue: str,
    year: int,
    pdf_root: str | Path,
    out: str | Path,
) -> dict[str, int]:
    """Write the venue-year index and report how many rows have a local PDF.

    A row whose PDF is absent keeps a null path and is counted, never invented:
    a missing full text leaves a searchable abstract (spec §4).
    """

    rows = read_registry(registry, venue=venue, year=year)
    root = Path(pdf_root)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_suffix(target.suffix + ".partial")
    counts = {"rows": len(rows), "with_pdf": 0, "missing_pdf": 0}
    with staged.open("w", encoding="utf-8") as handle:
        for row in rows:
            relative = row.get("hf_pdf_path")
            local: Path | None = None
            digest: str | None = None
            size: int | None = None
            if relative:
                candidate = root / str(relative)
                if candidate.is_file():
                    digest, size = checksum(candidate)
                    local = candidate
            counts["with_pdf" if local is not None else "missing_pdf"] += 1
            record = {
                **{column: row.get(column) for column in REGISTRY_COLUMNS},
                "authors": split_authors(row.get("authors")),
                "pdf_path": str(local) if local is not None else None,
                "sha256": digest,
                "bytes": size,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    staged.replace(target)
    return counts
