"""Building a venue-year index from the pinned registry parquet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from copilot.corpus.mirror import MirrorIndexError, build_index

COLUMNS = (
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


def _registry(path: Path, rows: list[dict[str, object]]) -> Path:
    table = pa.table({column: [row.get(column) for row in rows] for column in COLUMNS})
    pq.write_table(table, path)
    return path


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "id": "6eebfe7a884184e6",
        "title": "A mirrored paper about retrieval",
        "authors": "Ada Lovelace; Grace Hopper",
        "abstract": "An abstract for the mirrored paper.",
        "venue": "ICLR",
        "year": 2024,
        "track": "ICLR 2024 poster",
        "source": "iclr",
        "pdf_url": "https://openreview.net/pdf?id=mirror-a",
        "forum_url": "https://openreview.net/forum?id=mirror-a",
        "hf_pdf_path": "pdfs/iclr/2024/6e/6eebfe7a884184e6.pdf",
    }
    row.update(overrides)
    return row


def test_index_carries_registry_fields_and_verified_local_pdfs(tmp_path):
    payload = b"%PDF-1.4 mirrored"
    pdf = tmp_path / "pdfs" / "pdfs" / "iclr" / "2024" / "6e" / "6eebfe7a884184e6.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(payload)
    registry = _registry(tmp_path / "papers.parquet", [_row(), _row(id="other", year=2023)])

    out = tmp_path / "iclr-2024.jsonl"
    counts = build_index(registry, venue="ICLR", year=2024, pdf_root=tmp_path / "pdfs", out=out)

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert counts == {"rows": 1, "with_pdf": 1, "missing_pdf": 0}
    (record,) = records
    assert record["id"] == "6eebfe7a884184e6"
    assert record["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert record["abstract"] == "An abstract for the mirrored paper."
    assert record["track"] == "ICLR 2024 poster"
    assert record["forum_url"] == "https://openreview.net/forum?id=mirror-a"
    assert record["pdf_path"] == str(pdf)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["bytes"] == len(payload)


def test_missing_local_pdfs_are_counted_not_invented(tmp_path):
    registry = _registry(tmp_path / "papers.parquet", [_row()])
    out = tmp_path / "iclr-2024.jsonl"
    counts = build_index(registry, venue="ICLR", year=2024, pdf_root=tmp_path / "pdfs", out=out)
    (record,) = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert counts == {"rows": 1, "with_pdf": 0, "missing_pdf": 1}
    assert record["pdf_path"] is None
    assert record["sha256"] is None


def test_empty_selection_is_an_error_rather_than_an_empty_index(tmp_path):
    registry = _registry(tmp_path / "papers.parquet", [_row()])
    with pytest.raises(MirrorIndexError, match="no_rows"):
        build_index(
            registry,
            venue="ICLR",
            year=2025,
            pdf_root=tmp_path / "pdfs",
            out=tmp_path / "out.jsonl",
        )
