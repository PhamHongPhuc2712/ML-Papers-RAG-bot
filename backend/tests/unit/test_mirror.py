"""Building a venue-year index from the pinned registry parquet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from copilot.corpus.mirror import (
    MirrorIndexError,
    build_index,
    mirror_venue_year,
    safe_member_path,
    shard_repo,
)

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


def test_shard_repo_names_the_per_venue_pdf_shard():
    assert shard_repo("ICLR") == "GenAI4ELab/papercli-papers-iclr"
    assert shard_repo("Interspeech") == "GenAI4ELab/papercli-papers-interspeech"


@pytest.mark.parametrize(
    "member",
    ["/etc/passwd", "../outside.pdf", "pdfs/../../outside.pdf", "", "   ", "C:/windows/x.pdf"],
)
def test_member_paths_that_escape_the_mirror_root_are_refused(member):
    """hf_pdf_path is data; it must not be able to write outside the mirror."""

    with pytest.raises(MirrorIndexError, match="unsafe_member_path"):
        safe_member_path(member)


def test_safe_member_paths_are_accepted():
    assert safe_member_path("pdfs/iclr/2024/6e/6eebfe7a884184e6.pdf") == Path(
        "pdfs/iclr/2024/6e/6eebfe7a884184e6.pdf"
    )


def _fetcher(payload: bytes, calls: list[str]):
    def fetch(repo: str, revision: str, member: str, root: Path) -> Path:
        calls.append(member)
        target = root / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    return fetch


def test_mirror_downloads_what_is_missing_and_skips_what_is_present(tmp_path):
    payload = b"%PDF-1.4 mirrored"
    rows = [_row(), _row(id="second", hf_pdf_path="pdfs/iclr/2024/se/second.pdf")]
    registry = _registry(tmp_path / "papers.parquet", rows)
    root = tmp_path / "papercli"
    present = root / "pdfs" / "pdfs" / "iclr" / "2024" / "6e" / "6eebfe7a884184e6.pdf"
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_bytes(payload)

    calls: list[str] = []
    result = mirror_venue_year(
        registry,
        venue="ICLR",
        year=2024,
        mirror_root=root,
        revision="deadbeef",
        fetch=_fetcher(payload, calls),
    )
    assert calls == ["pdfs/iclr/2024/se/second.pdf"]
    assert result["downloaded"] == 1
    assert result["already_present"] == 1
    assert result["failed"] == 0
    assert result["with_pdf"] == 2
    index = [json.loads(line) for line in (root / "iclr-2024.jsonl").read_text().splitlines()]
    assert {record["sha256"] for record in index} == {hashlib.sha256(payload).hexdigest()}
    assert all(record["mirror_revision"] == "deadbeef" for record in index)


def test_a_failed_download_is_reported_rather_than_silently_skipped(tmp_path):
    registry = _registry(tmp_path / "papers.parquet", [_row()])

    def fetch(repo: str, revision: str, member: str, root: Path) -> Path:
        raise OSError("connection reset")

    result = mirror_venue_year(
        registry,
        venue="ICLR",
        year=2024,
        mirror_root=tmp_path / "papercli",
        revision="deadbeef",
        fetch=fetch,
    )
    assert result["failed"] == 1
    assert result["downloaded"] == 0
    assert result["with_pdf"] == 0
    assert result["missing_pdf"] == 1


def test_mirror_refuses_to_start_without_free_disk(tmp_path):
    registry = _registry(tmp_path / "papers.parquet", [_row()])
    with pytest.raises(MirrorIndexError, match="insufficient_free_space"):
        mirror_venue_year(
            registry,
            venue="ICLR",
            year=2024,
            mirror_root=tmp_path / "papercli",
            revision="deadbeef",
            fetch=_fetcher(b"x", []),
            min_free_bytes=10 * 1024**3,
            free_bytes=lambda path: 1024,
        )


def test_only_rows_the_manifest_admits_are_fetched_or_indexed(tmp_path):
    """Findings, workshops and submissions are filtered before the download."""

    rows = [
        _row(id="main1", track="ICLR 2024 poster"),
        _row(
            id="find1",
            track="Findings of the Association for Computational Linguistics: ACL 2024",
            hf_pdf_path="pdfs/iclr/2024/fi/find1.pdf",
        ),
        _row(id="sub1", track="Submitted to ICLR 2023",
             hf_pdf_path="pdfs/iclr/2024/su/sub1.pdf"),
        _row(id="ws1", track="Proceedings of the 8th Workshop on Things",
             hf_pdf_path="pdfs/iclr/2024/ws/ws1.pdf"),
    ]
    registry = _registry(tmp_path / "papers.parquet", rows)
    calls: list[str] = []
    result = mirror_venue_year(
        registry,
        venue="ICLR",
        year=2024,
        mirror_root=tmp_path / "papercli",
        revision="deadbeef",
        fetch=_fetcher(b"%PDF-1.4 x", calls),
    )
    assert calls == ["pdfs/iclr/2024/6e/6eebfe7a884184e6.pdf"]
    assert result["rows"] == 1
    index = [
        json.loads(line)
        for line in (tmp_path / "papercli" / "iclr-2024.jsonl").read_text().splitlines()
    ]
    assert [record["id"] for record in index] == ["main1"]
