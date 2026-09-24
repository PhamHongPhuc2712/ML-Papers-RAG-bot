"""What may leave this machine: rights, privacy and manifest integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.corpus.export import (
    SCHEMA_VERSION,
    ExportError,
    coverage_rows,
    public_export_rows,
    record_digest,
    shard_paths,
    validate_manifest,
)


def test_private_or_unlicensed_text_is_not_exported():
    rows = [
        {"id": "a", "kind": "corpus", "redistribution": "allowed", "text": "public"},
        {"id": "b", "kind": "upload", "redistribution": "allowed", "text": "private"},
        {"id": "c", "kind": "corpus", "redistribution": "unknown", "text": "unknown"},
    ]
    assert [r["id"] for r in public_export_rows(rows)] == ["a"]


@pytest.mark.parametrize("redistribution", ["restricted", "unknown", "", None])
def test_only_explicitly_allowed_text_leaves(redistribution):
    rows = [{"id": "x", "kind": "corpus", "redistribution": redistribution, "text": "t"}]
    assert public_export_rows(rows) == []


def test_upload_rows_never_leave_even_when_marked_allowed():
    """Private uploads and user history never enter corpus exports (spec §4)."""

    rows = [{"id": "u", "kind": "upload", "redistribution": "allowed", "text": "private"}]
    assert public_export_rows(rows) == []


def test_coverage_counts_missing_abstracts_and_missing_fulltext_separately():
    papers = [
        {"venue": "ICLR", "year": 2024, "abstract": "a", "parse_status": "parsed"},
        {"venue": "ICLR", "year": 2024, "abstract": None, "parse_status": "parsed"},
        {"venue": "ICLR", "year": 2024, "abstract": "a", "parse_status": "oversized"},
        {"venue": "JMLR", "year": 2023, "abstract": "a", "parse_status": "not_pdf"},
    ]
    rows = {(row["venue"], row["year"]): row for row in coverage_rows(papers, expected={})}
    iclr = rows[("ICLR", 2024)]
    assert iclr["papers"] == 3
    assert iclr["with_abstract"] == 2
    assert iclr["with_fulltext"] == 2
    assert iclr["failed_parse"] == 1
    assert iclr["expected"] is None
    # An unknown denominator must never be reported as complete coverage.
    assert iclr["coverage_pct"] is None


def test_coverage_percentage_needs_a_known_denominator():
    papers = [{"venue": "ICLR", "year": 2024, "abstract": "a", "parse_status": "parsed"}] * 2
    (row,) = coverage_rows(papers, expected={("ICLR", 2024): 4})
    assert row["expected"] == 4
    assert row["coverage_pct"] == 50.0


def _manifest(
    tmp_path: Path, shard: bytes = b"payload", *, schema_version: int = SCHEMA_VERSION
) -> Path:
    import hashlib

    (tmp_path / "papers.parquet").write_bytes(shard)
    manifest = {
        "schema_version": schema_version,
        "run_id": "r1",
        "created_at": "2026-09-16T00:00:00+00:00",
        "shards": [
            {
                "path": "papers.parquet",
                "rows": 1,
                "bytes": len(shard),
                "sha256": hashlib.sha256(shard).hexdigest(),
            }
        ],
        "counts": {"papers": 1},
        "versions": {"parser": "pypdf-text-v2", "chunker": "fixed-window-v2"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_validate_accepts_an_intact_snapshot(tmp_path):
    assert validate_manifest(_manifest(tmp_path))["run_id"] == "r1"


def test_checksum_tampering_fails(tmp_path):
    path = _manifest(tmp_path)
    (tmp_path / "papers.parquet").write_bytes(b"tampered")
    with pytest.raises(ExportError, match="checksum_mismatch"):
        validate_manifest(path)


def test_schema_mismatch_fails_before_anything_is_read(tmp_path):
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExportError, match="schema_version_unsupported"):
        validate_manifest(path)


def test_a_missing_shard_is_reported(tmp_path):
    path = _manifest(tmp_path)
    (tmp_path / "papers.parquet").unlink()
    with pytest.raises(ExportError, match="shard_missing"):
        validate_manifest(path)


def test_a_version_one_snapshot_still_validates_and_names_its_tables(tmp_path):
    """The G1 snapshot predates shard roles; its file names are its tables."""

    manifest = validate_manifest(_manifest(tmp_path, schema_version=1))
    assert shard_paths(manifest, "papers") == ["papers.parquet"]
    assert shard_paths(manifest, "chunks") == []


def test_record_digest_depends_on_order_and_content():
    rows = [("a", "1" * 64), ("b", "2" * 64)]
    assert record_digest(rows) == record_digest(list(rows))
    assert record_digest(rows) != record_digest(list(reversed(rows)))
    assert record_digest(rows) != record_digest([("a", "1" * 64), ("b", "3" * 64)])
