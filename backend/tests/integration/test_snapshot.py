"""Snapshot export, validation and restore against a real database."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from sqlalchemy import text

from copilot.corpus.export import ExportError, export_snapshot, restore_snapshot, validate_manifest
from copilot.db.session import assert_safe_test_database

pytestmark = pytest.mark.integration

TABLES = (
    "chunks",
    "paper_versions",
    "paper_identifiers",
    "paper_authors",
    "authors",
    "papers",
    "venues",
)


def _truncate(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))


def _seed(engine) -> dict[str, str]:
    """Two papers: one fully parsed, one whose PDF never parsed."""

    ids = {"parsed": str(uuid4()), "failed": str(uuid4())}
    with engine.begin() as connection:
        venue = connection.execute(
            text(
                "insert into venues (id, name, track) values (gen_random_uuid(), 'ICLR', 'main')"
                " returning id"
            )
        ).scalar_one()
        author = connection.execute(
            text(
                "insert into authors (id, name, normalized_name) values"
                " (gen_random_uuid(), 'Ada Lovelace', 'ada lovelace') returning id"
            )
        ).scalar_one()
        for key, status in (("parsed", "parsed"), ("failed", "oversized")):
            connection.execute(
                text(
                    "insert into papers (id, title, abstract, publication_year, venue_id,"
                    " first_seen_at, acceptance_type, metadata_status) values (:id, :title,"
                    " :abstract, 2024, :venue, now(), 'poster', 'active')"
                ),
                {
                    "id": ids[key],
                    "title": f"A {key} paper",
                    "abstract": "An abstract." if key == "parsed" else None,
                    "venue": venue,
                },
            )
            connection.execute(
                text(
                    "insert into paper_identifiers (id, paper_id, namespace, value) values"
                    " (gen_random_uuid(), :paper_id, 'papercli', :value)"
                ),
                {"paper_id": ids[key], "value": f"id-{key}"},
            )
            connection.execute(
                text(
                    "insert into paper_authors (paper_id, position, author_id) values"
                    " (:paper_id, 0, :author)"
                ),
                {"paper_id": ids[key], "author": author},
            )
            version = connection.execute(
                text(
                    "insert into paper_versions (id, paper_id, source, source_url,"
                    " source_revision, content_sha256, redistribution, parser_version,"
                    " parse_status) values (gen_random_uuid(), :paper_id, 'papercli', 'u',"
                    " 'rev', :sha, 'unknown', 'pypdf-text-v2', :status) returning id"
                ),
                {"paper_id": ids[key], "sha": key * 8, "status": status},
            ).scalar_one()
            if status == "parsed":
                connection.execute(
                    text(
                        "insert into chunks (id, paper_version_id, section_path, section_ordinal,"
                        " kind, ordinal, text, token_count, parser_version, chunker_version)"
                        " values (gen_random_uuid(), :version, 'Method', 0, 'body', 0,"
                        " 'body text', 2, 'pypdf-text-v2', 'fixed-window-v2')"
                    ),
                    {"version": version},
                )
    return ids


def test_snapshot_exports_metadata_and_withholds_unlicensed_text(
    migrated_database, test_settings, tmp_path
):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    _seed(migrated_database)

    manifest = export_snapshot("r1", tmp_path / "snap", engine=migrated_database)

    assert manifest["counts"]["papers"] == 2
    assert manifest["counts"]["chunks_total"] == 1
    # Rights are unknown for this corpus, so no text may leave.
    assert manifest["counts"]["chunks_exported"] == 0
    assert manifest["counts"]["withheld_chunks"] == 1
    assert manifest["rights"]["withheld"] == ["unknown"]
    assert validate_manifest(tmp_path / "snap" / "manifest.json")["run_id"] == "r1"


def test_snapshots_are_immutable(migrated_database, test_settings, tmp_path):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    _seed(migrated_database)
    export_snapshot("r1", tmp_path / "snap", engine=migrated_database)
    with pytest.raises(ExportError, match="snapshot_exists"):
        export_snapshot("r2", tmp_path / "snap", engine=migrated_database)


def test_restore_reproduces_counts_and_identities(migrated_database, test_settings, tmp_path):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    ids = _seed(migrated_database)
    export_snapshot("r1", tmp_path / "snap", engine=migrated_database)

    _truncate(migrated_database)
    counts = restore_snapshot(tmp_path / "snap" / "manifest.json", engine=migrated_database)

    assert counts == {"papers": 2, "identifiers": 2}
    with migrated_database.connect() as connection:
        restored = {
            str(row[0]) for row in connection.execute(text("select id from papers")).all()
        }
    # The canonical work IDs survive the round trip, which is what makes a
    # snapshot a restore rather than a re-ingest.
    assert restored == set(ids.values())


def test_restore_refuses_a_populated_database(migrated_database, test_settings, tmp_path):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    _seed(migrated_database)
    export_snapshot("r1", tmp_path / "snap", engine=migrated_database)
    with pytest.raises(ExportError, match="target_not_empty"):
        restore_snapshot(tmp_path / "snap" / "manifest.json", engine=migrated_database)


def test_tampered_snapshot_is_refused_before_it_touches_the_database(
    migrated_database, test_settings, tmp_path
):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    _seed(migrated_database)
    export_snapshot("r1", tmp_path / "snap", engine=migrated_database)
    (tmp_path / "snap" / "papers.parquet").write_bytes(b"tampered")
    _truncate(migrated_database)
    with pytest.raises(ExportError, match="checksum_mismatch"):
        restore_snapshot(tmp_path / "snap" / "manifest.json", engine=migrated_database)
    with migrated_database.connect() as connection:
        assert connection.execute(text("select count(*) from papers")).scalar_one() == 0
    assert json.loads((tmp_path / "snap" / "manifest.json").read_text())["run_id"] == "r1"
