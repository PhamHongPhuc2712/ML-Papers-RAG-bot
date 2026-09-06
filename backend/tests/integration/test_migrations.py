"""PostgreSQL migration regressions for corpus identity hardening."""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from copilot.db.session import assert_safe_test_database, migrate_database

pytestmark = pytest.mark.integration


_IDENTITY_TABLES = (
    "paper_authors",
    "field_provenance",
    "source_records",
    "quarantine_records",
    "identity_conflicts",
    "paper_redirects",
    "paper_versions",
    "paper_identifiers",
    "authors",
    "papers",
    "venues",
)


def _truncate_identity_tables(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(_IDENTITY_TABLES)} CASCADE"))


def test_conflict_constraint_migration_reconciles_legacy_duplicates(
    migrated_database, test_settings
) -> None:
    assert_safe_test_database(test_settings)
    duplicate_source_id = UUID("00000000-0000-0000-0000-000000000101")
    retained_id = UUID("00000000-0000-0000-0000-000000000201")
    deleted_id = UUID("00000000-0000-0000-0000-000000000202")
    unrelated_id = UUID("00000000-0000-0000-0000-000000000203")

    try:
        migrate_database(migrated_database, "base")
        migrate_database(migrated_database, "0001_corpus")
        _truncate_identity_tables(migrated_database)
        with migrated_database.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO papers
                        (id, title, first_seen_at, acceptance_type, metadata_status)
                    VALUES (:paper_id, 'Legacy migration fixture', CURRENT_TIMESTAMP,
                            'unknown', 'active')
                    """
                ),
                {"paper_id": duplicate_source_id},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO source_records
                        (id, paper_id, source, source_revision, retrieved_at,
                         content_sha256, artifact_path, raw_json)
                    VALUES (:source_id, :paper_id, 'legacy', 'r1', CURRENT_TIMESTAMP,
                            :checksum, '/tmp/legacy.json', '{}'::jsonb)
                    """
                ),
                {
                    "source_id": duplicate_source_id,
                    "paper_id": duplicate_source_id,
                    "checksum": "1" * 64,
                },
            )
            for conflict_id, reason, created_at in (
                (retained_id, "incompatible_metadata", "2026-01-01T00:00:00+00:00"),
                (deleted_id, "incompatible_metadata", "2026-01-02T00:00:00+00:00"),
                (unrelated_id, "contradictory_strong_ids", "2026-01-03T00:00:00+00:00"),
            ):
                connection.execute(
                    text(
                        """
                        INSERT INTO identity_conflicts
                            (id, reason, status, paper_id, source_record_id, details, created_at)
                        VALUES (:conflict_id, :reason, 'open', :paper_id,
                                :source_id, '{}'::jsonb, :created_at)
                        """
                    ),
                    {
                        "conflict_id": conflict_id,
                        "reason": reason,
                        "paper_id": duplicate_source_id,
                        "source_id": duplicate_source_id,
                        "created_at": created_at,
                    },
                )

        migrate_database(migrated_database, "head")
        with migrated_database.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT id FROM identity_conflicts
                    WHERE source_record_id = :source_id
                    ORDER BY created_at, id
                    """
                ),
                {"source_id": duplicate_source_id},
            ).scalars().all()
            assert rows == [retained_id, unrelated_id]

        unique_constraints = inspect(migrated_database).get_unique_constraints(
            "identity_conflicts"
        )
        assert any(
            item["name"] == "uq_identity_conflicts_source_reason"
            for item in unique_constraints
        )
        with pytest.raises(IntegrityError):
            with migrated_database.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO identity_conflicts
                            (id, reason, status, paper_id, source_record_id, details)
                        VALUES (:conflict_id, 'incompatible_metadata', 'open', :paper_id,
                                :source_id, '{}'::jsonb)
                        """
                    ),
                    {
                        "conflict_id": UUID("00000000-0000-0000-0000-000000000204"),
                        "paper_id": duplicate_source_id,
                        "source_id": duplicate_source_id,
                    },
                )
    finally:
        migrate_database(migrated_database, "head")
        _truncate_identity_tables(migrated_database)
