"""Corpus migration replay against the isolated test database."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from copilot.db.session import migrate_database

pytestmark = pytest.mark.integration

CORPUS_TABLES = {
    "venues",
    "papers",
    "authors",
    "paper_identifiers",
    "paper_versions",
    "paper_authors",
    "source_records",
    "field_provenance",
    "identity_conflicts",
    "quarantine_records",
    "paper_redirects",
}


def test_migrations_replay_up_down_up(migrated_database):
    engine = migrated_database
    assert CORPUS_TABLES <= set(inspect(engine).get_table_names())

    migrate_database(engine, "base")
    remaining = set(inspect(engine).get_table_names())
    assert not (CORPUS_TABLES & remaining)
    assert "corpus_releases" not in remaining

    migrate_database(engine, "head")
    names = set(inspect(engine).get_table_names())
    assert CORPUS_TABLES <= names
    assert {"corpus_releases", "active_release"} <= names


def test_corpus_constraints_are_created_hardened(migrated_database):
    with migrated_database.connect() as connection:
        version_index = connection.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'uq_paper_versions_source_revision_content_version'"
            )
        ).scalar_one()
        conflict_constraint = connection.execute(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conname = 'uq_identity_conflicts_source_reason'"
            )
        ).scalar_one_or_none()
    assert "NULLS NOT DISTINCT" in version_index.upper()
    assert conflict_constraint == "uq_identity_conflicts_source_reason"
