"""GET /v1/corpus/coverage over a real database."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from copilot.db.session import assert_safe_test_database

pytestmark = pytest.mark.integration


def _seed(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE chunks, paper_versions, papers, venues CASCADE"))
        venue = connection.execute(
            text(
                "insert into venues (id, name, track) values (gen_random_uuid(), 'ICLR', 'main')"
                " returning id"
            )
        ).scalar_one()
        for index, (abstract, status) in enumerate(
            [("a", "parsed"), (None, "parsed"), ("a", "oversized")]
        ):
            paper = connection.execute(
                text(
                    "insert into papers (id, title, abstract, publication_year, venue_id,"
                    " first_seen_at, acceptance_type, metadata_status) values"
                    " (gen_random_uuid(), :title, :abstract, 2024, :venue, now(), 'poster',"
                    " 'active') returning id"
                ),
                {"title": f"Paper {index}", "abstract": abstract, "venue": venue},
            ).scalar_one()
            connection.execute(
                text(
                    "insert into paper_versions (id, paper_id, source, source_url,"
                    " source_revision, content_sha256, redistribution, parser_version,"
                    " parse_status) values (gen_random_uuid(), :paper, 'papercli', 'u', 'rev',"
                    " :sha, 'unknown', 'pypdf-text-v2', :status)"
                ),
                {"paper": paper, "sha": f"{index:064d}", "status": status},
            )


def test_coverage_reports_abstract_and_fulltext_separately(
    api_client, migrated_database, test_settings
):
    assert_safe_test_database(test_settings)
    _seed(migrated_database)

    response = api_client.get("/v1/corpus/coverage")
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["as_of"]
    (row,) = body["venue_years"]
    assert (row["venue"], row["year"]) == ("ICLR", 2024)
    assert row["papers"] == 3
    assert row["with_abstract"] == 2
    assert row["with_fulltext"] == 2
    assert row["failed_parse"] == 1
    assert body["totals"]["papers"] == 3


def test_coverage_is_public_and_leaks_no_configuration(
    api_client, migrated_database, test_settings
):
    assert_safe_test_database(test_settings)
    _seed(migrated_database)
    body = api_client.get("/v1/corpus/coverage").json()
    serialized = str(body)
    for secret in ("password", "postgresql://", "secret", "token"):
        assert secret not in serialized.lower()
