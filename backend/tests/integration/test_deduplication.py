"""PostgreSQL acceptance tests for canonical paper identity and provenance."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import text

from copilot.cli import load_fixtures
from copilot.corpus.dedupe import (
    IdentityConflictError,
    QuarantineError,
    merge_papers,
    resolve_paper,
)
from copilot.db.models import (
    FieldProvenance,
    IdentityConflict,
    Paper,
    PaperIdentifier,
    PaperRedirect,
    PaperVersion,
    QuarantineRecord,
    SourceRecord,
)
from copilot.db.session import assert_safe_test_database, session_factory

pytestmark = pytest.mark.integration


@pytest.fixture
def identity_engine(migrated_database, test_settings):
    """Isolate identity rows while enforcing the destructive test namespace guard."""

    assert_safe_test_database(test_settings)
    tables = (
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
    with migrated_database.begin() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(tables)} CASCADE"))
    yield migrated_database
    with migrated_database.begin() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(tables)} CASCADE"))


def _record(
    title: str = "Reliable Retrieval",
    *,
    source: str = "openreview",
    doi: str | None = None,
    arxiv: str | None = None,
    authors: list[str] | None = None,
    year: int | None = 2024,
    source_revision: str = "revision-1",
    source_url: str = "https://papers.example.test/paper.pdf",
) -> dict[str, object]:
    external_ids: dict[str, str] = {}
    if doi is not None:
        external_ids["doi"] = doi
    if arxiv is not None:
        external_ids["arxiv"] = arxiv
    return {
        "source": source,
        "external_ids": external_ids,
        "title": title,
        "authors": authors or ["Ada Lovelace", "Grace Hopper"],
        "venue": {"name": "ICLR", "track": "main"},
        "year": year,
        "source_revision": source_revision,
        "retrieved_at": "2024-06-01T12:00:00+00:00",
        "acceptance_decision": "accepted",
        "source_url": source_url,
    }


def _resolve(engine, record: dict[str, object], staging_dir: Path) -> UUID:
    record = dict(record)
    record["_staging_dir"] = str(staging_dir)
    factory = session_factory(engine)
    with factory() as session:
        paper_id = resolve_paper(record, session)
        session.commit()
        return paper_id


def test_two_providers_sharing_doi_resolve_to_one_work(identity_engine, tmp_path: Path) -> None:
    first = _resolve(
        identity_engine,
        _record(doi="https://doi.org/10.5555/Shared.DOI", source="proceedings"),
        tmp_path,
    )
    second = _resolve(
        identity_engine,
        _record(doi="doi:10.5555/shared.doi", source="openreview"),
        tmp_path,
    )

    assert first == second
    factory = session_factory(identity_engine)
    with factory() as session:
        assert session.query(Paper).count() == 1
        assert session.query(PaperIdentifier).count() == 1
        assert session.query(SourceRecord).count() == 2


def test_repeated_resolution_is_idempotent(identity_engine, tmp_path: Path) -> None:
    record = _record(doi="10.5555/replay")
    first = _resolve(identity_engine, record, tmp_path)
    second = _resolve(identity_engine, record, tmp_path)

    assert first == second
    factory = session_factory(identity_engine)
    with factory() as session:
        assert session.query(Paper).count() == 1
        assert session.query(SourceRecord).count() == 1
        assert session.query(PaperVersion).count() == 1


def test_missing_doi_is_valid(identity_engine, tmp_path: Path) -> None:
    paper_id = _resolve(identity_engine, _record(), tmp_path)

    factory = session_factory(identity_engine)
    with factory() as session:
        assert session.get(Paper, paper_id) is not None
        assert session.query(PaperIdentifier).count() == 0


def test_invalid_doi_is_quarantined_when_caller_commits_review(
    identity_engine, tmp_path: Path
) -> None:
    record = _record(doi="not a doi")
    record["_staging_dir"] = str(tmp_path)
    factory = session_factory(identity_engine)
    with factory() as session:
        with session.begin():
            with pytest.raises(QuarantineError, match="invalid_doi"):
                resolve_paper(record, session)

    factory = session_factory(identity_engine)
    with factory() as session:
        quarantine = session.query(QuarantineRecord).one()
        assert quarantine.reason == "invalid_doi"
        assert quarantine.source == "openreview"
        assert session.query(Paper).count() == 0


def test_arxiv_versions_share_work_but_keep_distinct_documents(
    identity_engine, tmp_path: Path
) -> None:
    first = _resolve(identity_engine, _record(arxiv="2401.01234v1"), tmp_path)
    second = _resolve(
        identity_engine, _record(arxiv="https://arxiv.org/abs/2401.01234v2"), tmp_path
    )

    assert first == second
    factory = session_factory(identity_engine)
    with factory() as session:
        versions = session.query(PaperVersion).all()
        assert len(versions) == 2
        assert {version.version for version in versions} == {"v1", "v2"}
        identifier = session.query(PaperIdentifier).filter_by(namespace="arxiv").one()
        assert identifier.version is None


def test_doi_and_arxiv_aliases_join_one_compatible_work(identity_engine, tmp_path: Path) -> None:
    first = _resolve(identity_engine, _record(doi="10.5555/alias"), tmp_path)
    second = _resolve(identity_engine, _record(arxiv="2401.01234v1"), tmp_path)

    assert first == second
    factory = session_factory(identity_engine)
    with factory() as session:
        assert {item.namespace for item in session.query(PaperIdentifier).all()} == {
            "doi",
            "arxiv",
        }


def test_preprint_and_proceedings_aliases_can_have_distinct_years(
    identity_engine, tmp_path: Path
) -> None:
    first = _resolve(identity_engine, _record(doi="10.5555/year-alias", year=2024), tmp_path)
    second = _resolve(
        identity_engine,
        _record(arxiv="2401.01234v1", year=2023, source="arxiv"),
        tmp_path,
    )

    assert first == second


def test_contradictory_strong_aliases_create_review_conflict(
    identity_engine, tmp_path: Path
) -> None:
    doi_paper = _resolve(identity_engine, _record(doi="10.5555/one"), tmp_path)
    arxiv_paper = _resolve(
        identity_engine,
        _record(title="A Separate Work", arxiv="2401.01234v1"),
        tmp_path,
    )
    record = _record(doi="10.5555/one", arxiv="2401.01234v1")
    record["_staging_dir"] = str(tmp_path)

    factory = session_factory(identity_engine)
    with factory() as session:
        with pytest.raises(IdentityConflictError, match="contradictory_strong_ids"):
            resolve_paper(record, session)
        session.commit()

    assert doi_paper != arxiv_paper
    factory = session_factory(identity_engine)
    with factory() as session:
        conflict = session.query(IdentityConflict).one()
        assert conflict.reason == "contradictory_strong_ids"
        assert session.query(Paper).count() == 2
        assert (
            session.query(FieldProvenance)
            .filter_by(source_record_id=conflict.source_record_id)
            .count()
            >= 7
        )


def test_strong_id_collision_with_incompatible_metadata_creates_conflict(
    identity_engine, tmp_path: Path
) -> None:
    _resolve(identity_engine, _record(doi="10.5555/collision"), tmp_path)
    record = _record(
        title="A Different Paper",
        doi="10.5555/collision",
        authors=["Alan Turing"],
        year=2025,
    )
    record["_staging_dir"] = str(tmp_path)

    factory = session_factory(identity_engine)
    with factory() as session:
        with pytest.raises(IdentityConflictError, match="incompatible_metadata"):
            resolve_paper(record, session)
        session.commit()

    factory = session_factory(identity_engine)
    with factory() as session:
        assert session.query(Paper).count() == 1
        assert session.query(IdentityConflict).one().reason == "incompatible_metadata"


def test_unicode_title_variants_preserve_each_original_spelling(
    identity_engine, tmp_path: Path
) -> None:
    first_title = "  Ｍodel\u00a0  Cards  "
    second_title = "Model Cards"
    first = _resolve(identity_engine, _record(title=first_title), tmp_path)
    second = _resolve(identity_engine, _record(title=second_title, source="proceedings"), tmp_path)

    assert first == second
    factory = session_factory(identity_engine)
    with factory() as session:
        assert {row.original_title for row in session.query(SourceRecord).all()} == {
            first_title,
            second_title,
        }
        assert session.query(FieldProvenance).filter_by(field_name="title").count() == 2


def test_identical_titles_with_disjoint_authors_remain_separate(
    identity_engine, tmp_path: Path
) -> None:
    first = _resolve(
        identity_engine,
        _record(title="Same Title", authors=["Ada Lovelace"]),
        tmp_path,
    )
    second = _resolve(
        identity_engine,
        _record(title="Same Title", authors=["Marie Curie"], source="proceedings"),
        tmp_path,
    )

    assert first != second
    factory = session_factory(identity_engine)
    with factory() as session:
        assert session.query(Paper).count() == 2


def test_compatible_title_year_and_authors_merge_without_strong_id(
    identity_engine, tmp_path: Path
) -> None:
    first = _resolve(
        identity_engine,
        _record(title="Same Title", authors=["Ada Lovelace"], year=2024),
        tmp_path,
    )
    second = _resolve(
        identity_engine,
        _record(
            title=" Same\u00a0Title ",
            authors=["Ada Lovelace", "Grace Hopper"],
            year=2024,
            source="proceedings",
        ),
        tmp_path,
    )

    assert first == second


def test_concurrent_doi_resolution_converges_without_orphan_papers(
    identity_engine, tmp_path: Path
) -> None:
    barrier = threading.Barrier(2)

    def worker(index: int) -> UUID:
        barrier.wait(timeout=10)
        return _resolve(
            identity_engine,
            _record(doi="10.5555/concurrent", source=f"provider-{index}"),
            tmp_path / str(index),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, [1, 2]))

    assert results[0] == results[1]
    factory = session_factory(identity_engine)
    with factory() as session:
        assert session.query(Paper).count() == 1
        assert session.query(PaperIdentifier).count() == 1


def test_concurrent_replay_of_one_source_record_is_idempotent(
    identity_engine, tmp_path: Path
) -> None:
    barrier = threading.Barrier(2)
    record = _record(doi="10.5555/concurrent-replay")

    def worker(_index: int) -> UUID:
        barrier.wait(timeout=10)
        return _resolve(identity_engine, record, tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, [1, 2]))

    assert results[0] == results[1]
    factory = session_factory(identity_engine)
    with factory() as session:
        assert session.query(Paper).count() == 1
        assert session.query(SourceRecord).count() == 1


def test_manual_merge_preserves_versions_and_records_redirect(
    identity_engine, tmp_path: Path
) -> None:
    survivor = _resolve(identity_engine, _record(doi="10.5555/survivor"), tmp_path)
    losing = _resolve(
        identity_engine,
        _record(title="A Separate Work", arxiv="2401.01234v1", source="arxiv"),
        tmp_path,
    )
    factory = session_factory(identity_engine)
    with factory() as session:
        merge_papers(survivor, losing, session)
        session.commit()

    factory = session_factory(identity_engine)
    with factory() as session:
        redirect = session.query(PaperRedirect).one()
        assert redirect.from_paper_id == losing
        assert redirect.to_paper_id == survivor
        assert session.get(Paper, losing).merged_into == survivor
        assert session.query(PaperVersion).filter_by(paper_id=survivor).count() == 2
        assert session.query(PaperIdentifier).filter_by(paper_id=survivor).count() == 2


def test_raw_artifact_checksum_and_field_provenance_are_stable(
    identity_engine, tmp_path: Path
) -> None:
    record = _record(doi="10.5555/provenance")
    paper_id = _resolve(identity_engine, record, tmp_path)
    factory = session_factory(identity_engine)
    with factory() as session:
        source_record = session.query(SourceRecord).one()
        artifact = Path(source_record.artifact_path)
        payload = artifact.read_bytes()
        assert source_record.paper_id == paper_id
        assert hashlib.sha256(payload).hexdigest() == source_record.content_sha256
        assert json.loads(payload)["title"] == "Reliable Retrieval"
        assert (
            session.query(FieldProvenance)
            .filter_by(source_record_id=source_record.id)
            .count()
            >= 4
        )
        first_checksum = source_record.content_sha256

    _resolve(identity_engine, record, tmp_path)
    factory = session_factory(identity_engine)
    with factory() as session:
        assert session.query(SourceRecord).one().content_sha256 == first_checksum
        assert session.query(FieldProvenance).count() >= 4


def test_malformed_required_record_is_rejected(identity_engine, tmp_path: Path) -> None:
    record = _record()
    record.pop("title")
    record["_staging_dir"] = str(tmp_path)
    factory = session_factory(identity_engine)
    with factory() as session:
        with pytest.raises(ValueError, match="title_required"):
            resolve_paper(record, session)


def test_fixture_loader_replay_is_idempotent(identity_engine, tmp_path: Path) -> None:
    fixture_path = Path(__file__).parents[2] / ".." / "data" / "fixtures" / "metadata.jsonl"
    database_url = os.environ["TEST_DATABASE_URL"]

    first = load_fixtures(fixture_path, database_url=database_url, staging_dir=tmp_path)
    second = load_fixtures(fixture_path, database_url=database_url, staging_dir=tmp_path)

    assert first["loaded"] == second["loaded"]
    factory = session_factory(identity_engine)
    with factory() as session:
        counts = {
            "papers": session.query(Paper).count(),
            "sources": session.query(SourceRecord).count(),
            "versions": session.query(PaperVersion).count(),
        }
    assert counts == {"papers": 2, "sources": 3, "versions": 3}
