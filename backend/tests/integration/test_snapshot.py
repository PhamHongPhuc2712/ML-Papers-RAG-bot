"""Snapshot export, validation, restore and the guarded release switch."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select

from copilot.corpus.export import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    ExportError,
    ExportSchemaError,
    ManifestValidationError,
    export_snapshot,
    manifest_checksum,
    read_shards,
    restore_snapshot,
    validate_manifest,
)
from copilot.corpus.releases import (
    IndexValidation,
    ReleaseNotValidatedError,
    activate_release,
    stage_release,
)
from copilot.db.models import (
    ActiveRelease,
    Author,
    Chunk,
    CorpusRelease,
    Paper,
    PaperAuthor,
    PaperIdentifier,
    PaperVersion,
    Venue,
)
from copilot.db.session import session_scope

pytestmark = pytest.mark.integration


def _seed(engine, *, redistribution: str = "unknown") -> dict[str, int]:
    """Two papers, one with full text, so counts are distinguishable."""

    with session_scope(engine) as session:
        venue = Venue(id=uuid4(), name="ICLR", track="main")
        session.add(venue)
        author = Author(id=uuid4(), name="Ada Lovelace", normalized_name="ada lovelace")
        session.add(author)
        session.flush()

        with_text = Paper(
            id=uuid4(),
            title="A paper with full text",
            normalized_title="a paper with full text",
            abstract="An abstract.",
            publication_year=2024,
            venue_id=venue.id,
            first_seen_at=datetime.now(UTC),
            pdf_url="https://example.invalid/a.pdf",
            acceptance_type="poster",
            metadata_status="active",
        )
        # Missing abstract and missing PDF must be counted separately.
        without = Paper(
            id=uuid4(),
            title="A paper without abstract or pdf",
            normalized_title="a paper without abstract or pdf",
            abstract=None,
            publication_year=2024,
            venue_id=venue.id,
            first_seen_at=datetime.now(UTC),
            pdf_url=None,
            acceptance_type="poster",
            metadata_status="active",
        )
        session.add_all([with_text, without])
        session.flush()
        session.add(PaperAuthor(paper_id=with_text.id, position=0, author_id=author.id))
        session.add(
            PaperIdentifier(
                id=uuid4(), paper_id=with_text.id, namespace="openreview", value="note-1"
            )
        )
        version = PaperVersion(
            id=uuid4(),
            paper_id=with_text.id,
            source="openreview",
            source_url="https://example.invalid/a.pdf",
            source_revision="rev-1",
            content_sha256="a" * 64,
            redistribution=redistribution,
            parser_version="pypdf-text-v2",
            parse_status="parsed",
        )
        session.add(version)
        session.flush()
        session.add(
            Chunk(
                id=uuid4(),
                paper_version_id=version.id,
                section_path="Method",
                section_ordinal=0,
                kind="body",
                page_start=1,
                page_end=1,
                ordinal=0,
                text="the method text",
                token_count=3,
                parser_version="pypdf-text-v2",
                chunker_version="fixed-window-v1",
                evidence_default=True,
            )
        )
    return {"papers": 2, "chunks": 1}


def _clear(engine) -> None:
    with session_scope(engine) as session:
        session.execute(delete(ActiveRelease))
        session.execute(delete(CorpusRelease))
        session.execute(delete(Chunk))
        session.execute(delete(PaperVersion))
        session.execute(delete(PaperIdentifier))
        session.execute(delete(PaperAuthor))
        session.execute(delete(Paper))
        session.execute(delete(Author))
        session.execute(delete(Venue))


@pytest.fixture
def corpus_engine(test_settings, migrated_database):
    _clear(migrated_database)
    yield migrated_database
    _clear(migrated_database)


def test_snapshot_exports_validates_and_counts_missing_fields(corpus_engine, tmp_path):
    _seed(corpus_engine)
    destination = tmp_path / "pilot"

    manifest = export_snapshot("run-1", destination, engine=corpus_engine)

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["counts"]["metadata"] == 2
    assert manifest["counts"]["fulltext"] == 1
    assert (destination / MANIFEST_NAME).is_file()
    assert validate_manifest(destination)["run_id"] == "run-1"

    coverage = manifest["coverage"][0]
    assert coverage["ingested_count"] == 2
    # Separate counts, not one conflated "incomplete" number.
    assert coverage["missing_abstract_count"] == 1
    assert coverage["missing_pdf_url_count"] == 1
    assert coverage["missing_fulltext_count"] == 1
    # No authoritative denominator exists, so percentages stay null, never 100.
    assert coverage["expected_count"] is None
    assert coverage["abstract_coverage_percent"] is None
    assert coverage["fulltext_coverage_percent"] is None


def test_public_export_withholds_text_without_established_rights(corpus_engine, tmp_path):
    _seed(corpus_engine, redistribution="unknown")

    public = export_snapshot("run-2", tmp_path / "public", engine=corpus_engine, public_only=True)

    assert public["counts"]["metadata"] == 0
    assert public["counts"]["fulltext"] == 0
    assert public["withheld_rows"]["metadata"] == 2
    assert public["withheld_rows"]["fulltext"] == 1


def test_public_export_includes_rows_once_rights_are_established(corpus_engine, tmp_path):
    _seed(corpus_engine, redistribution="eligible")

    public = export_snapshot("run-3", tmp_path / "public", engine=corpus_engine, public_only=True)

    assert public["counts"]["fulltext"] == 1
    # The paper without any version has no established rights and stays withheld.
    assert public["counts"]["metadata"] == 1
    assert read_shards(tmp_path / "public", "fulltext")[0]["text"] == "the method text"


def test_checksum_tampering_is_detected(corpus_engine, tmp_path):
    _seed(corpus_engine)
    destination = tmp_path / "pilot"
    export_snapshot("run-4", destination, engine=corpus_engine)

    shard = next(destination.glob("metadata-*.parquet"))
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(ManifestValidationError, match="shard_size_mismatch|shard_checksum"):
        validate_manifest(destination)


def test_manifest_tampering_is_detected(corpus_engine, tmp_path):
    _seed(corpus_engine)
    destination = tmp_path / "pilot"
    export_snapshot("run-5", destination, engine=corpus_engine)

    path = destination / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["counts"]["metadata"] = 999
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="manifest_checksum_mismatch"):
        validate_manifest(destination)


def test_schema_mismatch_fails_before_any_mutation(corpus_engine, tmp_path):
    _seed(corpus_engine)
    destination = tmp_path / "pilot"
    export_snapshot("run-6", destination, engine=corpus_engine)

    path = destination / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = SCHEMA_VERSION + 1
    manifest["manifest_sha256"] = manifest_checksum(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExportSchemaError, match="schema_version_unsupported"):
        validate_manifest(destination)

    # The restore target is untouched because validation runs first.
    _clear(corpus_engine)
    with pytest.raises(ExportSchemaError):
        restore_snapshot(destination, corpus_engine)
    with session_scope(corpus_engine) as session:
        assert session.execute(select(func.count(Paper.id))).scalar_one() == 0


def test_snapshots_are_immutable_and_prior_ones_survive(corpus_engine, tmp_path):
    _seed(corpus_engine)
    destination = tmp_path / "pilot"
    first = export_snapshot("run-7", destination, engine=corpus_engine)

    with pytest.raises(ExportError, match="destination_not_empty"):
        export_snapshot("run-8", destination, engine=corpus_engine)

    # The original snapshot is still intact and still validates.
    assert validate_manifest(destination)["manifest_sha256"] == first["manifest_sha256"]
    second = export_snapshot("run-8", tmp_path / "pilot-2", engine=corpus_engine)
    assert validate_manifest(tmp_path / "pilot-2")["run_id"] == second["run_id"]


def test_restore_into_an_empty_database_reproduces_counts_and_identities(
    corpus_engine, tmp_path
):
    _seed(corpus_engine)
    destination = tmp_path / "pilot"
    export_snapshot("run-9", destination, engine=corpus_engine)

    with session_scope(corpus_engine) as session:
        before_papers = sorted(str(row) for row in session.execute(select(Paper.id)).scalars())
        before_chunks = sorted(str(row) for row in session.execute(select(Chunk.id)).scalars())

    _clear(corpus_engine)
    counts = restore_snapshot(destination, corpus_engine)

    assert counts["papers"] == 2
    assert counts["chunks"] == 1
    assert counts["identifiers"] == 1
    assert counts["versions"] == 1
    with session_scope(corpus_engine) as session:
        after_papers = sorted(str(row) for row in session.execute(select(Paper.id)).scalars())
        after_chunks = sorted(str(row) for row in session.execute(select(Chunk.id)).scalars())
    # Identities survive the round trip, not merely the counts.
    assert after_papers == before_papers
    assert after_chunks == before_chunks


def test_restore_refuses_a_non_empty_database(corpus_engine, tmp_path):
    _seed(corpus_engine)
    destination = tmp_path / "pilot"
    export_snapshot("run-10", destination, engine=corpus_engine)

    with pytest.raises(ExportError, match="restore_target_not_empty"):
        restore_snapshot(destination, corpus_engine)


def test_no_release_becomes_active_before_its_indexes_validate(corpus_engine):
    with session_scope(corpus_engine) as session:
        stage_release(
            session,
            release_id="rel-1",
            manifest_sha256="b" * 64,
            paper_collection="paper_abstracts_rel-1",
            chunk_collection="paper_chunks_rel-1",
            model_revision="bge-m3@pinned",
        )

    # The default validator refuses: P2.2 builds the indexes this would serve.
    with session_scope(corpus_engine) as session:
        with pytest.raises(ReleaseNotValidatedError, match="index_validation_unavailable"):
            activate_release(session, "rel-1")

    with session_scope(corpus_engine) as session:
        assert session.execute(select(ActiveRelease)).scalar_one_or_none() is None
        assert session.get(CorpusRelease, "rel-1").status == "staged"


def test_release_activates_once_indexes_validate_and_keeps_the_prior_row(corpus_engine):
    def validated(release):
        return IndexValidation(True)

    with session_scope(corpus_engine) as session:
        for name in ("rel-1", "rel-2"):
            stage_release(
                session,
                release_id=name,
                manifest_sha256="c" * 64,
                paper_collection=f"paper_abstracts_{name}",
                chunk_collection=f"paper_chunks_{name}",
                model_revision="bge-m3@pinned",
            )

    with session_scope(corpus_engine) as session:
        activate_release(session, "rel-1", validator=validated)
    with session_scope(corpus_engine) as session:
        activate_release(session, "rel-2", validator=validated)

    with session_scope(corpus_engine) as session:
        pointer = session.execute(select(ActiveRelease)).scalar_one()
        assert pointer.corpus_release_id == "rel-2"
        # The prior release row survives for rollback.
        assert session.get(CorpusRelease, "rel-1").status == "superseded"


def test_coverage_endpoint_reports_null_percentages_without_a_denominator(
    api_client, corpus_engine
):
    _seed(corpus_engine)

    response = api_client.get("/v1/corpus/coverage")

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    venue = body["venues"][0]
    assert venue["venue"] == "ICLR"
    assert venue["ingested_count"] == 2
    assert venue["expected_count"] is None
    assert venue["abstract_coverage_percent"] is None


def test_export_refuses_rows_carrying_private_fields(tmp_path):
    from copilot.corpus.export import _write_shards

    with pytest.raises(ExportSchemaError, match="private_field"):
        _write_shards(
            [{"id": "a", "user_id": "u1"}],
            "metadata",
            Path(tmp_path),
            target_shard_bytes=1024,
        )
