"""A release serves requests only after its index pair has been validated."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from copilot.corpus.releases import ReleaseError, activate_release, stage_release
from copilot.db.session import assert_safe_test_database

pytestmark = pytest.mark.integration


def _clear(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("delete from active_release"))
        connection.execute(text("delete from corpus_releases"))


def _stage(engine, release_id: str = "r-2026-09") -> str:
    return stage_release(
        engine,
        release_id,
        paper_collection=f"paper_abstracts_{release_id}",
        chunk_collection=f"paper_chunks_{release_id}",
        model_revision="bge-m3@5617a9f",
        manifest_sha256="ab" * 32,
        counts={"papers": 2, "chunks": 5},
    )


def _active(engine) -> str | None:
    with engine.connect() as connection:
        return connection.execute(
            text("select corpus_release_id from active_release")
        ).scalar_one_or_none()


def test_staging_does_not_change_what_requests_read(migrated_database, test_settings):
    assert_safe_test_database(test_settings)
    _clear(migrated_database)
    _stage(migrated_database)
    assert _active(migrated_database) is None


def test_activation_is_refused_when_the_index_pair_does_not_validate(
    migrated_database, test_settings
):
    assert_safe_test_database(test_settings)
    _clear(migrated_database)
    _stage(migrated_database)
    with pytest.raises(ReleaseError, match="index_validation_failed"):
        activate_release(migrated_database, "r-2026-09", validator=lambda release: False)
    assert _active(migrated_database) is None


def test_activation_switches_the_pointer_once_validation_passes(migrated_database, test_settings):
    assert_safe_test_database(test_settings)
    _clear(migrated_database)
    _stage(migrated_database)
    seen: list[str] = []

    def validator(release: dict) -> bool:
        seen.append(release["paper_collection"])
        return True

    activate_release(migrated_database, "r-2026-09", validator=validator)
    assert seen == ["paper_abstracts_r-2026-09"]
    assert _active(migrated_database) == "r-2026-09"


def test_switching_releases_replaces_the_pointer_rather_than_adding_one(
    migrated_database, test_settings
):
    assert_safe_test_database(test_settings)
    _clear(migrated_database)
    _stage(migrated_database, "r-old")
    _stage(migrated_database, "r-new")
    activate_release(migrated_database, "r-old", validator=lambda release: True)
    activate_release(migrated_database, "r-new", validator=lambda release: True)
    with migrated_database.connect() as connection:
        assert connection.execute(text("select count(*) from active_release")).scalar_one() == 1
    assert _active(migrated_database) == "r-new"


def test_an_unknown_release_cannot_be_activated(migrated_database, test_settings):
    assert_safe_test_database(test_settings)
    _clear(migrated_database)
    with pytest.raises(ReleaseError, match="release_unknown"):
        activate_release(migrated_database, "nope", validator=lambda release: True)


def test_the_serving_release_cannot_be_dropped(migrated_database, test_settings):
    from copilot.corpus.releases import drop_release

    assert_safe_test_database(test_settings)
    _clear(migrated_database)
    _stage(migrated_database, "r-live")
    _stage(migrated_database, "r-spare")
    activate_release(migrated_database, "r-live", validator=lambda release: True)
    with pytest.raises(ReleaseError, match="release_active"):
        drop_release(migrated_database, "r-live")
    assert drop_release(migrated_database, "r-spare").id == "r-spare"
    assert _active(migrated_database) == "r-live"


def test_restaging_a_release_with_another_model_is_refused(migrated_database, test_settings):
    assert_safe_test_database(test_settings)
    _clear(migrated_database)
    _stage(migrated_database)
    with pytest.raises(ReleaseError, match="release_conflict"):
        stage_release(
            migrated_database,
            "r-2026-09",
            paper_collection="paper_abstracts_r-2026-09",
            chunk_collection="paper_chunks_r-2026-09",
            model_revision="another-model",
            manifest_sha256="ab" * 32,
        )
