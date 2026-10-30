"""Re-chunking the stored corpus without the source PDFs."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from copilot.corpus.chunk import (
    ChunkerConfig,
    Section,
    TokenizerSpec,
    chunk_sections,
    whitespace_spans,
)
from copilot.corpus.documents import store_parsed_document
from copilot.corpus.parse import PARSER_VERSION, ParseResult, ParseStatus
from copilot.corpus.rechunk import (
    RechunkStats,
    rechunk_corpus,
    reconstruct_document,
)
from copilot.db.models import PaperVersion
from copilot.db.session import assert_safe_test_database, session_factory

pytestmark = pytest.mark.integration

FIXED = ChunkerConfig(
    policy="fixed-window",
    chunker_version="fixed-window-v2",
    target_tokens=20,
    hard_cap_tokens=30,
    overlap_tokens=5,
    references_in_default_evidence=False,
    uuid_namespace=__import__("uuid").UUID("9d2f5a6c-7b3e-4c1a-8e5d-2f6b9c1d3e47"),
    tokenizer=TokenizerSpec(kind="whitespace"),
)
PACK = ChunkerConfig(
    policy="paragraph-pack",
    chunker_version="paragraph-pack-v1",
    target_tokens=40,
    hard_cap_tokens=60,
    overlap_tokens=5,
    references_in_default_evidence=False,
    uuid_namespace=FIXED.uuid_namespace,
    tokenizer=TokenizerSpec(kind="whitespace"),
    max_tokens=50,
    paragraph_max_tokens=80,
    overlap_sentences=1,
)


def _sections() -> list[Section]:
    body = "\n".join(
        [
            "The first line of the opening paragraph runs the full column width here",
            "and continues at full width for a while longer before it stops.",
            "A second paragraph opens here and also runs the full column width along",
            "and then ends.",
        ]
    )
    return [
        Section("Introduction", body, 1, 2, 0),
        Section("References", "One two three four five six seven eight nine ten.", 9, 9, 1,
                kind="references"),
    ]


def _seed(engine, sections: list[Section]) -> PaperVersion:
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE chunks, paper_versions, papers, venues CASCADE"))
        paper = connection.execute(
            text(
                "insert into papers (id, title, publication_year, first_seen_at,"
                " acceptance_type, metadata_status) values (gen_random_uuid(), 'A paper', 2024,"
                " now(), 'poster', 'active') returning id"
            )
        ).scalar_one()
        version_id = connection.execute(
            text(
                "insert into paper_versions (id, paper_id, source, source_url, source_revision,"
                " content_sha256, redistribution, parser_version, parse_status) values"
                " (gen_random_uuid(), :p, 'papercli', 'u', 'rev', :sha, 'unknown', :pv, 'parsed')"
                " returning id"
            ),
            {"p": paper, "sha": "ab" * 32, "pv": PARSER_VERSION},
        ).scalar_one()

    factory = session_factory(engine)
    with factory() as session:
        with session.begin():
            version = session.get(PaperVersion, version_id)
            assert version is not None
            result = ParseResult(
                status=ParseStatus.PARSED,
                error_code=None,
                sections=tuple(sections),
                content_sha256=version.content_sha256,
                parser_version=version.parser_version,
                page_count=2,
                quality="low",
            )
            chunks = chunk_sections(
                sections,
                whitespace_spans,
                target=FIXED.target_tokens,
                overlap=FIXED.overlap_tokens,
                hard_cap=FIXED.hard_cap_tokens,
            )
            store_parsed_document(
                session, version, result, chunks,
                chunker_version=FIXED.chunker_version, namespace=FIXED.uuid_namespace,
            )
    return version_id


def test_sections_are_reconstructed_from_their_own_chunks(migrated_database, test_settings):
    """The round trip that licenses re-chunking without the PDF."""

    assert_safe_test_database(test_settings)
    sections = _sections()
    version_id = _seed(migrated_database, sections)
    factory = session_factory(migrated_database)
    stats = RechunkStats()
    with factory() as session:
        rows = session.execute(
            text(
                "select section_ordinal, ordinal, text, fragment, kind, page_start, page_end,"
                " section_path from chunks where paper_version_id = :v order by ordinal"
            ),
            {"v": version_id},
        ).all()
        rebuilt = reconstruct_document(rows, whitespace_spans, FIXED.overlap_tokens, stats)
    assert [s.name for s in rebuilt] == ["Introduction", "References"]
    for original, restored in zip(sections, rebuilt, strict=True):
        assert restored.text == original.text
        assert (restored.page_start, restored.page_end, restored.kind) == (
            original.page_start,
            original.page_end,
            original.kind,
        )


def test_rechunking_replaces_every_chunk_under_the_new_policy(migrated_database, test_settings):
    assert_safe_test_database(test_settings)
    _seed(migrated_database, _sections())
    factory = session_factory(migrated_database)
    with factory() as session:
        before = session.execute(
            text("select count(*), min(chunker_version) from chunks")
        ).one()

    stats = rechunk_corpus(
        migrated_database,
        whitespace_spans,
        PACK,
        source_overlap_tokens=FIXED.overlap_tokens,
        session_factory=factory,
    )

    with factory() as session:
        rows = session.execute(
            text("select chunker_version, count(*) from chunks group by chunker_version")
        ).all()
        ordinals = session.execute(
            text("select ordinal from chunks order by ordinal")
        ).scalars().all()
    assert before[1] == "fixed-window-v2"
    # Nothing from the old revision survives, and ordinals stay contiguous.
    assert rows == [("paragraph-pack-v1", len(ordinals))]
    assert ordinals == list(range(len(ordinals)))
    assert stats.versions == 1
    assert stats.chunks_before == before[0]
    assert stats.chunks_after == len(ordinals)


def test_rechunking_is_resumable_and_idempotent(migrated_database, test_settings):
    assert_safe_test_database(test_settings)
    _seed(migrated_database, _sections())
    factory = session_factory(migrated_database)
    first = rechunk_corpus(
        migrated_database, whitespace_spans, PACK,
        source_overlap_tokens=FIXED.overlap_tokens, session_factory=factory,
    )
    second = rechunk_corpus(
        migrated_database, whitespace_spans, PACK,
        source_overlap_tokens=FIXED.overlap_tokens, session_factory=factory,
    )
    # A second pass finds nothing left on the old revision.
    assert first.versions == 1
    assert second.versions == 0
    with factory() as session:
        stored = session.execute(text("select count(*) from chunks")).scalar_one()
    assert stored == first.chunks_after
