"""Parser acceptance tests against repository-owned fixture PDFs.

Only the persistence test needs services; the rest run offline in CI.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from pypdf import PdfWriter

from copilot.corpus.chunk import (
    chunk_id,
    chunk_sections,
    load_parsing_config,
    whitespace_detokenize,
    whitespace_tokenize,
)
from copilot.corpus.parse import (
    PARSER_VERSION,
    ParsedPage,
    ParseErrorCode,
    ParseStatus,
    PDFParseError,
    parse_pdf,
    parse_pdf_result,
    sections_from_pages,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PDF = ROOT / "data" / "fixtures" / "papers" / "fixture.pdf"
CONFIG = ROOT / "configs" / "parsing.yaml"


def _write_pdf(path: Path, *, encrypt: str | None = None) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    if encrypt is not None:
        writer.encrypt(encrypt)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def test_fixture_headings_pages_and_kinds_survive():
    sections = parse_pdf(FIXTURE_PDF)
    by_name = {section.name: section for section in sections}
    assert {"Abstract", "Introduction", "Method", "Results", "References"} <= set(by_name)
    assert by_name["Abstract"].kind == "abstract"
    assert (by_name["Introduction"].page_start, by_name["Introduction"].page_end) == (1, 1)
    assert (by_name["Method"].page_start, by_name["Method"].page_end) == (2, 2)
    assert by_name["References"].kind == "references"
    assert by_name["References"].page_start == 3
    assert "cite evidence that exists" in by_name["Introduction"].text
    assert [section.ordinal for section in sections] == list(range(len(sections)))

    tables = [section for section in sections if section.kind == "table"]
    assert len(tables) == 1
    assert tables[0].name.startswith("Table 1")
    assert "target 450" in tables[0].text


def test_parse_result_is_deterministic_and_records_provenance():
    first = parse_pdf_result(FIXTURE_PDF)
    second = parse_pdf_result(FIXTURE_PDF)
    assert first.status is ParseStatus.PARSED
    assert first.error_code is None
    assert first.parser_version == PARSER_VERSION
    assert len(first.content_sha256) == 64
    assert first.page_count == 3
    assert first.quality == "low"
    assert first.sections == second.sections
    assert first.content_sha256 == second.content_sha256


def test_unknown_page_positions_remain_null():
    pages = [ParsedPage(number=None, text="Abstract\nWe study things.\n1 Introduction\nBody.")]
    sections = sections_from_pages(pages)
    assert [section.name for section in sections] == ["Abstract", "Introduction"]
    assert all(section.page_start is None and section.page_end is None for section in sections)


def test_encrypted_pdf_is_typed(tmp_path):
    path = _write_pdf(tmp_path / "encrypted.pdf", encrypt="secret")
    result = parse_pdf_result(path)
    assert result.status is ParseStatus.FAILED
    assert result.error_code is ParseErrorCode.ENCRYPTED
    assert result.sections == ()
    with pytest.raises(PDFParseError) as excinfo:
        parse_pdf(path)
    assert excinfo.value.code is ParseErrorCode.ENCRYPTED


def test_corrupt_non_pdf_and_missing_inputs_are_typed(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.7\n1 0 obj << /Type /Catalog >> garbage")
    assert parse_pdf_result(corrupt).error_code is ParseErrorCode.CORRUPT

    not_pdf = tmp_path / "notes.pdf"
    not_pdf.write_bytes(b"hello, not a pdf at all")
    assert parse_pdf_result(not_pdf).error_code is ParseErrorCode.NOT_PDF

    assert parse_pdf_result(tmp_path / "absent.pdf").error_code is ParseErrorCode.MISSING


def test_oversized_pdf_is_rejected_before_parsing():
    result = parse_pdf_result(FIXTURE_PDF, max_bytes=16)
    assert result.status is ParseStatus.FAILED
    assert result.error_code is ParseErrorCode.OVERSIZED


def test_pdf_without_extractable_text_is_a_typed_failure(tmp_path):
    path = _write_pdf(tmp_path / "blank.pdf")
    result = parse_pdf_result(path)
    assert result.status is ParseStatus.FAILED
    assert result.error_code is ParseErrorCode.EMPTY_TEXT


def test_fixture_chunks_under_the_versioned_policy_exclude_references_by_default():
    config = load_parsing_config(CONFIG)
    sections = parse_pdf(FIXTURE_PDF)
    chunks = chunk_sections(
        sections,
        whitespace_tokenize,
        whitespace_detokenize,
        target=config.chunker.target_tokens,
        overlap=config.chunker.overlap_tokens,
        hard_cap=config.chunker.hard_cap_tokens,
    )
    assert chunks
    assert all(chunk["token_count"] <= config.chunker.hard_cap_tokens for chunk in chunks)
    references = [chunk for chunk in chunks if chunk["kind"] == "references"]
    assert references and not any(chunk["evidence_default"] for chunk in references)
    assert any(chunk["kind"] == "table" for chunk in chunks)
    assert config.parser.parser_version == PARSER_VERSION
    assert config.chunker.chunker_version == "fixed-window-v1"
    assert isinstance(config.chunker.uuid_namespace, UUID)


@pytest.mark.integration
def test_parse_results_persist_with_stable_chunk_identity(migrated_database, tmp_path):
    from sqlalchemy import select

    from copilot.corpus.dedupe import resolve_paper
    from copilot.corpus.documents import store_parsed_document
    from copilot.db.models import Chunk, PaperVersion
    from copilot.db.session import session_factory

    config = load_parsing_config(CONFIG)
    result = parse_pdf_result(FIXTURE_PDF)
    record = {
        "source": "fixture",
        "external_ids": {"doi": "10.5555/parser-persist"},
        "title": "Reliable Retrieval for Research Copilots",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "venue": {"name": "ICLR", "track": "main"},
        "year": 2024,
        "source_revision": "revision-1",
        "retrieved_at": "2024-06-01T12:00:00+00:00",
        "acceptance_decision": "accepted",
        "source_url": "https://papers.example.test/fixture.pdf",
    }
    with session_factory(migrated_database)() as session:
        paper_id = resolve_paper(record, session, staging_dir=tmp_path)
        session.flush()
        version = session.execute(
            select(PaperVersion).where(PaperVersion.paper_id == paper_id)
        ).scalar_one()

        def persist(chunker_version: str) -> list[UUID]:
            chunks = chunk_sections(
                result.sections,
                whitespace_tokenize,
                whitespace_detokenize,
                target=config.chunker.target_tokens,
                overlap=config.chunker.overlap_tokens,
                hard_cap=config.chunker.hard_cap_tokens,
            )
            return store_parsed_document(
                session,
                version,
                result,
                chunks,
                chunker_version=chunker_version,
                namespace=config.chunker.uuid_namespace,
            )

        first = persist("fixed-window-v1")
        again = persist("fixed-window-v1")
        assert first == again
        assert version.parser_version == PARSER_VERSION
        assert version.parse_status == "parsed"
        assert version.content_sha256 == result.content_sha256
        stored = (
            session.execute(select(Chunk).where(Chunk.paper_version_id == version.id))
            .scalars()
            .all()
        )
        assert sorted(chunk.id for chunk in stored) == sorted(first)
        assert all(chunk.chunker_version == "fixed-window-v1" for chunk in stored)
        assert first[0] == chunk_id(
            paper_id, result.content_sha256, PARSER_VERSION, "fixed-window-v1", 0, 0
        )

        changed = persist("fixed-window-v2")
        assert set(changed).isdisjoint(first)
        session.rollback()


def test_control_characters_from_font_encodings_are_stripped():
    """Real ICLR PDFs yield C0 control codes that PostgreSQL text columns reject.

    pypdf maps LaTeX ligature and math glyphs onto low codepoints when a font
    carries a custom encoding. NUL in particular makes psycopg raise DataError,
    so a document that parses cleanly still fails to persist. Sanitizing at the
    Section boundary covers every PageAdapter, not only this one.
    """

    # Not vertical tab or form feed: str.splitlines already treats those as breaks.
    nul, soh, stx, dle, delete = chr(0), chr(1), chr(2), chr(16), chr(127)
    pages = [
        ParsedPage(number=1, text=f"Method{chr(10)}identi{soh}cation of{nul} e{stx}ects\tkept"),
        ParsedPage(number=2, text=f"Results{chr(10)}accuracy {dle}improved{delete}"),
    ]

    sections = sections_from_pages(pages)

    body = "\n".join(section.text for section in sections)
    assert not [character for character in body if character in {nul, soh, stx, dle, delete}]
    # Stripping rejoins the word rather than splitting it, but the glyph's own
    # characters are gone: a "ffi"/"ff" ligature arrives as "identication"/"eects".
    # That loss is real and is what the 20-paper audit has to weigh.
    assert "identication of eects\tkept" in body
    assert "accuracy improved" in body
    assert "\t" in body, "legitimate whitespace must survive"
