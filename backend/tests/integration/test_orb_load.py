"""Loading Open RAG Bench papers through the corpus pipeline (ORB plan, O2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select, text

from copilot.corpus.chunk import TokenizerSpec, load_parsing_config, tokenizer_cache_path
from copilot.db.models import Chunk, PaperIdentifier, PaperVersion
from copilot.db.session import assert_safe_test_database, session_scope
from copilot.evaluation.orb import OrbPaths, paper_ids_for
from copilot.evaluation.orb_load import load_orb_corpus, orb_record

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PDF = (ROOT / "data" / "fixtures" / "papers" / "fixture.pdf").read_bytes()
FIXTURE_TOKENIZER = ROOT / "data" / "fixtures" / "tokenizer" / "tokenizer.json"
TABLES = (
    "chunks",
    "jobs",
    "source_checkpoints",
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
PUBLIC_IP = "93.184.216.34"
DOCS = {
    "2401.00001v2": {
        "id": "2401.00001v2",
        "title": "Graph neural networks\n  for molecules",
        "abstract": "Message passing on molecular graphs.",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "categories": ["cs.LG"],
        "published": "2024-01-02T00:00:00Z",
        "updated": "2024-02-02T00:00:00Z",
        "sections": [{"section_id": 0, "text": "#### Abstract\n\nMessage passing."}],
    },
    "2401.00002v1": {
        "id": "2401.00002v1",
        "title": "Diffusion models for images",
        "abstract": "Denoising generates images.",
        "authors": ["Grace Hopper"],
        "categories": ["cs.CV"],
        "published": "2024-01-03T00:00:00Z",
        "updated": "2024-01-03T00:00:00Z",
        "sections": [],
    },
}


@dataclass
class FakeStream:
    status_code: int
    headers: dict[str, str]
    body: bytes

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def close(self) -> None:
        return None


@dataclass
class FakeTransport:
    routes: dict[str, tuple[int, dict[str, str], bytes]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    # URL to how many times its stream read should time out before succeeding.
    stalls: dict[str, int] = field(default_factory=dict)

    def stream(self, url, *, headers=None, timeout=None) -> FakeStream:
        self.calls.append(url)
        if self.stalls.get(url, 0) > 0:
            self.stalls[url] -= 1
            raise httpx.ReadTimeout("The read operation timed out")
        status, response_headers, body = self.routes[url]
        return FakeStream(status, response_headers, body)


def _parsing_config(data_dir: Path):
    payload = FIXTURE_TOKENIZER.read_bytes()
    spec = TokenizerSpec(
        kind="model",
        repo="fixtures/tiny-bpe",
        revision="v1",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    target = tokenizer_cache_path(spec, data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    config = load_parsing_config(ROOT / "configs" / "parsing.yaml")
    return replace(config, chunker=replace(config.chunker, tokenizer=spec))


def _benchmark(root: Path) -> OrbPaths:
    labels = root / "pdf" / "arxiv"
    (labels / "corpus").mkdir(parents=True)
    (labels / "pdf_urls.json").write_text(
        json.dumps({doc: f"https://arxiv.test/pdf/{doc}" for doc in DOCS})
    )
    for doc, paper in DOCS.items():
        (labels / "corpus" / f"{doc}.json").write_text(json.dumps(paper))
    return OrbPaths(root)


@pytest.fixture
def clean_database(migrated_database, test_settings):
    assert_safe_test_database(test_settings)
    with migrated_database.begin() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
    yield migrated_database
    with migrated_database.begin() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))


def test_a_record_carries_the_versioned_arxiv_id_and_no_venue():
    record = orb_record(DOCS["2401.00001v2"], "https://arxiv.test/pdf/2401.00001v2")
    assert record["arxiv"] == "2401.00001v2"
    assert record["title"] == "Graph neural networks for molecules"
    assert record["year"] == 2024 and record["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert record["source"] == "orb" and "venue" not in record


def test_two_orb_papers_load_with_arxiv_identifiers_and_chunks(clean_database, tmp_path):
    paths = _benchmark(tmp_path)
    transport = FakeTransport()
    for doc in DOCS:
        transport.routes[f"https://arxiv.test/pdf/{doc}"] = (
            200,
            {"content-type": "application/pdf"},
            FIXTURE_PDF,
        )
    sleeps: list[float] = []
    kwargs = {
        "engine": clean_database,
        "parsing_config": _parsing_config(tmp_path),
        "data_dir": tmp_path,
        "transport": transport,
        "allowed_hosts": frozenset({"arxiv.test"}),
        "resolver": lambda host: [PUBLIC_IP],
        "sleep": sleeps.append,
    }
    report = load_orb_corpus(paths, limit=None, **kwargs)
    assert report.papers == 2 and report.parsed == 2 and report.failed == {}
    assert report.chunks > 0 and report.chunk_tokens_max is not None
    assert sleeps == [1.0, 1.0], "one polite pause per download"
    with session_scope(clean_database) as session:
        ids = session.execute(
            select(PaperIdentifier.value, PaperIdentifier.version).where(
                PaperIdentifier.namespace == "arxiv"
            )
        ).all()
        # The alias is the versionless stem; the version lives on the paper version.
        assert sorted(value for value, _ in ids) == ["2401.00001", "2401.00002"]
        assert session.execute(select(func.count(Chunk.id))).scalar_one() == report.chunks
        versions = session.execute(select(PaperVersion)).scalars().all()
        assert sorted(v.version for v in versions) == ["v1", "v2"]
        assert {v.parse_status for v in versions} == {"parsed"}
        assert {v.source for v in versions} == {"orb"}
        assert all(v.content_sha256 == hashlib.sha256(FIXTURE_PDF).hexdigest() for v in versions)
    # Every PDF is kept, named by its checksum.
    assert sorted(p.name for p in paths.pdfs.iterdir()) == [
        hashlib.sha256(FIXTURE_PDF).hexdigest() + ".pdf"
    ]
    # The gold documents resolve by id, versioned as ORB writes them.
    mapped = paper_ids_for(DOCS, clean_database)
    assert set(mapped) == set(DOCS)
    assert len(set(mapped.values())) == 2
    # A second run downloads nothing and skips both.
    again = load_orb_corpus(paths, limit=None, **kwargs)
    assert (again.papers, again.parsed, again.skipped) == (2, 0, 2)
    assert len(transport.calls) == 2


def test_a_failed_download_is_recorded_and_the_loop_continues(clean_database, tmp_path):
    paths = _benchmark(tmp_path)
    transport = FakeTransport()
    transport.routes["https://arxiv.test/pdf/2401.00001v2"] = (404, {}, b"")
    transport.routes["https://arxiv.test/pdf/2401.00002v1"] = (
        200,
        {"content-type": "application/pdf"},
        FIXTURE_PDF,
    )
    report = load_orb_corpus(
        paths,
        engine=clean_database,
        parsing_config=_parsing_config(tmp_path),
        data_dir=tmp_path,
        transport=transport,
        allowed_hosts=frozenset({"arxiv.test"}),
        resolver=lambda host: [PUBLIC_IP],
        sleep=lambda _: None,
    )
    assert (report.papers, report.parsed) == (2, 1)
    assert report.failed == {"download:download_http": 1}
    assert report.failures == {"2401.00001v2": "download:download_http"}
    with session_scope(clean_database) as session:
        statuses = dict(
            session.execute(select(PaperVersion.version, PaperVersion.parse_status)).all()
        )
    assert statuses == {"v2": "pending", "v1": "parsed"}


def test_a_host_outside_the_policy_is_refused_not_fetched(clean_database, tmp_path):
    paths = _benchmark(tmp_path)
    transport = FakeTransport()
    report = load_orb_corpus(
        paths,
        engine=clean_database,
        parsing_config=_parsing_config(tmp_path),
        data_dir=tmp_path,
        transport=transport,
        allowed_hosts=frozenset({"arxiv.org"}),
        resolver=lambda host: [PUBLIC_IP],
        sleep=lambda _: None,
        limit=1,
    )
    assert report.failed == {"download_policy:host": 1} and transport.calls == []


def test_a_stalled_download_is_retried_then_recorded_not_fatal(clean_database, tmp_path):
    """A read timeout from the publisher killed the first real run at paper 436."""

    paths = _benchmark(tmp_path)
    transport = FakeTransport()
    for doc in DOCS:
        transport.routes[f"https://arxiv.test/pdf/{doc}"] = (
            200,
            {"content-type": "application/pdf"},
            FIXTURE_PDF,
        )
    # The first paper stalls once and recovers; the second stalls past the retries.
    transport.stalls["https://arxiv.test/pdf/2401.00001v2"] = 1
    transport.stalls["https://arxiv.test/pdf/2401.00002v1"] = 5
    sleeps: list[float] = []
    report = load_orb_corpus(
        paths,
        engine=clean_database,
        parsing_config=_parsing_config(tmp_path),
        data_dir=tmp_path,
        transport=transport,
        allowed_hosts=frozenset({"arxiv.test"}),
        resolver=lambda host: [PUBLIC_IP],
        sleep=sleeps.append,
    )
    assert (report.papers, report.parsed) == (2, 1)
    assert report.failed == {"download:download_transport": 1}
    assert report.failures == {"2401.00002v1": "download:download_transport"}
    # One backoff for the recovered paper, two for the one given up on, one polite pause.
    assert sleeps.count(15.0) == 3 and sleeps.count(1.0) == 1


def test_a_reparse_reads_the_stored_pdfs_and_replaces_the_chunks(clean_database, tmp_path):
    """Re-chunking the loaded corpus under another policy never touches the network."""

    paths = _benchmark(tmp_path)
    transport = FakeTransport()
    for doc in DOCS:
        transport.routes[f"https://arxiv.test/pdf/{doc}"] = (
            200,
            {"content-type": "application/pdf"},
            FIXTURE_PDF,
        )
    config = _parsing_config(tmp_path)
    kwargs = {
        "engine": clean_database,
        "data_dir": tmp_path,
        "transport": transport,
        "allowed_hosts": frozenset({"arxiv.test"}),
        "resolver": lambda host: [PUBLIC_IP],
        "sleep": lambda seconds: None,
    }
    first = load_orb_corpus(paths, parsing_config=config, **kwargs)
    assert first.parsed == 2 and len(transport.calls) == 2
    with session_scope(clean_database) as session:
        before = set(session.execute(select(Chunk.id)).scalars().all())
        versions = {v.id: v.content_sha256 for v in session.execute(select(PaperVersion)).scalars()}

    smaller = replace(
        config,
        chunker=replace(
            config.chunker,
            chunker_version="paragraph-pack-test-v2",
            target_tokens=32,
            max_tokens=32,
            paragraph_max_tokens=32,
            overlap_sentences=0,
        ),
    )
    # Without --reparse a parsed version is skipped, whatever the policy says.
    skipped = load_orb_corpus(paths, parsing_config=smaller, **kwargs)
    assert (skipped.parsed, skipped.skipped) == (0, 2)

    reparsed = load_orb_corpus(paths, parsing_config=smaller, reparse=True, **kwargs)
    assert reparsed.parsed == 2 and reparsed.skipped == 0 and reparsed.failed == {}
    assert len(transport.calls) == 2, "a reparse downloads nothing"
    with session_scope(clean_database) as session:
        rows = session.execute(select(Chunk.id, Chunk.chunker_version, Chunk.token_count)).all()
        assert rows and {row.chunker_version for row in rows} == {"paragraph-pack-test-v2"}
        assert not before & {row.id for row in rows}, "every chunk identity changed"
        assert max(row.token_count for row in rows) <= 32
        # The versions keep their identity and checksum; only their chunking moved.
        after = {v.id: v.content_sha256 for v in session.execute(select(PaperVersion)).scalars()}
        assert after == versions
    # A second reparse under the same policy is a no-op.
    again = load_orb_corpus(paths, parsing_config=smaller, reparse=True, **kwargs)
    assert (again.parsed, again.skipped) == (0, 2)

    # A version whose PDF is gone from disk is recorded, not fetched.
    for pdf in paths.pdfs.iterdir():
        pdf.unlink()
    third = replace(
        smaller, chunker=replace(smaller.chunker, chunker_version="paragraph-pack-test-v3")
    )
    missing = load_orb_corpus(paths, parsing_config=third, reparse=True, **kwargs)
    assert missing.failed == {"reparse:pdf_missing": 2} and len(transport.calls) == 2
