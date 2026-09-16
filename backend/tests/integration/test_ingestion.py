"""Ingestion pipeline acceptance tests with synthetic provider responses.

Provider transports and DNS resolution are injected; the only real services
are the isolated PostgreSQL and Qdrant containers.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlencode

import pytest
import yaml
from sqlalchemy import func, select, text

from copilot.corpus.chunk import (
    TokenizerSpec,
    load_parsing_config,
    tokenizer_cache_path,
)
from copilot.corpus.ingest import IngestContext, build_handlers, ingest
from copilot.corpus.parse import PARSER_VERSION
from copilot.db.models import Chunk, Job, Paper, PaperIdentifier, PaperVersion, SourceCheckpoint
from copilot.db.session import assert_safe_test_database, session_factory
from copilot.jobs.queue import complete, enqueue, heartbeat, lease
from copilot.jobs.worker import JobOutcome, Worker

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
PROVIDERS = Path(__file__).resolve().parents[1] / "fixtures" / "providers"
FIXTURE_PDF = (ROOT / "data" / "fixtures" / "papers" / "fixture.pdf").read_bytes()
FIXTURE_SHA = hashlib.sha256(FIXTURE_PDF).hexdigest()
PUBLIC_IP = "93.184.216.34"
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


@dataclass
class FakeResponse:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> object:
        return json.loads(self.body)


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


FIXTURE_TOKENIZER = ROOT / "data" / "fixtures" / "tokenizer" / "tokenizer.json"


def parsing_config_for(data_dir: Path) -> object:
    """Load the real policy but pin it to the committed fixture tokenizer.

    The suite exercises the same pinned-artifact path as production — cache
    location, checksum, ``Tokenizer.from_file`` — without shipping BGE-M3's
    17 MB file into CI.
    """

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


class FakeTransport:
    """Scripted responses keyed by URL and query; the last response repeats."""

    def __init__(self) -> None:
        self.routes: dict[str, list[FakeResponse]] = {}
        self.calls: list[str] = []
        self.headers: list[dict[str, str]] = []

    @staticmethod
    def key(url: str, params: dict[str, object] | None = None) -> str:
        if not params:
            return url
        return f"{url}?{urlencode(sorted((k, str(v)) for k, v in params.items()))}"

    def add(
        self, url: str, *responses: FakeResponse, params: dict[str, object] | None = None
    ) -> None:
        self.routes.setdefault(self.key(url, params), []).extend(responses)

    def replace(self, url: str, *responses: FakeResponse) -> None:
        self.routes[self.key(url)] = list(responses)

    def _next(self, key: str) -> FakeResponse:
        queue = self.routes.get(key)
        if not queue:
            raise AssertionError(f"unexpected request: {key}")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def get(self, url, *, params=None, headers=None, timeout=None) -> FakeResponse:
        key = self.key(url, dict(params or {}))
        self.calls.append(key)
        self.headers.append(dict(headers or {}))
        return self._next(key)

    def post(self, url, *, json=None, headers=None, timeout=None) -> FakeResponse:
        self.calls.append(f"POST {url}")
        self.headers.append(dict(headers or {}))
        return self._next(self.key(url))

    def stream(self, url, *, headers=None, timeout=None) -> FakeStream:
        self.calls.append(url)
        response = self._next(self.key(url))
        return FakeStream(response.status_code, response.headers, response.body)


def _json(name: str) -> FakeResponse:
    return FakeResponse(200, {"content-type": "application/json"}, (PROVIDERS / name).read_bytes())


def _pdf(body: bytes = FIXTURE_PDF) -> FakeResponse:
    return FakeResponse(200, {"content-type": "application/pdf"}, body)


def _manifest(tmp_path: Path, extra_sources: dict[str, object] | None = None) -> Path:
    data = {
        "schema_version": 1,
        "venue": "ICLR",
        "years": [2024],
        "track": "main",
        "sample": {"limit": 100, "order": "source_item_id"},
        "sources": {
            "openreview": {
                "api_base": "https://api.openreview.test",
                "venue_id": "ICLR.cc/2024/Conference",
                "page_size": 2,
                "pdf_base": "https://openreview.test/pdf?id=",
                "forum_base": "https://openreview.test/forum?id=",
            },
            **(extra_sources or {}),
        },
        "download": {
            "allowed_hosts": ["openreview.test", "cdn.openreview.test", "internal.openreview.test"],
            "max_bytes": 1048576,
        },
    }
    path = tmp_path / "corpus.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@dataclass
class Pipeline:
    engine: object
    transport: FakeTransport
    manifest: Path
    context: IngestContext
    handlers: dict[str, object]
    worker: Worker

    def ingest(self, limit: int | None = None) -> str:
        return ingest(self.manifest, limit, engine=self.engine, transport=self.transport)

    def session(self):
        return session_factory(self.engine)()

    def release_due_jobs(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                text("UPDATE jobs SET next_attempt_at = now() WHERE status = 'retry_wait'")
            )


def _truncate(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))


@pytest.fixture
def pipeline(migrated_database, test_settings, tmp_path) -> Iterator[Pipeline]:
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    transport = FakeTransport()
    listing = "https://api.openreview.test/notes"
    common = {"content.venueid": "ICLR.cc/2024/Conference", "limit": 2}
    transport.add(listing, _json("openreview_notes_offset_0.json"), params={**common, "offset": 0})
    transport.add(listing, _json("openreview_notes_offset_2.json"), params={**common, "offset": 2})
    transport.add(
        "https://openreview.test/pdf?id=note-a",
        FakeResponse(302, {"location": "https://cdn.openreview.test/a.pdf"}),
    )
    transport.add("https://cdn.openreview.test/a.pdf", _pdf())
    transport.add("https://openreview.test/pdf?id=note-b", _pdf())

    manifest = _manifest(tmp_path)
    context = IngestContext(
        manifest=yaml.safe_load(manifest.read_text(encoding="utf-8")),
        transport=transport,
        staging_dir=tmp_path / "sources",
        data_dir=tmp_path,
        parsing_config=parsing_config_for(tmp_path),
        resolver=lambda host: [PUBLIC_IP],
    )
    handlers = build_handlers(context)
    worker = Worker(migrated_database, handlers, lease_seconds=60, heartbeat_seconds=15)
    yield Pipeline(migrated_database, transport, manifest, context, handlers, worker)
    _truncate(migrated_database)


def test_pilot_ingests_eligible_papers_through_durable_jobs(pipeline):
    run_id = pipeline.ingest(limit=2)
    assert run_id
    processed = pipeline.worker.run_until_idle("worker-1")
    assert processed == 6  # 2 resolve + 2 download + 2 parse

    with pipeline.session() as session:
        titles = session.execute(select(Paper.title).order_by(Paper.title)).scalars().all()
        assert titles == ["Accepted Paper A", "Accepted Paper B"]
        versions = session.execute(select(PaperVersion)).scalars().all()
        assert {version.parse_status for version in versions} == {"parsed"}
        assert {version.content_sha256 for version in versions} == {FIXTURE_SHA}
        assert {version.parser_version for version in versions} == {PARSER_VERSION}
        assert session.execute(select(func.count()).select_from(Chunk)).scalar_one() >= 2
        jobs = session.execute(select(Job)).scalars().all()
        assert {job.status for job in jobs} == {"succeeded"}
        assert Counter(job.kind for job in jobs) == {
            "resolve_record": 2,
            "download_pdf": 2,
            "parse_pdf": 2,
        }
        checkpoint = session.execute(select(SourceCheckpoint)).scalar_one()
        assert (checkpoint.source, checkpoint.partition) == ("openreview", "ICLR:2024:main")
        assert checkpoint.run_id == run_id

    assert "https://cdn.openreview.test/a.pdf" in pipeline.transport.calls
    stored = sorted((pipeline.context.staging_dir / "pdfs").glob("*.pdf"))
    assert [path.name for path in stored] == [f"{FIXTURE_SHA}.pdf"]


def test_repeated_ingestion_creates_no_duplicates(pipeline):
    first_run = pipeline.ingest(limit=2)
    pipeline.worker.run_until_idle("worker-1")
    with pipeline.session() as session:
        paper_ids = set(session.execute(select(Paper.id)).scalars().all())
        version_count = session.execute(select(func.count()).select_from(PaperVersion)).scalar_one()
        chunk_count = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
        job_count = session.execute(select(func.count()).select_from(Job)).scalar_one()

    second_run = pipeline.ingest(limit=2)
    assert second_run != first_run
    assert pipeline.worker.run_until_idle("worker-2") == 0

    with pipeline.session() as session:
        assert set(session.execute(select(Paper.id)).scalars().all()) == paper_ids
        assert (
            session.execute(select(func.count()).select_from(PaperVersion)).scalar_one()
            == version_count
        )
        assert session.execute(select(func.count()).select_from(Chunk)).scalar_one() == chunk_count
        assert session.execute(select(func.count()).select_from(Job)).scalar_one() == job_count
        checkpoint = session.execute(select(SourceCheckpoint)).scalar_one()
        assert checkpoint.run_id == second_run


def test_lease_is_exclusive_and_completion_requires_the_live_token(
    migrated_database, test_settings
):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    factory = session_factory(migrated_database)
    with factory() as session:
        job_id = enqueue(session, "noop", {}, "noop:1")
        assert enqueue(session, "noop", {"ignored": True}, "noop:1") == job_id
        session.commit()

    with factory() as first_session, factory() as second_session:
        first = lease(first_session, "worker-1")
        first_session.commit()
        assert first is not None and first.id == job_id and first.attempt == 1
        assert lease(second_session, "worker-2") is None
        second_session.commit()

    with factory() as session:
        assert heartbeat(session, job_id, first.token) is True
        assert complete(session, job_id, first.token, {"ok": True}) is True
        session.commit()
    with factory() as session:
        assert complete(session, job_id, first.token) is False
        assert heartbeat(session, job_id, first.token) is False
        job = session.get(Job, job_id)
        assert job is not None and job.status == "succeeded" and job.result == {"ok": True}


def test_expired_lease_cannot_overwrite_the_new_owner(migrated_database, test_settings):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    factory = session_factory(migrated_database)
    with factory() as session:
        job_id = enqueue(session, "noop", {}, "noop:expired")
        session.commit()
    with factory() as session:
        stale = lease(session, "worker-1")
        session.commit()
    assert stale is not None
    with migrated_database.begin() as connection:
        connection.execute(
            text("UPDATE jobs SET lease_until = now() - interval '1 second' WHERE id = :id"),
            {"id": job_id},
        )
    with factory() as session:
        fresh = lease(session, "worker-2")
        session.commit()
    assert fresh is not None and fresh.token != stale.token and fresh.attempt == 2

    with factory() as session:
        assert heartbeat(session, job_id, stale.token) is False
        assert complete(session, job_id, stale.token) is False
        assert complete(session, job_id, fresh.token) is True
        session.commit()
    with factory() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.status == "succeeded" and job.worker_id == "worker-2"


def test_throttled_download_waits_for_the_longer_retry_after(pipeline):
    pipeline.transport.replace(
        "https://openreview.test/pdf?id=note-b",
        FakeResponse(429, {"retry-after": "120"}),
        _pdf(),
    )
    pipeline.ingest(limit=2)
    pipeline.worker.run_until_idle("worker-1")

    with pipeline.session() as session:
        throttled = session.execute(
            select(Job).where(Job.kind == "download_pdf", Job.status == "retry_wait")
        ).scalar_one()
        assert throttled.attempt == 1
        assert throttled.error_code == "throttled"
        remaining = session.execute(
            text("SELECT EXTRACT(EPOCH FROM (next_attempt_at - now())) FROM jobs WHERE id = :id"),
            {"id": throttled.id},
        ).scalar_one()
        assert 100 < float(remaining) <= 120

    assert pipeline.worker.run_until_idle("worker-1") == 0  # not due yet
    pipeline.release_due_jobs()
    assert pipeline.worker.run_until_idle("worker-1") == 2  # download then parse
    with pipeline.session() as session:
        assert {job.status for job in session.execute(select(Job)).scalars()} == {"succeeded"}


def test_kill_after_metadata_write_is_resumable(pipeline):
    original = pipeline.handlers["resolve_record"]

    def crashing(job, beat):
        outcome = original(job, beat)

        def writes(session):
            outcome.writes(session)
            raise RuntimeError("simulated worker death after metadata write")

        return JobOutcome(result=outcome.result, writes=writes)

    pipeline.handlers["resolve_record"] = crashing
    pipeline.ingest(limit=1)
    assert pipeline.worker.run_once("worker-1") is True
    with pipeline.session() as session:
        assert session.execute(select(func.count()).select_from(Paper)).scalar_one() == 0
        job = session.execute(select(Job).where(Job.kind == "resolve_record")).scalar_one()
        assert job.status == "retry_wait" and job.error_code == "handler_exception"

    pipeline.handlers["resolve_record"] = original
    pipeline.release_due_jobs()
    assert pipeline.worker.run_until_idle("worker-1") == 3
    with pipeline.session() as session:
        assert session.execute(select(func.count()).select_from(Paper)).scalar_one() == 1
        assert {job.status for job in session.execute(select(Job)).scalars()} == {"succeeded"}


def test_kill_after_parse_write_is_resumable(pipeline):
    original = pipeline.handlers["parse_pdf"]

    def crashing(job, beat):
        outcome = original(job, beat)

        def writes(session):
            outcome.writes(session)
            raise RuntimeError("simulated worker death after parse write")

        return JobOutcome(result=outcome.result, writes=writes)

    pipeline.handlers["parse_pdf"] = crashing
    pipeline.ingest(limit=1)
    pipeline.worker.run_until_idle("worker-1")
    with pipeline.session() as session:
        assert session.execute(select(func.count()).select_from(Chunk)).scalar_one() == 0
        version = session.execute(select(PaperVersion)).scalar_one()
        assert version.parse_status == "pending"

    pipeline.handlers["parse_pdf"] = original
    pipeline.release_due_jobs()
    assert pipeline.worker.run_until_idle("worker-1") == 1
    with pipeline.session() as session:
        chunk_count = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
        assert chunk_count > 0
        version = session.execute(select(PaperVersion)).scalar_one()
        assert version.parse_status == "parsed"
        # Re-running the parse writes must not duplicate chunks.
        assert (
            session.execute(
                select(func.count(func.distinct(Chunk.id))).select_from(Chunk)
            ).scalar_one()
            == chunk_count
        )


def test_failed_enrichment_leaves_the_base_paper_searchable(pipeline, tmp_path):
    pipeline.manifest = _manifest(
        tmp_path,
        {"semantic_scholar": {"api_base": "https://s2.test/graph/v1", "enabled": True}},
    )
    pipeline.context.manifest = yaml.safe_load(pipeline.manifest.read_text(encoding="utf-8"))
    pipeline.transport.add(
        "https://s2.test/graph/v1/paper/search",
        FakeResponse(500, {}, b"upstream error"),
        params={
            "query": "Accepted Paper A",
            "fields": "title,externalIds,citationCount",
            "limit": 5,
        },
    )
    pipeline.ingest(limit=1)
    pipeline.worker.run_until_idle("worker-1")

    with pipeline.session() as session:
        paper = session.execute(select(Paper)).scalar_one()
        assert paper.title == "Accepted Paper A"
        version = session.execute(select(PaperVersion)).scalar_one()
        assert version.parse_status == "parsed"
        enrich = session.execute(select(Job).where(Job.kind == "enrich_paper")).scalar_one()
        assert enrich.status == "retry_wait" and enrich.error_code == "provider_error"
        assert enrich.attempt == 1


def test_successful_enrichment_attaches_external_identifiers(pipeline, tmp_path):
    pipeline.manifest = _manifest(
        tmp_path,
        {"semantic_scholar": {"api_base": "https://s2.test/graph/v1", "enabled": True}},
    )
    pipeline.context.manifest = yaml.safe_load(pipeline.manifest.read_text(encoding="utf-8"))
    body = json.dumps(
        {
            "data": [
                {
                    "paperId": "s2-paper-a",
                    "title": "Accepted Paper A",
                    "externalIds": {"DOI": "10.5555/Paper-A", "ArXiv": "2401.00001v2"},
                    "citationCount": 7,
                }
            ]
        }
    ).encode()
    pipeline.transport.add(
        "https://s2.test/graph/v1/paper/search",
        FakeResponse(200, {"content-type": "application/json"}, body),
        params={
            "query": "Accepted Paper A",
            "fields": "title,externalIds,citationCount",
            "limit": 5,
        },
    )
    pipeline.ingest(limit=1)
    assert pipeline.worker.run_until_idle("worker-1") == 4

    with pipeline.session() as session:
        paper = session.execute(select(Paper)).scalar_one()
        identifiers = {
            (row.namespace, row.value, row.version)
            for row in session.execute(
                select(PaperIdentifier).where(PaperIdentifier.paper_id == paper.id)
            ).scalars()
        }
        assert ("doi", "10.5555/paper-a", None) in identifiers
        # P1.2 keeps the arXiv alias versionless; the version lives on paper_versions.
        assert ("arxiv", "2401.00001", None) in identifiers
        assert ("openreview", "note-a", None) in identifiers


def test_private_network_redirect_is_rejected_without_losing_the_abstract(pipeline):
    pipeline.transport.replace(
        "https://openreview.test/pdf?id=note-a",
        FakeResponse(302, {"location": "https://internal.openreview.test/a.pdf"}),
    )
    pipeline.context.resolver = lambda host: (
        ["10.0.0.5"] if host == "internal.openreview.test" else [PUBLIC_IP]
    )
    pipeline.ingest(limit=1)
    pipeline.worker.run_until_idle("worker-1")

    with pipeline.session() as session:
        download = session.execute(select(Job).where(Job.kind == "download_pdf")).scalar_one()
        assert download.status == "failed"
        assert download.error_code == "download_policy:address"
        paper = session.execute(select(Paper)).scalar_one()
        assert paper.abstract == "We study retrieval for paper A."
        version = session.execute(select(PaperVersion)).scalar_one()
        assert version.parse_status == "pending"
        assert session.execute(select(func.count()).select_from(Chunk)).scalar_one() == 0
    assert "https://internal.openreview.test/a.pdf" not in pipeline.transport.calls


def test_poisoned_pdf_produces_a_classified_failure(pipeline):
    pipeline.transport.replace(
        "https://openreview.test/pdf?id=note-b", _pdf(b"%PDF-1.7\n1 0 obj << garbage")
    )
    pipeline.ingest(limit=2)
    pipeline.worker.run_until_idle("worker-1")

    with pipeline.session() as session:
        versions = {
            version.paper.title: version
            for version in session.execute(select(PaperVersion)).scalars()
        }
        assert versions["Accepted Paper A"].parse_status == "parsed"
        poisoned = versions["Accepted Paper B"]
        assert poisoned.parse_status == "corrupt"
        assert poisoned.parser_version == PARSER_VERSION
        assert (
            session.execute(
                select(func.count()).select_from(Chunk).where(Chunk.paper_version_id == poisoned.id)
            ).scalar_one()
            == 0
        )
        parse_jobs = session.execute(select(Job).where(Job.kind == "parse_pdf")).scalars().all()
        assert {job.status for job in parse_jobs} == {"succeeded"}
        assert any(job.result.get("error_code") == "corrupt" for job in parse_jobs)


def test_openreview_login_token_is_sent_when_credentials_are_configured(pipeline, tmp_path):
    from copilot.corpus.sources.openreview import OpenReviewSource

    transport = pipeline.transport
    transport.add(
        "https://api.openreview.test/login",
        FakeResponse(200, {"content-type": "application/json"}, b'{"token": "tok-123"}'),
    )
    config = dict(pipeline.context.manifest["sources"]["openreview"])
    source = OpenReviewSource(
        config,
        transport,
        venue="ICLR",
        year=2024,
        track="main",
        credentials=("user@example.test", "secret"),
    )
    records, cursor = source.fetch_page(None)
    assert [record["source_item_id"] for record in records] == ["note-a", "note-w"]
    assert cursor == "2"
    assert transport.calls[0] == "POST https://api.openreview.test/login"
    assert transport.headers[-1] == {"Authorization": "Bearer tok-123"}

    anonymous = OpenReviewSource(config, transport, venue="ICLR", year=2024, track="main")
    challenge = FakeResponse(
        403, {"content-type": "application/json"}, b'{"name": "ChallengeRequiredError"}'
    )
    listing_key = FakeTransport.key(
        "https://api.openreview.test/notes",
        {"content.venueid": "ICLR.cc/2024/Conference", "limit": 2, "offset": 0},
    )
    transport.routes[listing_key] = [challenge]
    with pytest.raises(Exception, match="challenge_required"):
        anonymous.fetch_page(None)


def _mirror(tmp_path: Path, *, pdf_path: str = "pdfs/a.pdf") -> tuple[Path, dict[str, object]]:
    """A local papercli mirror: one record, one PDF, no network routes."""

    root = tmp_path / "papercli"
    (root / "pdfs").mkdir(parents=True, exist_ok=True)
    (root / "pdfs" / "a.pdf").write_bytes(FIXTURE_PDF)
    records = root / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "forum_id": "mirror-a",
                "venue": "ICLR",
                "year": 2024,
                "title": "A mirrored paper about retrieval",
                "abstract": "An abstract for the mirrored paper.",
                "forum_url": "https://openreview.net/forum?id=mirror-a",
                "openreview_pdf_url": (
                    "https://api2.openreview.net/attachment?name=pdf&id=mirror-a"
                ),
                "pdf_path": pdf_path,
                "bytes": len(FIXTURE_PDF),
                "sha256": FIXTURE_SHA,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    data: dict[str, object] = {
        "schema_version": 1,
        "venue": "ICLR",
        "years": [2024],
        "track": "main",
        "sample": {"limit": 10, "order": "source_item_id"},
        "sources": {
            "papercli": {
                "records": str(records),
                "root_dir": str(root),
                "dataset": "GenAI4ELab/papercli-papers",
                "dataset_revision": "90a1fbd",
                "pdf_dataset": "GenAI4ELab/papercli-papers-iclr",
                "pdf_dataset_revision": "050f8a4",
                "membership_is_acceptance": True,
                "page_size": 100,
            }
        },
    }
    manifest = tmp_path / "mirror.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return manifest, data


def _run_mirror(engine, tmp_path: Path, **kwargs) -> tuple[FakeTransport, int]:
    manifest, data = _mirror(tmp_path, **kwargs)
    transport = FakeTransport()  # no routes registered: any request would raise
    context = IngestContext(
        manifest=data,
        transport=transport,
        staging_dir=tmp_path / "sources",
        data_dir=tmp_path,
        parsing_config=parsing_config_for(tmp_path),
    )
    worker = Worker(engine, build_handlers(context))
    ingest(manifest, None, engine=engine, transport=transport)
    return transport, worker.run_until_idle("mirror-1")


def test_local_mirror_ingests_and_adopts_pdfs_without_network(
    migrated_database, test_settings, tmp_path
):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    try:
        transport, processed = _run_mirror(migrated_database, tmp_path)
        with session_factory(migrated_database)() as session:
            papers = session.execute(select(func.count()).select_from(Paper)).scalar_one()
            versions = list(session.execute(select(PaperVersion)).scalars())
            chunks = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
            identifiers = list(session.execute(select(PaperIdentifier)).scalars())
        assert papers == 1
        assert [identifier.namespace for identifier in identifiers] == ["openreview"]
        assert len(versions) == 1
        assert versions[0].source == "papercli"
        assert versions[0].content_sha256 == FIXTURE_SHA
        assert versions[0].parse_status == "parsed"
        assert versions[0].parser_version == PARSER_VERSION
        assert chunks > 0
        assert processed >= 2
        # The mirror is local: a run must not reach for the network at all.
        assert transport.calls == []
    finally:
        _truncate(migrated_database)


def test_mirror_pdf_outside_the_configured_root_is_refused(
    migrated_database, test_settings, tmp_path
):
    assert_safe_test_database(test_settings)
    _truncate(migrated_database)
    try:
        _run_mirror(migrated_database, tmp_path, pdf_path="../../etc/passwd")
        with session_factory(migrated_database)() as session:
            versions = list(session.execute(select(PaperVersion)).scalars())
            chunks = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
            failed = list(
                session.execute(select(Job).where(Job.kind == "adopt_pdf")).scalars()
            )
        # Metadata survives; the traversal attempt fails permanently and parses nothing.
        assert len(versions) == 1
        assert chunks == 0
        assert failed and failed[0].status == "failed"
        assert failed[0].error_code is not None
        assert "mirror_path" in failed[0].error_code
    finally:
        _truncate(migrated_database)
