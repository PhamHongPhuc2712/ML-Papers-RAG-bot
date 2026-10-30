"""Pilot ingestion: membership listing, deterministic sampling, and job handlers.

``ingest`` lists a venue-year through its membership sources, keeps eligible
records, sorts them by source item ID, takes the sample limit, and enqueues
one durable ``resolve_record`` job per record together with the source
checkpoint in a single transaction. The expensive per-paper pipeline
(resolve → download → parse, plus optional enrichment) runs as leased jobs, so
a killed worker resumes without duplicates.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..db.models import Paper, PaperAuthor, PaperIdentifier, PaperVersion, SourceCheckpoint
from ..db.session import session_scope
from ..jobs.queue import LeasedJob, enqueue
from ..jobs.worker import Handler, Heartbeat, JobError, JobOutcome, ThrottledError
from .chunk import ParsingConfig, chunk_document, token_spans_for
from .dedupe import IdentityConflictError, QuarantineError, RecordValidationError, resolve_paper
from .documents import store_parsed_document
from .download import DEFAULT_ALLOWED_HOSTS, DownloadPolicyError, Resolver, default_resolver
from .download import download_pdf as download_pdf_file
from .parse import ParseStatus, parse_pdf_result
from .sources.arxiv import ArxivSource
from .sources.base import (
    ProviderError,
    ProviderThrottled,
    SourceAdapter,
    Transport,
    is_eligible,
    load_manifest,
    partition_key,
    utc_now,
)
from .sources.openreview import DEFAULT_API_BASE as DEFAULT_OPENREVIEW_API
from .sources.openreview import OpenReviewSource
from .sources.papercli import PapercliSource
from .sources.proceedings import ProceedingsSource
from .sources.semantic_scholar import SemanticScholarSource

MEMBERSHIP_SOURCES: dict[
    str,
    type[OpenReviewSource] | type[ProceedingsSource] | type[ArxivSource] | type[PapercliSource],
] = {
    "openreview": OpenReviewSource,
    "proceedings": ProceedingsSource,
    "papercli": PapercliSource,
    "arxiv": ArxivSource,
}


@dataclass
class IngestContext:
    """Everything handlers need that is not the job itself; injectable in tests."""

    manifest: dict[str, Any]
    transport: Transport
    staging_dir: Path
    # The one local data root: the pinned chunking tokenizer is cached under it.
    data_dir: Path
    parsing_config: ParsingConfig
    resolver: Resolver = default_resolver
    now: Callable[[], datetime] = field(default=utc_now)
    _openreview: OpenReviewSource | None = field(default=None, init=False, repr=False)

    @property
    def pdf_dir(self) -> Path:
        return self.staging_dir / "pdfs"

    @property
    def download(self) -> Mapping[str, Any]:
        value = self.manifest.get("download", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def allowed_hosts(self) -> frozenset[str]:
        hosts = self.download.get("allowed_hosts")
        return (
            frozenset(str(h) for h in hosts) if isinstance(hosts, list) else DEFAULT_ALLOWED_HOSTS
        )

    @property
    def max_bytes(self) -> int:
        return int(self.download.get("max_bytes", self.parsing_config.parser.max_pdf_bytes))

    @property
    def mirror_roots(self) -> tuple[Path, ...]:
        """Directories a mirrored PDF may be adopted from; anything else is refused."""

        roots: list[Path] = [self.staging_dir]
        config = self.source_config("papercli")
        if config is not None:
            declared = config.get("root_dir") or Path(str(config["records"])).parent
            roots.append(Path(str(declared)))
        return tuple(root.resolve() for root in roots)

    def source_config(self, name: str) -> Mapping[str, Any] | None:
        sources = self.manifest.get("sources", {})
        config = sources.get(name) if isinstance(sources, Mapping) else None
        if not isinstance(config, Mapping) or not config.get("enabled", True):
            return None
        return config

    def download_headers(self, url: str) -> dict[str, str]:
        """Bearer token for OpenReview API downloads; other hosts get no credentials."""

        config = self.source_config("openreview")
        if config is None:
            return {}
        api_host = urlsplit(str(config.get("api_base", DEFAULT_OPENREVIEW_API))).hostname
        if urlsplit(url).hostname != api_host:
            return {}
        if self._openreview is None:
            self._openreview = OpenReviewSource(
                config,
                self.transport,
                venue=str(self.manifest["venue"]),
                year=int(self.manifest["years"][0]),
                track=str(self.manifest["track"]),
                now=self.now,
            )
        return self._openreview.auth_headers()


def build_membership_sources(
    manifest: Mapping[str, Any],
    transport: Transport,
    *,
    year: int,
    now: Callable[[], datetime] = utc_now,
) -> list[SourceAdapter]:
    adapters: list[SourceAdapter] = []
    sources = manifest.get("sources", {})
    if not isinstance(sources, Mapping):
        return adapters
    for name, factory in MEMBERSHIP_SOURCES.items():
        config = sources.get(name)
        if not isinstance(config, Mapping) or not config.get("enabled", True):
            continue
        if name == "arxiv" and not config.get("membership", False):
            continue  # arXiv supplements; it only admits papers when explicitly configured
        adapters.append(
            factory(
                config,
                transport,
                venue=str(manifest["venue"]),
                year=year,
                track=str(manifest["track"]),
                now=now,
            )
        )
    return adapters


def list_candidates(
    manifest: Mapping[str, Any],
    transport: Transport,
    *,
    now: Callable[[], datetime] = utc_now,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Page through every membership source; return eligible records and page counts."""

    eligible: list[dict[str, Any]] = []
    pages: dict[str, int] = {}
    for year in manifest["years"]:
        for adapter in build_membership_sources(manifest, transport, year=int(year), now=now):
            cursor: str | None = None
            key = f"{adapter.source}:{year}"
            while True:
                records, cursor = adapter.fetch_page(cursor)
                pages[key] = pages.get(key, 0) + 1
                eligible.extend(record for record in records if is_eligible(record, manifest))
                if cursor is None:
                    break
    eligible.sort(key=lambda record: str(record.get("source_item_id")))
    return eligible, pages


def _resolve_key(record: Mapping[str, Any]) -> str:
    return (
        f"resolve:{record['source']}:{record.get('source_item_id')}:{record.get('source_revision')}"
    )


def ingest(
    manifest_path: str | Path,
    limit: int | None,
    *,
    engine: Engine,
    transport: Transport,
    run_id: str | None = None,
    now: Callable[[], datetime] = utc_now,
) -> str:
    """List, sample deterministically, enqueue, and checkpoint. Returns the run ID."""

    manifest = load_manifest(manifest_path)
    started = now()
    run_id = run_id or (
        f"{manifest['venue']}-{'-'.join(str(y) for y in manifest['years'])}-"
        f"{started:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    eligible, pages = list_candidates(manifest, transport, now=now)
    sample = manifest.get("sample", {})
    sample_limit = (
        limit
        if limit is not None
        else (int(sample["limit"]) if isinstance(sample, Mapping) and sample.get("limit") else None)
    )
    selected = eligible[:sample_limit] if sample_limit else eligible

    with session_scope(engine) as session:
        for record in selected:
            enqueue(
                session,
                "resolve_record",
                {"run_id": run_id, "record": record},
                _resolve_key(record),
            )
        for year in manifest["years"]:
            sources = build_membership_sources(manifest, transport, year=int(year), now=now)
            for source_name in {adapter.source for adapter in sources}:
                page_count = pages.get(f"{source_name}:{year}", 0)
                cursor_value = (
                    f"complete:pages={page_count}:eligible={len(eligible)}:selected={len(selected)}"
                )
                statement = (
                    pg_insert(SourceCheckpoint)
                    .values(
                        id=uuid4(),
                        source=source_name,
                        partition=partition_key(manifest, int(year)),
                        cursor=cursor_value,
                        run_id=run_id,
                    )
                    .on_conflict_do_update(
                        index_elements=["source", "partition"],
                        set_={"cursor": cursor_value, "run_id": run_id, "updated_at": started},
                    )
                )
                session.execute(statement)
    return run_id


def _find_version(
    session: Session, paper_id: UUID, source: str, source_revision: str, version: str | None
) -> PaperVersion | None:
    statement = select(PaperVersion).where(
        PaperVersion.paper_id == paper_id,
        PaperVersion.source == source,
        PaperVersion.source_revision == source_revision,
        PaperVersion.version == version if version is not None else PaperVersion.version.is_(None),
    )
    return session.execute(statement).scalar_one_or_none()


def _paper_context(session: Session, paper_id: UUID) -> tuple[list[str], dict[str, str]]:
    authors = [
        link.author.name
        for link in session.execute(
            select(PaperAuthor)
            .where(PaperAuthor.paper_id == paper_id)
            .order_by(PaperAuthor.position)
        ).scalars()
    ]
    identifiers = {
        row.namespace: row.value + (row.version or "")
        for row in session.execute(
            select(PaperIdentifier).where(PaperIdentifier.paper_id == paper_id)
        ).scalars()
    }
    return authors, identifiers


def build_handlers(context: IngestContext) -> dict[str, Handler]:
    """Job handlers closed over the context; external work stays outside transactions."""

    parser = context.parsing_config.parser
    chunker = context.parsing_config.chunker
    # Resolved once, at worker startup: an absent or altered tokenizer must stop
    # the run here rather than silently re-measure windows in the middle of it.
    spans = token_spans_for(chunker.tokenizer, context.data_dir)

    def resolve_record(job: LeasedJob, beat: Heartbeat) -> JobOutcome:
        record = dict(job.payload["record"])
        run_id = str(job.payload.get("run_id", ""))
        result: dict[str, Any] = {"run_id": run_id}

        def writes(session: Session) -> None:
            try:
                paper_id = resolve_paper(dict(record), session, staging_dir=context.staging_dir)
            except QuarantineError as error:
                result.update(outcome="quarantined", code=error.code)
                return
            except IdentityConflictError as error:
                result.update(outcome="conflict", code=error.code)
                return
            except RecordValidationError as error:
                raise JobError(f"record_invalid:{error.code}", retryable=False) from error
            result.update(outcome="resolved", paper_id=str(paper_id))
            pdf_path = record.get("pdf_path")
            pdf_url = record.get("pdf_url")
            target = (
                f"{record['source']}:{record.get('source_item_id')}:"
                f"{record.get('source_revision')}"
            )
            common = {
                "run_id": run_id,
                "paper_id": str(paper_id),
                "source": record["source"],
                "source_item_id": record.get("source_item_id"),
                "source_revision": record.get("source_revision", "unknown"),
                "version": record.get("version"),
            }
            # A mirrored copy already on disk is adopted; only a remote-only record
            # goes through the hardened downloader.
            if isinstance(pdf_path, str) and pdf_path:
                enqueue(
                    session,
                    "adopt_pdf",
                    {**common, "path": pdf_path, "expected_sha256": record.get("pdf_sha256")},
                    f"adopt:{target}",
                )
            elif isinstance(pdf_url, str) and pdf_url:
                enqueue(
                    session,
                    "download_pdf",
                    {**common, "pdf_url": pdf_url},
                    f"download:{target}",
                )
            if context.source_config("semantic_scholar") is not None:
                authors, identifiers = _paper_context(session, paper_id)
                enqueue(
                    session,
                    "enrich_paper",
                    {
                        "run_id": run_id,
                        "paper_id": str(paper_id),
                        "title": record.get("title"),
                        "authors": authors,
                        "year": record.get("year"),
                        "known_ids": identifiers,
                    },
                    f"enrich:semantic_scholar:{paper_id}",
                )

        return JobOutcome(result=result, writes=writes)

    def download_pdf(job: LeasedJob, beat: Heartbeat) -> JobOutcome:
        payload = job.payload
        try:
            downloaded = download_pdf_file(
                str(payload["pdf_url"]),
                context.pdf_dir,
                context.transport,
                allowed_hosts=context.allowed_hosts,
                max_bytes=context.max_bytes,
                resolver=context.resolver,
                headers=context.download_headers(str(payload["pdf_url"])),
            )
        except DownloadPolicyError as error:
            raise JobError(f"download_policy:{error.reason}", retryable=False) from error
        except ProviderThrottled as error:
            raise ThrottledError(error.retry_after) from error
        except ProviderError as error:
            raise JobError(error.code) from error
        except OSError as error:
            raise JobError("download_io") from error
        result: dict[str, Any] = {
            "sha256": downloaded.sha256,
            "bytes": downloaded.size,
            "final_url": downloaded.final_url,
        }

        def writes(session: Session) -> None:
            version = _find_version(
                session,
                UUID(str(payload["paper_id"])),
                str(payload["source"]),
                str(payload.get("source_revision", "unknown")),
                payload.get("version"),
            )
            if version is None:
                raise JobError("version_missing", retryable=False)
            if version.content_sha256 != downloaded.sha256:
                version.content_sha256 = downloaded.sha256
            version.source_url = downloaded.final_url
            session.flush()
            enqueue(
                session,
                "parse_pdf",
                {
                    "run_id": payload.get("run_id"),
                    "paper_version_id": str(version.id),
                    "path": str(downloaded.path),
                    "content_sha256": downloaded.sha256,
                },
                f"parse:{version.id}:{downloaded.sha256}:{parser.parser_version}:{chunker.chunker_version}",
            )

        return JobOutcome(result=result, writes=writes)

    def adopt_pdf(job: LeasedJob, beat: Heartbeat) -> JobOutcome:
        """Register a PDF already mirrored on disk, without any network access."""

        payload = job.payload
        candidate = Path(str(payload["path"]))
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise JobError("mirror_path_missing", retryable=False) from error
        if not any(resolved.is_relative_to(root) for root in context.mirror_roots):
            raise JobError("mirror_path_outside_root", retryable=False)
        digest = hashlib.sha256()
        size = 0
        try:
            with resolved.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
                    size += len(block)
        except OSError as error:
            raise JobError("mirror_read_failed") from error
        checksum = digest.hexdigest()
        expected = payload.get("expected_sha256")
        if isinstance(expected, str) and expected and expected != checksum:
            raise JobError("mirror_checksum_mismatch", retryable=False)
        result: dict[str, Any] = {"sha256": checksum, "bytes": size, "path": str(resolved)}

        def writes(session: Session) -> None:
            version = _find_version(
                session,
                UUID(str(payload["paper_id"])),
                str(payload["source"]),
                str(payload.get("source_revision", "unknown")),
                payload.get("version"),
            )
            if version is None:
                raise JobError("version_missing", retryable=False)
            version.content_sha256 = checksum
            session.flush()
            enqueue(
                session,
                "parse_pdf",
                {
                    "run_id": payload.get("run_id"),
                    "paper_version_id": str(version.id),
                    "path": str(resolved),
                    "content_sha256": checksum,
                },
                f"parse:{version.id}:{checksum}:{parser.parser_version}:{chunker.chunker_version}",
            )

        return JobOutcome(result=result, writes=writes)

    def parse_pdf(job: LeasedJob, beat: Heartbeat) -> JobOutcome:
        payload = job.payload
        parsed = parse_pdf_result(Path(str(payload["path"])), max_bytes=parser.max_pdf_bytes)
        chunks = (
            chunk_document(parsed.sections, spans, chunker)
            if parsed.status is ParseStatus.PARSED
            else []
        )
        result: dict[str, Any] = {
            "status": parsed.status.value,
            "error_code": parsed.error_code.value if parsed.error_code else None,
            "pages": parsed.page_count,
            "parser_version": parsed.parser_version,
        }

        def writes(session: Session) -> None:
            version = session.get(PaperVersion, UUID(str(payload["paper_version_id"])))
            if version is None:
                raise JobError("version_missing", retryable=False)
            ids = store_parsed_document(
                session,
                version,
                parsed,
                chunks,
                chunker_version=chunker.chunker_version,
                namespace=chunker.uuid_namespace,
            )
            result["chunks"] = len(ids)

        return JobOutcome(result=result, writes=writes)

    def enrich_paper(job: LeasedJob, beat: Heartbeat) -> JobOutcome:
        config = context.source_config("semantic_scholar")
        if config is None:
            return JobOutcome(result={"outcome": "disabled"})
        payload = job.payload
        year = payload.get("year")
        adapter = SemanticScholarSource(
            config,
            context.transport,
            venue=str(context.manifest["venue"]),
            year=int(year) if isinstance(year, int) else int(context.manifest["years"][0]),
            track=str(context.manifest["track"]),
            now=context.now,
        )
        known = {str(k): str(v) for k, v in dict(payload.get("known_ids", {})).items()}
        title = str(payload.get("title") or "")
        try:
            found = adapter.lookup(title=title, doi=known.get("doi"), arxiv=known.get("arxiv"))
        except ProviderThrottled as error:
            raise ThrottledError(error.retry_after) from error
        except ProviderError as error:
            raise JobError(error.code) from error
        if found is None:
            return JobOutcome(result={"outcome": "not_found"})
        record = adapter.enrichment_record(
            found,
            title=title,
            authors=[str(name) for name in payload.get("authors", [])],
            known_ids=known,
        )
        result: dict[str, Any] = {
            "outcome": "enriched",
            "citation_count": record.get("citation_count"),
        }

        def writes(session: Session) -> None:
            try:
                paper_id = resolve_paper(record, session, staging_dir=context.staging_dir)
            except QuarantineError as error:
                result.update(outcome="quarantined", code=error.code)
                return
            except IdentityConflictError as error:
                result.update(outcome="conflict", code=error.code)
                return
            if session.get(Paper, paper_id) is None:
                raise JobError("paper_missing", retryable=False)
            result["paper_id"] = str(paper_id)

        return JobOutcome(result=result, writes=writes)

    return {
        "resolve_record": resolve_record,
        "download_pdf": download_pdf,
        "adopt_pdf": adopt_pdf,
        "parse_pdf": parse_pdf,
        "enrich_paper": enrich_paper,
    }
