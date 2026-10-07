"""Load Open RAG Bench's 1,000 papers through our own pipeline (ORB plan, O2).

The benchmark ships Mistral-OCR sections, but the retrieval it measures here is
the retrieval that serves ``copilot_v2``, so the papers go through the same
identity resolution, the same hardened download, ``pypdf-text-v2`` and
``paragraph-pack-v1``. The corpus JSON is read only for metadata: title,
abstract, authors, publication date and the arXiv id, which the identity layer
stores as a paper identifier so the dataset's gold documents resolve by id and
never by title.

No job queue: a thousand papers is one loop, one transaction per paper, resumable
because a paper whose version is already parsed is skipped. Every PDF is kept
under ``DATA_DIR/benchmarks/orb/pdfs`` for O3's section mapping.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import Engine, select

from ..corpus.chunk import ParsingConfig, chunk_document, token_spans_for
from ..corpus.dedupe import IdentityResolutionError, resolve_paper
from ..corpus.documents import store_parsed_document
from ..corpus.download import DownloadPolicyError, Resolver, default_resolver, download_pdf
from ..corpus.parse import ParseStatus, parse_pdf_result
from ..corpus.sources.base import ProviderError, ProviderThrottled, Transport
from ..db.models import PaperVersion
from ..db.session import session_scope
from .orb import ORB_REVISION, OrbError, OrbPaths

ORB_SOURCE = "orb"
ARXIV_HOSTS = frozenset({"arxiv.org", "export.arxiv.org"})
# arXiv asks bulk clients to pace themselves; one request a second is well inside it.
DEFAULT_DELAY_SECONDS = 1.0
THROTTLE_ATTEMPTS = 5
# A stalled or dropped connection mid-download is retried after a pause, then recorded.
TRANSPORT_ATTEMPTS = 3
TRANSPORT_BACKOFF_SECONDS = 15.0


@dataclass(slots=True)
class LoadReport:
    papers: int = 0
    parsed: int = 0
    skipped: int = 0
    failed: dict[str, int] = field(default_factory=dict)
    chunks: int = 0
    chunk_tokens_p50: float | None = None
    chunk_tokens_max: int | None = None
    failures: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "papers": self.papers,
            "parsed": self.parsed,
            "skipped": self.skipped,
            "failed": dict(sorted(self.failed.items())),
            "chunks": self.chunks,
            "chunk_tokens_p50": self.chunk_tokens_p50,
            "chunk_tokens_max": self.chunk_tokens_max,
        }


def _year(paper: Mapping[str, Any]) -> int | None:
    published = paper.get("published")
    if isinstance(published, str) and len(published) >= 4 and published[:4].isdigit():
        return int(published[:4])
    return None


def orb_record(
    paper: Mapping[str, Any], pdf_url: str, *, revision: str = ORB_REVISION
) -> dict[str, Any]:
    """A source record for ``resolve_paper`` from one ORB corpus file.

    The arXiv id goes in as the ``arxiv`` identifier, versioned as written; the
    identity layer keeps the stem as the alias and the version beside it. No
    venue: ORB papers are arXiv preprints, not accepted main-track papers, and
    this corpus is never exported or activated.
    """

    doc_id = str(paper["id"])
    title = " ".join(str(paper.get("title") or "").split())
    if not title:
        raise OrbError("paper_without_title", doc_id)
    authors = [str(name) for name in paper.get("authors") or [] if str(name).strip()]
    abstract = paper.get("abstract")
    return {
        "source": ORB_SOURCE,
        "source_item_id": doc_id,
        "source_revision": revision,
        "title": title,
        "abstract": " ".join(str(abstract).split()) if isinstance(abstract, str) else None,
        "authors": authors,
        "year": _year(paper),
        "arxiv": doc_id,
        "pdf_url": pdf_url,
        "source_url": f"https://arxiv.org/abs/{doc_id}",
        "redistribution": "unknown",
    }


def _version_for(session: Any, paper_id: UUID, revision: str, version: str | None) -> PaperVersion:
    statement = select(PaperVersion).where(
        PaperVersion.paper_id == paper_id,
        PaperVersion.source == ORB_SOURCE,
        PaperVersion.source_revision == revision,
        PaperVersion.version == version if version is not None else PaperVersion.version.is_(None),
    )
    found: PaperVersion | None = session.execute(statement).scalar_one_or_none()
    if found is None:
        raise OrbError("version_missing", str(paper_id))
    return found


def _arxiv_version(doc_id: str) -> str | None:
    from ..corpus.normalize import normalize_arxiv

    return normalize_arxiv(doc_id)[1]


def _download(
    url: str,
    paths: OrbPaths,
    transport: Transport,
    *,
    allowed_hosts: frozenset[str],
    max_bytes: int,
    resolver: Resolver,
    sleep: Callable[[float], None],
) -> Any:
    throttled = 0
    stalled = 0
    while True:
        try:
            return download_pdf(
                url,
                paths.pdfs,
                transport,
                allowed_hosts=allowed_hosts,
                max_bytes=max_bytes,
                resolver=resolver,
            )
        except ProviderThrottled as error:
            throttled += 1
            if throttled >= THROTTLE_ATTEMPTS:
                raise
            sleep(float(error.retry_after or 5.0))
        except httpx.TransportError as error:
            # A read timeout or a dropped connection from the publisher, seen on the
            # first real run after 433 papers: not a policy failure and not a 5xx.
            stalled += 1
            if stalled >= TRANSPORT_ATTEMPTS:
                raise ProviderError("download_transport", type(error).__name__) from error
            sleep(TRANSPORT_BACKOFF_SECONDS)


def load_orb_corpus(
    paths: OrbPaths,
    *,
    engine: Engine,
    parsing_config: ParsingConfig,
    data_dir: Path,
    transport: Transport,
    limit: int | None = None,
    revision: str = ORB_REVISION,
    allowed_hosts: frozenset[str] = ARXIV_HOSTS,
    resolver: Resolver = default_resolver,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> LoadReport:
    """Resolve, download, parse and chunk every paper in ``pdf_urls.json``.

    A paper whose version is already parsed is skipped, so a stopped run resumes.
    A download or parse that fails is recorded with its typed reason and the
    loop continues; the PDF, when there is one, stays on disk for a retry.
    """

    urls = json.loads(paths.member("pdf/arxiv/pdf_urls.json").read_text(encoding="utf-8"))
    if not isinstance(urls, Mapping):
        raise OrbError("pdf_urls_invalid")
    doc_ids = sorted(urls)
    if limit is not None:
        doc_ids = doc_ids[:limit]
    parser = parsing_config.parser
    chunker = parsing_config.chunker
    spans = token_spans_for(chunker.tokenizer, data_dir)
    staging = paths.root / "artifacts"
    report = LoadReport()
    token_counts: list[int] = []
    failed: Counter[str] = Counter()

    for doc_id in doc_ids:
        report.papers += 1
        corpus_file = paths.corpus_file(doc_id)
        if not corpus_file.exists():
            failed["corpus_file_missing"] += 1
            report.failures[doc_id] = "corpus_file_missing"
            continue
        paper = json.loads(corpus_file.read_text(encoding="utf-8"))
        record = orb_record(paper, str(urls[doc_id]), revision=revision)
        version_label = _arxiv_version(doc_id)

        with session_scope(engine) as session:
            try:
                paper_id = resolve_paper(record, session, staging_dir=staging)
            except IdentityResolutionError as error:
                failed[f"identity:{error.code}"] += 1
                report.failures[doc_id] = f"identity:{error.code}"
                continue
            version = _version_for(session, paper_id, revision, version_label)
            if version.parse_status == ParseStatus.PARSED.value:
                report.skipped += 1
                continue
            version_id = version.id

        try:
            downloaded = _download(
                str(urls[doc_id]),
                paths,
                transport,
                allowed_hosts=allowed_hosts,
                max_bytes=max(int(parser.max_pdf_bytes), 1),
                resolver=resolver,
                sleep=sleep,
            )
        except DownloadPolicyError as error:
            failed[f"download_policy:{error.reason}"] += 1
            report.failures[doc_id] = f"download_policy:{error.reason}"
            continue
        except (ProviderThrottled, ProviderError) as error:
            code = getattr(error, "code", "throttled")
            failed[f"download:{code}"] += 1
            report.failures[doc_id] = f"download:{code}"
            continue
        except OSError:
            failed["download:io"] += 1
            report.failures[doc_id] = "download:io"
            continue
        if delay_seconds > 0:
            sleep(delay_seconds)

        parsed = parse_pdf_result(downloaded.path, max_bytes=parser.max_pdf_bytes)
        chunks = (
            chunk_document(parsed.sections, spans, chunker)
            if parsed.status is ParseStatus.PARSED
            else []
        )
        with session_scope(engine) as session:
            stored = session.get(PaperVersion, version_id)
            if stored is None:
                raise OrbError("version_missing", str(version_id))
            stored.content_sha256 = downloaded.sha256
            stored.source_url = downloaded.final_url
            ids = store_parsed_document(
                session,
                stored,
                parsed,
                chunks,
                chunker_version=chunker.chunker_version,
                namespace=chunker.uuid_namespace,
            )
        if parsed.status is ParseStatus.PARSED:
            report.parsed += 1
            report.chunks += len(ids)
            token_counts.extend(int(chunk["token_count"]) for chunk in chunks)
        else:
            assert parsed.error_code is not None
            failed[f"parse:{parsed.error_code.value}"] += 1
            report.failures[doc_id] = f"parse:{parsed.error_code.value}"
        if progress is not None:
            progress(doc_id, report.as_dict())

    report.failed = dict(failed)
    if token_counts:
        report.chunk_tokens_p50 = float(statistics.median(token_counts))
        report.chunk_tokens_max = max(token_counts)
    return report
