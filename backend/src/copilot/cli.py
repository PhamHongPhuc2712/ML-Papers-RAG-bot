"""Command-line entry points: fixture replay, pilot ingestion, and the job worker."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

from .corpus.dedupe import IdentityConflictError, QuarantineError, resolve_paper
from .db.session import make_engine, migrate_database, session_factory

DEFAULT_FIXTURE_PATH = Path("data/fixtures/metadata.jsonl")
DEFAULT_MANIFEST_PATH = Path("configs/corpus.yaml")
DEFAULT_PARSING_PATH = Path("configs/parsing.yaml")
DEFAULT_ARTIFACTS_PATH = Path("configs/artifacts.yaml")
DEFAULT_MODELS_PATH = Path("configs/models.yaml")
DEFAULT_SEARCH_PATH = Path("configs/search.yaml")
DEFAULT_EXPERIMENT_PATH = Path("configs/experiments/retrieval.yaml")
DEFAULT_EVALUATION_PATH = Path("configs/evaluation.yaml")
DEFAULT_SMOKE_PATH = Path("data/fixtures/retrieval-smoke")
DEFAULT_DATASET_PATH = Path("data/fixtures/retrieval")


def default_data_dir() -> Path | None:
    """The single local data root, read from the environment (spec §3)."""

    root = os.environ.get("COPILOT_DATA_DIR") or os.environ.get("DATA_DIR") or ""
    root = root.strip()
    if not root:
        return None
    return Path(root.rstrip("\\/") or root)


def default_staging_dir() -> Path | None:
    """Raw source artifacts live under the configured data root unless overridden."""

    root = default_data_dir()
    return None if root is None else root / "sources"


def load_fixtures(
    fixture_path: str | Path,
    *,
    database_url: str,
    staging_dir: str | Path,
) -> dict[str, int]:
    """Replay JSONL metadata records into the configured PostgreSQL database."""

    path = Path(fixture_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    engine = make_engine(database_url)
    try:
        migrate_database(engine)
        factory = session_factory(engine)
        counts = {"loaded": 0, "quarantined": 0, "conflicts": 0}
        with factory() as session:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    try:
                        decoded = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"fixture_json_invalid:{line_number}") from exc
                    if not isinstance(decoded, dict):
                        raise ValueError(f"fixture_record_invalid:{line_number}")
                    with session.begin():
                        try:
                            resolve_paper(decoded, session, staging_dir=staging_dir)
                        except QuarantineError:
                            counts["quarantined"] += 1
                        except IdentityConflictError:
                            counts["conflicts"] += 1
                        else:
                            counts["loaded"] += 1
        return counts
    finally:
        engine.dispose()


def _database_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    from .config import Settings

    return Settings().database_url  # type: ignore[call-arg]


def _llm_daily_cap() -> Any:
    from .config import Settings

    return Settings().llm_daily_spend_cap_usd  # type: ignore[call-arg]


def _staging_dir(explicit: Path | None, parser: argparse.ArgumentParser) -> Path:
    staging_dir = explicit or default_staging_dir()
    if staging_dir is None:
        parser.error("--staging-dir is required when DATA_DIR is not set")
    return staging_dir


def _data_dir(explicit: Path | None, parser: argparse.ArgumentParser) -> Path:
    data_dir = explicit or default_data_dir()
    if data_dir is None:
        parser.error("--data-dir is required when DATA_DIR is not set")
    return data_dir


def run_ingest(
    manifest: Path, limit: int | None, *, database_url: str, staging_dir: Path
) -> dict[str, object]:
    from .corpus.ingest import ingest
    from .corpus.sources.base import HttpxTransport

    engine = make_engine(database_url)
    transport = HttpxTransport()
    try:
        migrate_database(engine)
        run_id = ingest(manifest, limit, engine=engine, transport=transport)
        return {"run_id": run_id, "manifest": str(manifest), "limit": limit}
    finally:
        transport.close()
        engine.dispose()


def run_mirror_index(
    *,
    registry: Path | None,
    venue: str,
    year: int,
    pdf_root: Path | None,
    out: Path | None,
    data_dir: Path,
) -> dict[str, object]:
    """Project one venue-year out of the registry parquet into an ingest index."""

    from .corpus.mirror import build_index

    mirror = data_dir / "sources" / "papercli"
    registry = registry or mirror / "papers.parquet"
    pdf_root = pdf_root or mirror / "pdfs"
    out = out or mirror / f"{venue.lower()}-{year}.jsonl"
    counts = build_index(registry, venue=venue, year=year, pdf_root=pdf_root, out=out)
    return {"venue": venue, "year": year, "out": str(out), **counts}


def run_mirror(
    *,
    venue: str,
    year: int,
    registry: Path | None,
    revision: str | None,
    workers: int,
    min_free_gb: float,
    data_dir: Path,
) -> dict[str, object]:
    """Capture one venue-year's PDFs at a pinned revision and index them."""

    from .corpus.mirror import mirror_venue_year, resolve_revision, shard_repo

    mirror = data_dir / "sources" / "papercli"
    registry = registry or mirror / "papers.parquet"
    revision = revision or resolve_revision(shard_repo(venue))
    return mirror_venue_year(
        registry,
        venue=venue,
        year=year,
        mirror_root=mirror,
        revision=revision,
        workers=workers,
        min_free_bytes=int(min_free_gb * 1024**3),
    )


def _exports_dir(data_dir: Path, config: Path) -> Path:
    """Snapshots stage under the data root, never inside the repository."""

    import yaml

    raw = yaml.safe_load(config.read_text(encoding="utf-8")) if config.is_file() else {}
    directory = ((raw or {}).get("export") or {}).get("directory", "exports")
    return data_dir / str(directory)


def _snapshot_path(value: Path, data_dir: Path, config: Path) -> Path:
    return value if value.is_absolute() else _exports_dir(data_dir, config) / value


def run_export(
    *, run_id: str, out: Path, database_url: str, data_dir: Path, artifacts: Path
) -> dict[str, object]:
    """Write an immutable snapshot of the corpus's exportable content."""

    from .corpus.export import export_snapshot

    destination = _snapshot_path(out, data_dir, artifacts)
    engine = make_engine(database_url)
    try:
        manifest = export_snapshot(run_id, destination, engine=engine)
        return {
            "out": str(destination),
            "shards": len(manifest["shards"]),
            **{k: manifest[k] for k in ("counts", "rights", "digests")},
        }
    finally:
        engine.dispose()


def run_validate(*, manifest: Path, data_dir: Path, artifacts: Path) -> dict[str, object]:
    from .corpus.export import validate_manifest

    path = _snapshot_path(manifest, data_dir, artifacts)
    checked = validate_manifest(path)
    return {
        "manifest": str(path),
        "valid": True,
        "run_id": checked["run_id"],
        "counts": checked["counts"],
        "shards": [shard["path"] for shard in checked["shards"]],
    }


def run_restore(
    *, manifest: Path, database_url: str, data_dir: Path, artifacts: Path
) -> dict[str, object]:
    """Load a validated snapshot into an empty database to prove portability."""

    from .corpus.export import restore_snapshot

    path = _snapshot_path(manifest, data_dir, artifacts)
    engine = make_engine(database_url)
    try:
        migrate_database(engine)
        return {"manifest": str(path), **restore_snapshot(path, engine=engine)}
    finally:
        engine.dispose()


def run_coverage(*, database_url: str, artifacts: Path) -> dict[str, object]:
    from .corpus.api import corpus_coverage, expected_totals

    engine = make_engine(database_url)
    try:
        return corpus_coverage(engine, expected_totals(artifacts))
    finally:
        engine.dispose()


def run_compare_chunkers(
    *, source: Path, sample: int, seed: int, data_dir: Path, parsing_config: Path
) -> dict[str, object]:
    """Chunk the same PDFs under both policies and report where each one cuts."""

    from .corpus.chunk import load_parsing_config, token_spans_for
    from .corpus.compare import compare_policies

    config = load_parsing_config(parsing_config)
    spans = token_spans_for(config.chunker.tokenizer, data_dir)
    pdfs = sorted(Path(source).rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs under {source}")
    return compare_policies(
        pdfs,
        spans,
        config.chunker,
        max_pdf_bytes=config.parser.max_pdf_bytes,
        sample=sample,
        seed=seed,
    )


def run_rechunk(
    *,
    database_url: str,
    data_dir: Path,
    parsing_config: Path,
    source_overlap: int,
    limit: int | None,
    shards: int,
    shard: int,
) -> dict[str, object]:
    """Re-chunk stored documents under the configured policy, without PDFs."""

    import sys

    from .corpus.chunk import load_parsing_config, token_spans_for
    from .corpus.rechunk import RechunkStats, rechunk_corpus

    config = load_parsing_config(parsing_config)
    spans = token_spans_for(config.chunker.tokenizer, data_dir)
    engine = make_engine(database_url)
    try:
        factory = session_factory(engine)

        def report(stats: RechunkStats) -> None:
            print(json.dumps({"progress": stats.as_dict()}, sort_keys=True), file=sys.stderr)

        stats = rechunk_corpus(
            engine,
            spans,
            config.chunker,
            source_overlap_tokens=source_overlap,
            session_factory=factory,
            limit=limit,
            shards=shards,
            shard=shard,
            progress=report,
        )
        return {
            "policy": config.chunker.policy,
            "chunker_version": config.chunker.chunker_version,
            **stats.as_dict(),
        }
    finally:
        engine.dispose()


def run_corpus(
    *,
    config: Path,
    database_url: str,
    data_dir: Path,
    base_manifest: Path,
    parsing_config: Path,
    only: list[str] | None,
    keep_pdfs: bool,
    dry_run: bool,
    workers: int | None,
    download_workers: int | None = None,
    state: Path | None = None,
    fresh: bool = False,
) -> dict[str, object]:
    """Drive the venue-year plan: mirror, ingest, work, verify, sweep, repeat."""

    import subprocess
    import sys
    from datetime import UTC, datetime
    from uuid import uuid4

    import yaml

    from .corpus.ingest import ingest
    from .corpus.mirror import mirror_venue_year, resolve_revision, shard_repo
    from .corpus.sources.base import HttpxTransport, load_manifest
    from .corpus.venues import (
        CampaignPaths,
        VenueYear,
        default_state_path,
        load_venue_plan,
        parsed_checksums,
        pending_jobs,
        run_campaign,
        sweep_pdfs,
        venue_manifest,
    )

    plan, defaults = load_venue_plan(config)
    if only:
        wanted = set(only)
        plan = [item for item in plan if item.key in wanted or item.venue in wanted]
    parse_workers = workers or int(defaults.get("workers", 4))
    fetch_workers = download_workers or int(defaults.get("download_workers", 8))
    min_free_bytes = int(float(defaults.get("min_free_gb", 80)) * 1024**3)

    campaign_id = f"corpus-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    paths = CampaignPaths(data_dir=data_dir, run_dir=data_dir / "runs" / campaign_id)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    # The plan's progress outlives this invocation, so the run directory holds
    # only the manifests it generated.
    state_path = Path(state) if state else default_state_path(data_dir, config)
    if fresh and state_path.exists():
        state_path.unlink()
    engine = make_engine(database_url)
    transport = HttpxTransport()
    base = load_manifest(base_manifest)
    revisions: dict[str, str] = {}

    def revision(venue: str) -> str:
        if venue not in revisions:
            revisions[venue] = resolve_revision(shard_repo(venue))
        return revisions[venue]

    def mirror(item: VenueYear) -> dict[str, object]:
        return mirror_venue_year(
            paths.registry,
            venue=item.venue,
            year=item.year,
            mirror_root=paths.mirror_root,
            revision=revision(item.venue),
            workers=fetch_workers,
            min_free_bytes=min_free_bytes,
            out=paths.index(item.venue, item.year),
        )

    def enqueue(item: VenueYear) -> str:
        manifest = venue_manifest(
            base,
            venue=item.venue,
            year=item.year,
            index=paths.index(item.venue, item.year),
            mirror_root=paths.mirror_root,
            pdf_revision=revision(item.venue),
        )
        target = paths.manifest(item.venue, item.year)
        target.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return ingest(target, None, engine=engine, transport=transport)

    def work(item: VenueYear, run_id: str) -> dict[str, object]:
        manifest = paths.manifest(item.venue, item.year)
        command = [
            sys.executable,
            "-m",
            "copilot.cli",
            "worker",
            "run",
            "--database-url",
            database_url,
            "--manifest",
            str(manifest),
            "--parsing-config",
            str(parsing_config),
            "--staging-dir",
            str(data_dir / "sources"),
            "--data-dir",
            str(data_dir),
        ]
        processes = [
            subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                [*command, "--worker-id", f"{item.venue.lower()}{item.year}-{index}"]
            )
            for index in range(parse_workers)
        ]
        for process in processes:
            process.wait()
        unfinished, failed_jobs = pending_jobs(engine, run_id)
        return {
            "workers": parse_workers,
            "unfinished": unfinished,
            "failed_jobs": failed_jobs,
            "parsed_sha256": parsed_checksums(engine),
        }

    def sweep(item: VenueYear, checksums: set[str]) -> dict[str, int]:
        return sweep_pdfs(
            paths.index(item.venue, item.year),
            checksums,
            paths.pdf_root,
            dry_run=dry_run,
        )

    try:
        migrate_database(engine)
        result = run_campaign(
            plan,
            mirror=mirror,
            ingest=enqueue,
            work=work,
            sweep=sweep,
            state_path=state_path,
            keep_pdfs=keep_pdfs,
        )
        return {
            "campaign_id": campaign_id,
            "run_dir": str(paths.run_dir),
            "state": str(state_path),
            **result,
        }
    finally:
        transport.close()
        engine.dispose()


def run_worker(
    *,
    database_url: str,
    staging_dir: Path,
    data_dir: Path,
    manifest: Path,
    parsing_config: Path,
    worker_id: str,
    once: bool,
    max_jobs: int | None,
) -> dict[str, object]:
    from .corpus.chunk import load_parsing_config
    from .corpus.ingest import IngestContext, build_handlers
    from .corpus.sources.base import HttpxTransport, load_manifest
    from .jobs.worker import Worker

    engine = make_engine(database_url)
    transport = HttpxTransport()
    try:
        migrate_database(engine)
        context = IngestContext(
            manifest=load_manifest(manifest),
            transport=transport,
            staging_dir=staging_dir,
            data_dir=data_dir,
            parsing_config=load_parsing_config(parsing_config),
        )
        worker = Worker(engine, build_handlers(context))
        if once:
            processed = 1 if worker.run_once(worker_id) else 0
        else:
            processed = worker.run_until_idle(worker_id, max_jobs=max_jobs)
        return {"worker_id": worker_id, "processed": processed}
    finally:
        transport.close()
        engine.dispose()


def _service_settings() -> tuple[str, str]:
    """The Qdrant URL and collection namespace this environment uses."""

    from .config import Settings

    settings = Settings()  # type: ignore[call-arg]
    return settings.qdrant_url, settings.qdrant_collection_prefix


def _directory_bytes(path: Path) -> int | None:
    """Disk actually allocated, as du reports it.

    Qdrant preallocates its WAL and mmap files sparsely: a 1,024-point
    collection is 847 MB of apparent size and 14 MB of allocated blocks.
    """

    if not path.is_dir():
        return None
    return sum(item.stat().st_blocks * 512 for item in path.rglob("*") if item.is_file())


def _host_headroom(data_dir: Path) -> dict[str, object]:
    """Free disk under the data root and available memory, recorded beside a build (§12)."""

    import shutil

    available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    return {"disk_free_bytes": shutil.disk_usage(data_dir).free, "mem_available_bytes": available}


def run_fetch_model(*, models: Path, data_dir: Path) -> dict[str, object]:
    """Capture the pinned embedder and reranker once, offline, and verify them."""

    from .models.embeddings import fetch_model, load_embedding_spec
    from .search.rerank import load_reranker_spec

    embedding = load_embedding_spec(models)
    reranker = load_reranker_spec(models)
    return {
        "embedding": {
            "identity": embedding.identity,
            "directory": str(fetch_model(embedding, data_dir)),
        },
        "reranker": {
            "model": f"{reranker.repo}@{reranker.revision}",
            "directory": str(fetch_model(reranker, data_dir)),
        },
        "verified": True,
    }


def run_build_index(
    *,
    manifest: Path,
    models: Path,
    data_dir: Path,
    database_url: str,
    qdrant_url: str | None,
    prefix: str | None,
    release: str | None,
    collections: list[str],
    device: str,
    limit: int | None,
    cache: bool,
) -> dict[str, object]:
    """Build one or both collections of a release. Never activates it."""

    import sys
    import time

    from qdrant_client import QdrantClient

    from .corpus.releases import load_release
    from .models.embeddings import TransformerEmbedding, load_embedding_spec
    from .search.index import build_index, load_index_config, wait_until_indexed

    default_url, default_prefix = _service_settings()
    url = qdrant_url or default_url
    namespace = prefix if prefix is not None else default_prefix
    path = _snapshot_path(manifest, data_dir, DEFAULT_ARTIFACTS_PATH)
    spec = load_embedding_spec(models)
    config = load_index_config(models)
    model = TransformerEmbedding(spec, data_dir, device=device)
    # Upserts of a few hundred 1024-d points can outlast the client's 5 s default.
    client = QdrantClient(url=url, timeout=300)
    engine = make_engine(database_url)
    last = [0.0]

    def report(state: dict[str, object]) -> None:
        now = time.monotonic()
        if now - last[0] >= 30 or state["points"] == state["total"]:
            last[0] = now
            print(json.dumps({"progress": state}, sort_keys=True), file=sys.stderr, flush=True)

    try:
        release_id = build_index(
            path,
            model,
            engine=engine,
            client=client,
            prefix=namespace,
            data_dir=data_dir,
            config=config,
            kinds=collections,
            release_id=release,
            limit=limit,
            cache=cache,
            progress=report,
        )
        record = load_release(engine, release_id)
        built: dict[str, object] = {}
        for kind in collections:
            name = record.collection(kind)
            status = wait_until_indexed(client, name)
            details = dict(record.counts[kind])
            details.pop("canaries", None)
            info = client.get_collection(name)
            details.update(
                status=status,
                indexed_vectors=info.indexed_vectors_count,
                storage_bytes=_directory_bytes(data_dir / "qdrant" / "collections" / name),
            )
            built[kind] = details
        return {
            "release": release_id,
            "model": model.identity,
            "device": model.device,
            "precision": model.precision,
            "collections": built,
            "host": _host_headroom(data_dir),
        }
    finally:
        client.close()
        engine.dispose()


def _release_validation(
    release: str, *, models: Path, database_url: str, qdrant_url: str | None
) -> list[str]:
    from qdrant_client import QdrantClient

    from .corpus.releases import load_release
    from .models.embeddings import load_embedding_spec
    from .search.index import release_problems

    spec = load_embedding_spec(models)
    client = QdrantClient(url=qdrant_url or _service_settings()[0], timeout=300)
    engine = make_engine(database_url)
    try:
        return release_problems(
            load_release(engine, release),
            client=client,
            model_identity=spec.identity,
            dimensions=spec.dimensions,
        )
    finally:
        client.close()
        engine.dispose()


def run_validate_index(
    *, release: str, models: Path, database_url: str, qdrant_url: str | None
) -> dict[str, object]:
    """Report whether a release would activate, and every reason it would not."""

    problems = _release_validation(
        release, models=models, database_url=database_url, qdrant_url=qdrant_url
    )
    return {"release": release, "valid": not problems, "problems": problems}


def run_activate(
    *, release: str, models: Path, database_url: str, qdrant_url: str | None
) -> dict[str, object]:
    """Switch the serving pointer to an explicitly named, fully validated release."""

    from qdrant_client import QdrantClient

    from .corpus.releases import activate_release, capture_release
    from .models.embeddings import load_embedding_spec
    from .search.index import index_validator

    spec = load_embedding_spec(models)
    client = QdrantClient(url=qdrant_url or _service_settings()[0], timeout=300)
    engine = make_engine(database_url)
    try:
        previous = capture_release(engine)
        activate_release(
            engine,
            release,
            validator=index_validator(
                client, model_identity=spec.identity, dimensions=spec.dimensions
            ),
        )
        return {"active": release, "previous": previous.id if previous else None}
    finally:
        client.close()
        engine.dispose()


def run_drop_index(
    *, release: str, database_url: str, qdrant_url: str | None, data_dir: Path
) -> dict[str, object]:
    """Delete a release that is not serving: its row, collections and statistics."""

    import shutil

    from qdrant_client import QdrantClient

    from .corpus.releases import drop_release

    client = QdrantClient(url=qdrant_url or _service_settings()[0], timeout=300)
    engine = make_engine(database_url)
    try:
        record = drop_release(engine, release)
        deleted = []
        for name in (record.paper_collection, record.chunk_collection):
            if client.collection_exists(name):
                client.delete_collection(name)
                deleted.append(name)
        statistics = data_dir / "indexes" / release
        if statistics.is_dir():
            shutil.rmtree(statistics)
        return {"dropped": release, "collections": deleted}
    finally:
        client.close()
        engine.dispose()


def run_compare_precision(
    *, manifest: Path, models: Path, data_dir: Path, database_url: str, sample: int, seed: int
) -> dict[str, object]:
    """CPU float32 against GPU reduced precision on the same texts (spec §7)."""

    import random
    import time

    import numpy as np
    from sqlalchemy import text

    from .models.embeddings import TransformerEmbedding, load_embedding_spec
    from .search.index import snapshot_papers

    spec = load_embedding_spec(models)
    path = _snapshot_path(manifest, data_dir, DEFAULT_ARTIFACTS_PATH)
    papers = [document.text for document in snapshot_papers(path)]
    chosen_papers = random.Random(seed).sample(papers, sample)
    engine = make_engine(database_url)
    try:
        with engine.connect() as connection:
            chunks = [
                str(row[0])
                for row in connection.execute(
                    text(
                        "select text from chunks tablesample bernoulli (0.05) repeatable (:seed)"
                        " order by id limit :sample"
                    ),
                    {"seed": seed, "sample": sample},
                )
            ]
    finally:
        engine.dispose()

    cpu = TransformerEmbedding(spec, data_dir, device="cpu")
    gpu = TransformerEmbedding(spec, data_dir, device="cuda")
    report: dict[str, object] = {
        "identity": spec.identity,
        "cpu": {"precision": cpu.precision},
        "gpu": {"precision": gpu.precision},
    }
    for kind, texts in (("papers", chosen_papers), ("chunks", chunks)):
        limit = spec.max_tokens[kind]
        timings = {}
        vectors = {}
        for name, model in (("cpu", cpu), ("gpu", gpu)):
            started = time.monotonic()
            vectors[name] = model.encode_array(texts, max_tokens=limit).vectors
            timings[name] = round(time.monotonic() - started, 2)
        cosine = np.sum(vectors["cpu"] * vectors["gpu"], axis=1)
        # Whether each text's ten nearest neighbours survive the precision change,
        # which is what a ranking actually depends on.
        neighbours = {}
        for name, matrix in vectors.items():
            similarity = matrix @ matrix.T
            np.fill_diagonal(similarity, -np.inf)
            neighbours[name] = np.argsort(-similarity, axis=1)[:, :10]
        overlaps = [
            len(set(a) & set(b)) / 10
            for a, b in zip(neighbours["cpu"], neighbours["gpu"], strict=True)
        ]
        report[kind] = {
            "texts": len(texts),
            "seconds": timings,
            "texts_per_second": {k: round(len(texts) / v, 1) for k, v in timings.items() if v},
            "cosine_min": float(cosine.min()),
            "cosine_mean": float(cosine.mean()),
            "cosine_p01": float(np.quantile(cosine, 0.01)),
            "neighbour_top10_overlap_mean": float(np.mean(overlaps)),
            "neighbour_top10_overlap_min": float(np.min(overlaps)),
        }
    return report


def run_compare_oracle(
    *, release: str, database_url: str, qdrant_url: str | None, sample: int, seed: int, limit: int
) -> dict[str, object]:
    """Served BM25 against the exact in-memory oracle over the same paper texts (spec §7)."""

    import random
    import statistics
    import time

    from qdrant_client import QdrantClient

    from .corpus.releases import load_release
    from .search.index import CANARY_SLACK, PAPERS, same_ranking, snapshot_papers
    from .search.lexical import BM25
    from .search.sparse import SparseRetriever

    engine = make_engine(database_url)
    client = QdrantClient(url=qdrant_url or _service_settings()[0], timeout=300)
    try:
        record = load_release(engine, release)
        build = record.counts[PAPERS]
        documents = {
            document.id: document.text
            for document in snapshot_papers(
                Path(str(record.counts["manifest"])), limit=build.get("limit")
            )
        }
        oracle = BM25()
        oracle.fit(documents)
        titles = [text.split("\n", 1)[0] for text in documents.values()]
        queries = random.Random(seed).sample(titles, min(sample, len(titles)))
        retriever = SparseRetriever(engine, client)
        identical = 0
        overlaps: list[float] = []
        worst = 0.0
        latencies: list[float] = []
        for query in queries:
            expected = oracle.search(query, limit)
            started = time.monotonic()
            # Slack past the limit, so a tie that orders differently at the
            # boundary is found rather than counted as a miss.
            served = retriever.search(query, None, limit + CANARY_SLACK, release)
            latencies.append(time.monotonic() - started)
            if same_ranking([list(pair) for pair in expected], [list(pair) for pair in served]):
                identical += 1
            wanted = {doc_id for doc_id, _ in expected}
            top = {doc_id for doc_id, _ in served[:limit]}
            overlaps.append(len(wanted & top) / max(1, len(wanted)))
            scores = dict(served)
            for doc_id, score in expected:
                if doc_id in scores and score:
                    worst = max(worst, abs(scores[doc_id] - score) / score)
        latencies.sort()
        return {
            "release": release,
            "documents": len(documents),
            "queries": len(queries),
            "limit": limit,
            "identical_rankings": identical,
            "overlap_mean": statistics.fmean(overlaps),
            "overlap_min": min(overlaps),
            "max_relative_score_error": worst,
            "latency_ms": {
                "p50": round(1000 * latencies[len(latencies) // 2], 1),
                "p95": round(1000 * latencies[int(len(latencies) * 0.95)], 1),
            },
        }
    finally:
        client.close()
        engine.dispose()


def run_compare_reranker_precision(
    *,
    release: str,
    models: Path,
    data_dir: Path,
    database_url: str,
    qdrant_url: str | None,
    sample: int,
    seed: int,
    depth: int,
) -> dict[str, object]:
    """The reranker on CPU float32 against GPU reduced precision (spec §7).

    Candidates are what it will really see: the BM25 top ``depth`` for sampled
    paper titles. What matters is whether the order survives, so the report
    is per-query top-10 overlap and rank agreement, beside the raw score gap.
    """

    import random
    import statistics
    import time

    import numpy as np
    from qdrant_client import QdrantClient

    from .corpus.releases import load_release
    from .search.index import snapshot_papers
    from .search.rerank import CrossEncoderReranker, load_reranker_spec
    from .search.service import PaperStore
    from .search.sparse import SparseRetriever

    spec = load_reranker_spec(models)
    engine = make_engine(database_url)
    client = QdrantClient(url=qdrant_url or _service_settings()[0], timeout=300)
    try:
        record = load_release(engine, release)
        titles = [
            document.text.split("\n", 1)[0]
            for document in snapshot_papers(Path(str(record.counts["manifest"])))
        ]
        queries = random.Random(seed).sample(titles, sample)
        sparse = SparseRetriever(engine, client)
        store = PaperStore(engine)
        pairs = []
        for query in queries:
            ids = [paper_id for paper_id, _ in sparse.search(query, None, depth, release)]
            rows = store.load(ids)
            pairs.append((query, [rows[paper_id].text for paper_id in ids if paper_id in rows]))
        results: dict[str, list[list[float]]] = {}
        timings: dict[str, float] = {}
        precisions: dict[str, str] = {}
        for device in ("cpu", "cuda"):
            reranker = CrossEncoderReranker(spec, data_dir, device=device)
            precisions[device] = reranker.precision
            reranker.score("warm up", ["warm up"])
            started = time.monotonic()
            results[device] = [reranker.score(query, texts) for query, texts in pairs]
            timings[device] = time.monotonic() - started
            del reranker
    finally:
        client.close()
        engine.dispose()

    overlaps: list[float] = []
    agreements: list[float] = []
    gaps: list[float] = []
    for cpu_scores, gpu_scores in zip(results["cpu"], results["cuda"], strict=True):
        cpu = np.asarray(cpu_scores)
        gpu = np.asarray(gpu_scores)
        gaps.append(float(np.max(np.abs(cpu - gpu))))
        top_cpu = set(np.argsort(-cpu)[:10].tolist())
        top_gpu = set(np.argsort(-gpu)[:10].tolist())
        overlaps.append(len(top_cpu & top_gpu) / 10)
        # Share of candidate pairs both precisions put in the same order.
        order = np.sign(cpu[:, None] - cpu[None, :]) == np.sign(gpu[:, None] - gpu[None, :])
        agreements.append(float(order[np.triu_indices(len(cpu), 1)].mean()))
    pair_count = sum(len(texts) for _, texts in pairs)
    return {
        "model": f"{spec.repo}@{spec.revision}",
        "pair_max_tokens": spec.pair_max_tokens,
        "precision": precisions,
        "queries": len(pairs),
        "pairs": pair_count,
        "pairs_per_second": {k: round(pair_count / v, 1) for k, v in timings.items()},
        "top10_overlap_mean": statistics.fmean(overlaps),
        "top10_overlap_min": min(overlaps),
        "pairwise_order_agreement_mean": statistics.fmean(agreements),
        "pairwise_order_agreement_min": min(agreements),
        "max_abs_logit_gap": max(gaps),
    }


def _litsearch_queries(data_dir: Path, split: str, dataset: Path) -> list[str]:
    """In-domain LitSearch query text for one split, read from DATA_DIR only.

    The repository copy carries ids and splits but no query text: LitSearch
    declares no license, so the text stays local and never lands in a report.
    """

    from .evaluation.datasets import hydrate, read_dataset

    source = data_dir / "benchmarks" / "litsearch-dataset" / "queries.jsonl"
    texts = {
        str(row["query_id"]): str(row["query"])
        for row in (json.loads(line) for line in source.read_text(encoding="utf-8").splitlines())
        if row.get("query")
    }
    frozen = hydrate(read_dataset(dataset), texts)
    in_split = {record.query_id for record in frozen.splits if record.split == split}
    return [
        record.query
        for record in frozen.queries
        if record.query_id in in_split and record.in_domain
    ]


def run_search_pilot(
    *,
    models: Path,
    search_config: Path,
    data_dir: Path,
    database_url: str,
    qdrant_url: str | None,
    mode: str,
    queries: list[str],
    split: str,
    dataset: Path,
    limit_queries: int | None,
    traces: str | None,
) -> dict[str, object]:
    """Run the real service over a query set and report every stage's timing.

    Handwritten ``--query`` strings come back with their top results, to be
    quoted as trace examples. LitSearch queries report aggregates only, and
    their traces, if kept, are written under DATA_DIR, never the repository.
    """

    import statistics
    import time
    from collections import Counter

    from qdrant_client import QdrantClient

    from .contracts import PaperFilters, SearchRequest
    from .models.embeddings import TransformerEmbedding, load_embedding_spec
    from .search.dense import DenseRetriever
    from .search.rerank import CrossEncoderReranker, load_reranker_spec
    from .search.service import (
        SearchService,
        SearchUnavailable,
        ThreadedStageRunner,
        load_search_config,
    )
    from .search.sparse import SparseRetriever

    handwritten = bool(queries)
    texts = queries or _litsearch_queries(data_dir, split, dataset)
    if limit_queries is not None:
        texts = texts[:limit_queries]
    embedding = load_embedding_spec(models)
    engine = make_engine(database_url)
    client = QdrantClient(url=qdrant_url or _service_settings()[0], timeout=60)
    model = TransformerEmbedding(embedding, data_dir)
    reranker = CrossEncoderReranker(load_reranker_spec(models), data_dir)
    runner = ThreadedStageRunner()
    service = SearchService(
        engine=engine,
        lexical=SparseRetriever(engine, client),
        dense=DenseRetriever(engine, client, model, max_tokens=embedding.max_tokens["papers"]),
        reranker=reranker,
        config=load_search_config(search_config),
        runner=runner,
    )

    def request(query: str) -> SearchRequest:
        return SearchRequest.model_validate(
            {"query": query, "mode": mode, "filters": PaperFilters(), "limit": 20}
        )

    try:
        # Cold start is reported apart (spec §12): first-call CUDA work and the
        # BM25 statistics load belong to the process, not to any one query.
        started = time.monotonic()
        service.search_with_trace(request("retrieval augmented generation"))
        cold = time.monotonic() - started

        stage_seconds: dict[str, list[float]] = {}
        totals: list[float] = []
        warnings: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        degraded = 0
        examples = []
        written = []
        for query in texts:
            tick = time.monotonic()
            try:
                response, trace = service.search_with_trace(request(query))
            except SearchUnavailable as error:
                # A failed query stays in the denominator, as a failure.
                failures[error.code] += 1
                totals.append(time.monotonic() - tick)
                continue
            totals.append(time.monotonic() - tick)
            warnings.update(trace.warnings)
            degraded += int(trace.degraded)
            for stage, detail in trace.stages.items():
                if "seconds" in detail:
                    stage_seconds.setdefault(stage, []).append(float(detail["seconds"]))
            written.append({"query": query, "trace": trace.__dict__})
            if handwritten:
                examples.append(
                    {
                        "query": query,
                        "degraded": response.degraded,
                        "warnings": response.warnings,
                        "top": [
                            {
                                "rank": item.rank,
                                "title": response.papers[str(item.paper_id)].title,
                                "venue": response.papers[str(item.paper_id)].venue,
                                "year": response.papers[str(item.paper_id)].year,
                                "scores": {k: round(v, 4) for k, v in item.scores.items()},
                            }
                            for item in response.items[:5]
                        ],
                        "stage_ms": {
                            stage: round(1000 * float(detail["seconds"]), 1)
                            for stage, detail in trace.stages.items()
                            if "seconds" in detail
                        },
                    }
                )
    finally:
        runner.close()
        client.close()
        engine.dispose()

    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "p50_ms": round(1000 * statistics.median(ordered), 1),
            "p95_ms": round(1000 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
            "max_ms": round(1000 * ordered[-1], 1),
        }

    traces_path = None
    if traces:
        traces_path = data_dir / "runs" / f"{traces}.jsonl"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with traces_path.open("w", encoding="utf-8") as handle:
            for row in written:
                handle.write(json.dumps(row, default=str) + "\n")
    return {
        "mode": mode,
        "source": "handwritten" if handwritten else f"litsearch:{split}:in_domain",
        "queries": len(texts),
        "models": {
            "embedding": model.identity,
            "reranker": reranker.identity,
            "precision": {"embedding": model.precision, "reranker": reranker.precision},
        },
        "cold_start_ms": round(1000 * cold, 1),
        "total": summary(totals) if totals else None,
        "stages": {stage: summary(values) for stage, values in stage_seconds.items()},
        "degraded_requests": degraded,
        "warnings": dict(warnings),
        "failures": dict(failures),
        "traces": str(traces_path) if traces_path else None,
        "examples": examples,
    }


def run_litsearch_snapshot(*, out: Path, source: Path | None, data_dir: Path) -> dict[str, Any]:
    """Write LitSearch's corpus as a snapshot under the exports directory, offline."""

    import yaml

    from .evaluation.litsearch import local_paths, write_litsearch_snapshot

    evaluation = yaml.safe_load(DEFAULT_EVALUATION_PATH.read_text(encoding="utf-8"))
    pinned = evaluation["benchmarks"]["litsearch"]
    root = source or data_dir / "benchmarks" / "litsearch"
    destination = _snapshot_path(out, data_dir, DEFAULT_ARTIFACTS_PATH)
    manifest = write_litsearch_snapshot(
        local_paths(root, pinned["checksums"]),
        destination,
        run_id=destination.name,
        revision=pinned["revision"],
    )
    return {
        "out": str(destination / "manifest.json"),
        **{key: manifest[key] for key in ("counts", "digests", "golds")},
    }


def run_orb_fetch(*, data_dir: Path, revision: str | None) -> dict[str, Any]:
    """Fetch Open RAG Bench once, at a pinned commit, under DATA_DIR/benchmarks/orb."""

    from .evaluation.orb import ORB_REVISION, OrbPaths, fetch_orb

    paths = OrbPaths(data_dir / "benchmarks" / "orb")
    checksums = fetch_orb(paths, revision=revision or ORB_REVISION)
    return {"root": str(paths.root), "revision": revision or ORB_REVISION, "checksums": checksums}


def run_orb_dataset(*, out: Path, data_dir: Path, database_url: str | None) -> dict[str, Any]:
    """Freeze the text slice twice: text-free in the repository, hydratable under DATA_DIR."""

    from .evaluation.orb import (
        OrbPaths,
        dataset_counts,
        load_orb,
        paper_ids_for,
        sample_qa,
        text_slice,
        write_orb_dataset,
    )

    paths = OrbPaths(data_dir / "benchmarks" / "orb")
    queries = text_slice(load_orb(paths))
    assignment = sample_qa(queries)
    paper_ids: dict[str, str] = {}
    if database_url:
        engine = make_engine(database_url)
        try:
            paper_ids = paper_ids_for({query.doc_id for query in queries}, engine)
        finally:
            engine.dispose()
    write_orb_dataset(queries, assignment, out, include_text=False, paper_ids=paper_ids)
    write_orb_dataset(queries, assignment, paths.dataset, include_text=True, paper_ids=paper_ids)
    return {"out": str(out), "hydratable": str(paths.dataset), **dataset_counts(out)}


def run_eval_smoke(fixture: Path, *, update: bool) -> dict[str, Any]:
    """The synthetic smoke set against its frozen outputs; ``update`` re-freezes them."""

    from .evaluation.retrieval import check_smoke, run_smoke

    first = run_smoke(fixture)
    second = run_smoke(fixture)
    expected_path = fixture / "expected.json"
    if update:
        expected_path.write_text(json.dumps(first, indent=2, sort_keys=True) + "\n")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    check = check_smoke(first, expected)
    reproducible = first == second
    return {
        **check,
        "passed": check["passed"] and reproducible,
        "reproducible": reproducible,
        "metrics": {name: entry["metrics"] for name, entry in first.items()},
    }


def run_export_schema(out: Path) -> dict[str, object]:
    """The OpenAPI schema as reviewed JSON: sorted keys, so a diff shows real changes."""

    from .app import openapi_schema

    schema = openapi_schema()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "out": str(out),
        "version": schema["info"]["version"],
        "paths": sorted(schema["paths"]),
    }


def run_llm_spend(*, database_url: str, day: str | None, run_id: str | None) -> dict[str, Any]:
    """Hosted-LLM calls, tokens and cost by model, for one UTC day or one run."""

    from datetime import date

    from .models.spend import spend_report

    engine = make_engine(database_url)
    try:
        return spend_report(engine, day=date.fromisoformat(day) if day else None, run_id=run_id)
    finally:
        engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="copilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixtures = subparsers.add_parser("fixtures")
    fixture_commands = fixtures.add_subparsers(dest="fixture_command", required=True)
    load = fixture_commands.add_parser("load")
    load.add_argument("fixture_path", type=Path, nargs="?", default=DEFAULT_FIXTURE_PATH)
    load.add_argument("--database-url", required=True)
    load.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="raw source artifact directory; defaults to <DATA_DIR>/sources",
    )

    corpus = subparsers.add_parser("corpus")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    ingest = corpus_commands.add_parser("ingest", help="list a venue-year and enqueue pilot jobs")
    ingest.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    ingest.add_argument("--limit", type=int, default=None)
    ingest.add_argument("--database-url", default=None, help="defaults to DATABASE_URL")
    ingest.add_argument("--staging-dir", type=Path, default=None)

    mirror = corpus_commands.add_parser(
        "mirror-index", help="build a venue-year index from the registry parquet"
    )
    mirror.add_argument("--venue", required=True)
    mirror.add_argument("--year", type=int, required=True)
    mirror.add_argument("--registry", type=Path, default=None)
    mirror.add_argument("--pdf-root", type=Path, default=None)
    mirror.add_argument("--out", type=Path, default=None)
    mirror.add_argument("--data-dir", type=Path, default=None)

    capture = corpus_commands.add_parser(
        "mirror", help="download one venue-year's PDFs and write its index"
    )
    capture.add_argument("--venue", required=True)
    capture.add_argument("--year", type=int, required=True)
    capture.add_argument("--registry", type=Path, default=None)
    capture.add_argument(
        "--revision", default=None, help="PDF shard commit; resolved and recorded when omitted"
    )
    capture.add_argument("--workers", type=int, default=8)
    capture.add_argument("--min-free-gb", type=float, default=80.0)
    capture.add_argument("--data-dir", type=Path, default=None)

    corpus_run = corpus_commands.add_parser(
        "run", help="mirror, ingest, parse and sweep each venue-year in the plan"
    )
    corpus_run.add_argument("--config", type=Path, default=Path("configs/venues.yaml"))
    corpus_run.add_argument("--base-manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    corpus_run.add_argument("--parsing-config", type=Path, default=DEFAULT_PARSING_PATH)
    corpus_run.add_argument("--database-url", default=None, help="defaults to DATABASE_URL")
    corpus_run.add_argument("--data-dir", type=Path, default=None)
    corpus_run.add_argument(
        "--only", action="append", default=None, help="limit to VENUE or VENUE:YEAR; repeatable"
    )
    corpus_run.add_argument("--workers", type=int, default=None, help="parse workers")
    corpus_run.add_argument(
        "--download-workers", type=int, default=None, help="concurrent shard downloads"
    )
    corpus_run.add_argument("--keep-pdfs", action="store_true", help="never delete source PDFs")
    corpus_run.add_argument(
        "--state", type=Path, default=None, help="progress file; defaults to <plan>-state.json"
    )
    corpus_run.add_argument(
        "--fresh", action="store_true", help="discard recorded progress and start the plan over"
    )
    corpus_run.add_argument(
        "--dry-run", action="store_true", help="report what the sweep would delete"
    )

    export = corpus_commands.add_parser("export", help="write an immutable corpus snapshot")
    export.add_argument("--run", required=True, dest="run_id")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--database-url", default=None)
    export.add_argument("--data-dir", type=Path, default=None)
    export.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_PATH)

    validate = corpus_commands.add_parser("validate", help="verify a snapshot's checksums")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--data-dir", type=Path, default=None)
    validate.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_PATH)

    restore = corpus_commands.add_parser(
        "restore", help="restore a snapshot into an empty database"
    )
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--database-url", required=True, help="must name an empty database")
    restore.add_argument("--data-dir", type=Path, default=None)
    restore.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_PATH)

    coverage = corpus_commands.add_parser("coverage", help="print venue-year coverage")
    coverage.add_argument("--database-url", default=None)
    coverage.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_PATH)

    compare = corpus_commands.add_parser(
        "compare-chunkers", help="measure both chunking policies over the same PDFs"
    )
    compare.add_argument("--source", type=Path, required=True, help="directory of PDFs")
    compare.add_argument("--sample", type=int, default=40)
    compare.add_argument("--seed", type=int, default=2026)
    compare.add_argument("--data-dir", type=Path, default=None)
    compare.add_argument("--parsing-config", type=Path, default=DEFAULT_PARSING_PATH)

    rechunk = corpus_commands.add_parser(
        "rechunk", help="re-chunk stored documents under the configured policy"
    )
    rechunk.add_argument("--database-url", default=None)
    rechunk.add_argument("--data-dir", type=Path, default=None)
    rechunk.add_argument("--parsing-config", type=Path, default=DEFAULT_PARSING_PATH)
    rechunk.add_argument(
        "--source-overlap", type=int, default=60, help="overlap the stored chunks were cut with"
    )
    rechunk.add_argument("--limit", type=int, default=None)
    rechunk.add_argument("--shards", type=int, default=1)
    rechunk.add_argument("--shard", type=int, default=0)

    activate = corpus_commands.add_parser(
        "activate", help="switch the serving pointer to a validated release"
    )
    activate.add_argument("--release", required=True, help="explicit release id; never inferred")
    activate.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    activate.add_argument("--database-url", default=None)
    activate.add_argument("--qdrant-url", default=None)

    search = subparsers.add_parser("search")
    search_commands = search.add_subparsers(dest="search_command", required=True)
    fetch_model = search_commands.add_parser(
        "fetch-model", help="capture the pinned embedder and reranker offline and verify them"
    )
    fetch_model.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    fetch_model.add_argument("--data-dir", type=Path, default=None)

    build = search_commands.add_parser(
        "build-index", help="build a release's collections from a snapshot; never activates"
    )
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    build.add_argument("--release", default=None, help="defaults to the snapshot's run id")
    build.add_argument(
        "--collections",
        default="papers,chunks",
        help="comma-separated: papers, chunks (built in the order given)",
    )
    build.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    build.add_argument(
        "--limit", type=int, default=None, help="measurement build over the first N papers"
    )
    build.add_argument("--no-cache", action="store_true", help="re-encode every batch")
    build.add_argument("--database-url", default=None)
    build.add_argument("--qdrant-url", default=None)
    build.add_argument("--prefix", default=None, help="collection namespace; defaults to env")
    build.add_argument("--data-dir", type=Path, default=None)

    check = search_commands.add_parser(
        "validate-index", help="report whether a release would activate, and why not"
    )
    check.add_argument("--release", required=True)
    check.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    check.add_argument("--database-url", default=None)
    check.add_argument("--qdrant-url", default=None)

    drop = search_commands.add_parser(
        "drop-index", help="delete a release that is not serving, with its collections"
    )
    drop.add_argument("--release", required=True)
    drop.add_argument("--database-url", default=None)
    drop.add_argument("--qdrant-url", default=None)
    drop.add_argument("--data-dir", type=Path, default=None)

    precision = search_commands.add_parser(
        "compare-precision", help="CPU float32 against GPU reduced precision on the same texts"
    )
    precision.add_argument("--model", choices=["embedding", "reranker"], default="embedding")
    precision.add_argument("--manifest", type=Path, default=None, help="embedding: texts from")
    precision.add_argument("--release", default=None, help="reranker: BM25 candidates from")
    precision.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    precision.add_argument(
        "--sample", type=int, default=None, help="texts (embedding, 256) or queries (reranker, 10)"
    )
    precision.add_argument("--depth", type=int, default=50, help="reranker candidates per query")
    precision.add_argument("--seed", type=int, default=42)
    precision.add_argument("--database-url", default=None)
    precision.add_argument("--qdrant-url", default=None)
    precision.add_argument("--data-dir", type=Path, default=None)

    pilot = search_commands.add_parser(
        "pilot", help="run the search service over a query set with stage timing"
    )
    pilot.add_argument(
        "--mode", default="hybrid_rerank", choices=["bm25", "dense", "hybrid", "hybrid_rerank"]
    )
    pilot.add_argument("--query", action="append", default=[], help="handwritten query; repeatable")
    pilot.add_argument("--split", default="development", choices=["development", "validation"])
    pilot.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    pilot.add_argument("--limit-queries", type=int, default=None)
    pilot.add_argument("--traces", default=None, help="write traces to DATA_DIR/runs/NAME.jsonl")
    pilot.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    pilot.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_PATH)
    pilot.add_argument("--database-url", default=None)
    pilot.add_argument("--qdrant-url", default=None)
    pilot.add_argument("--data-dir", type=Path, default=None)

    oracle = search_commands.add_parser(
        "compare-oracle", help="served BM25 against the exact in-memory oracle"
    )
    oracle.add_argument("--release", required=True)
    oracle.add_argument("--sample", type=int, default=500)
    oracle.add_argument("--seed", type=int, default=42)
    oracle.add_argument("--limit", type=int, default=100)
    oracle.add_argument("--database-url", default=None)
    oracle.add_argument("--qdrant-url", default=None)

    api = subparsers.add_parser("api")
    api_commands = api.add_subparsers(dest="api_command", required=True)
    export_schema = api_commands.add_parser(
        "export-schema", help="write the public OpenAPI schema, sorted for review"
    )
    export_schema.add_argument("--out", type=Path, required=True)

    llm = subparsers.add_parser("llm")
    llm_commands = llm.add_subparsers(dest="llm_command", required=True)
    spend = llm_commands.add_parser(
        "spend", help="hosted-LLM calls, tokens and cost by model; never prompt text"
    )
    spend_scope = spend.add_mutually_exclusive_group()
    spend_scope.add_argument("--day", default=None, help="UTC day, YYYY-MM-DD; default today")
    spend_scope.add_argument("--run", default=None, help="an evaluation run id")
    spend.add_argument("--database-url", default=None, help="defaults to DATABASE_URL")

    evaluation = subparsers.add_parser("eval")
    evaluation_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    validate_dataset_cmd = evaluation_commands.add_parser(
        "validate-dataset", help="refuse a dataset that leaks between splits"
    )
    validate_dataset_cmd.add_argument("--path", type=Path, required=True)
    retrieval = evaluation_commands.add_parser(
        "retrieval", help="run an experiment's variants over one frozen split"
    )
    retrieval.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_PATH)
    retrieval.add_argument(
        "--split", required=True, choices=["development", "validation", "test", "retrieval"]
    )
    retrieval.add_argument("--out", type=Path, required=True)
    retrieval.add_argument(
        "--locked-test",
        action="store_true",
        help="the test split, once, for a release decision; never for tuning",
    )
    retrieval.add_argument(
        "--limit-queries", type=int, default=None, help="a partial run, recorded as such"
    )
    retrieval.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help="required with an LLM variant: refused if its worst case exceeds this, "
        "stopped if its real spend does",
    )
    retrieval.add_argument("--database-url", default=None)
    retrieval.add_argument("--qdrant-url", default=None)
    retrieval.add_argument("--data-dir", type=Path, default=None)
    smoke = evaluation_commands.add_parser(
        "smoke", help="the four baselines on the synthetic fixture, against frozen outputs"
    )
    smoke.add_argument("--fixture", type=Path, default=DEFAULT_SMOKE_PATH)
    smoke.add_argument(
        "--update", action="store_true", help="re-freeze the outputs after an intended change"
    )
    compare = evaluation_commands.add_parser(
        "compare", help="regression report: a candidate run against a baseline run"
    )
    compare.add_argument("--baseline", type=Path, required=True, help="a run's metrics.json")
    compare.add_argument("--candidate", type=Path, required=True, help="a run's metrics.json")
    litsearch_snapshot = evaluation_commands.add_parser(
        "litsearch-snapshot",
        help="LitSearch's corpus_clean as a corpus snapshot, for its own release (E3)",
    )
    litsearch_snapshot.add_argument(
        "--out", type=Path, default=Path("litsearch-v1"), help="relative to the exports directory"
    )
    litsearch_snapshot.add_argument(
        "--source", type=Path, default=None, help="defaults to DATA_DIR/benchmarks/litsearch"
    )
    litsearch_snapshot.add_argument("--data-dir", type=Path, default=None)
    orb_fetch = evaluation_commands.add_parser(
        "orb-fetch", help="download Open RAG Bench at its pinned revision, offline"
    )
    orb_fetch.add_argument("--revision", default=None, help="a 40-hex commit; default pinned")
    orb_fetch.add_argument("--data-dir", type=Path, default=None)
    orb_dataset = evaluation_commands.add_parser(
        "orb-dataset", help="freeze ORB's text slice: ids and labels in git, text under DATA_DIR"
    )
    orb_dataset.add_argument("--out", type=Path, default=Path("data/fixtures/orb"))
    orb_dataset.add_argument(
        "--database-url",
        default=None,
        help="when given, gold arXiv ids are resolved to paper ids in that corpus",
    )
    orb_dataset.add_argument("--data-dir", type=Path, default=None)
    report = evaluation_commands.add_parser("report", help="re-render an experiment's report.md")
    report.add_argument("--out", type=Path, required=True)
    gaps = evaluation_commands.add_parser(
        "gaps", help="where recorded runs lose the gold paper, offline; never the test split"
    )
    gaps.add_argument("--run", type=Path, required=True, help="an experiment's output directory")
    gaps.add_argument(
        "--split",
        action="append",
        default=None,
        choices=["development", "validation"],
        help="repeatable; defaults to every recorded diagnostic split",
    )
    gaps.add_argument("--focus", default="hybrid_rerank", help="variant to break down by slice")
    gaps.add_argument("--depth", type=int, default=50, help="the cut a gold must survive")

    worker = subparsers.add_parser("worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    run = worker_commands.add_parser("run", help="process leased jobs")
    run.add_argument("--worker-id", default=f"{socket.gethostname() or 'worker'}-{os.getpid()}")
    run.add_argument("--once", action="store_true", help="process at most one job")
    run.add_argument("--max-jobs", type=int, default=None)
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    run.add_argument("--parsing-config", type=Path, default=DEFAULT_PARSING_PATH)
    run.add_argument("--database-url", default=None, help="defaults to DATABASE_URL")
    run.add_argument("--staging-dir", type=Path, default=None)
    run.add_argument(
        "--data-dir", type=Path, default=None, help="local data root; defaults to DATA_DIR"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "fixtures" and args.fixture_command == "load":
        result: dict[str, object] = dict(
            load_fixtures(
                args.fixture_path,
                database_url=args.database_url,
                staging_dir=_staging_dir(args.staging_dir, parser),
            )
        )
    elif args.command == "eval" and args.eval_command == "retrieval":
        from .evaluation.report import render_report
        from .evaluation.retrieval import run_retrieval

        metrics = run_retrieval(
            args.config,
            args.split,
            args.out,
            data_dir=_data_dir(args.data_dir, parser),
            database_url=_database_url(args.database_url),
            qdrant_url=args.qdrant_url or _service_settings()[0],
            locked_test=args.locked_test,
            limit_queries=args.limit_queries,
            max_spend_usd=args.max_spend_usd,
            llm_daily_cap_usd=_llm_daily_cap(),
        )
        result = {
            "run_id": metrics["manifest"]["run_id"],
            "report": str(render_report(args.out)),
            "variants": {
                name: {
                    key: round(entry["summary"][key]["mean"], 4)
                    for key in ("recall@10", "recall@50", "ndcg@10", "mrr@10")
                }
                | {"p95_ms": entry["summary"]["latency"]["p95_ms"]}
                for name, entry in metrics["variants"].items()
            },
            "decision": (metrics.get("decision") or {}).get("chosen"),
        }
    elif args.command == "eval" and args.eval_command == "smoke":
        result = run_eval_smoke(args.fixture, update=args.update)
        if not result["passed"]:
            print(json.dumps(result, sort_keys=True, default=str))
            return 1
    elif args.command == "eval" and args.eval_command == "compare":
        from .evaluation.regression import compare_runs

        result = compare_runs(
            json.loads(args.baseline.read_text(encoding="utf-8")),
            json.loads(args.candidate.read_text(encoding="utf-8")),
        )
        if result["regressed"]:
            print(json.dumps(result, sort_keys=True, default=str))
            return 1
    elif args.command == "eval" and args.eval_command == "litsearch-snapshot":
        result = run_litsearch_snapshot(
            out=args.out, source=args.source, data_dir=_data_dir(args.data_dir, parser)
        )
    elif args.command == "eval" and args.eval_command == "report":
        from .evaluation.report import render_report

        result = {"report": str(render_report(args.out))}
    elif args.command == "eval" and args.eval_command == "gaps":
        from .evaluation.gaps import render_gaps

        path = render_gaps(args.run, splits=args.split, focus=args.focus, depth=args.depth)
        result = {"report": str(path), "data": str(path.with_suffix(".json"))}
    elif args.command == "eval" and args.eval_command == "validate-dataset":
        from .evaluation.datasets import validate_dataset

        result = dict(validate_dataset(args.path))
    elif args.command == "eval" and args.eval_command == "orb-fetch":
        result = run_orb_fetch(data_dir=_data_dir(args.data_dir, parser), revision=args.revision)
    elif args.command == "eval" and args.eval_command == "orb-dataset":
        result = run_orb_dataset(
            out=args.out,
            data_dir=_data_dir(args.data_dir, parser),
            database_url=args.database_url,
        )
    elif args.command == "corpus" and args.corpus_command == "ingest":
        result = run_ingest(
            args.manifest,
            args.limit,
            database_url=_database_url(args.database_url),
            staging_dir=_staging_dir(args.staging_dir, parser),
        )
    elif args.command == "corpus" and args.corpus_command == "mirror-index":
        result = run_mirror_index(
            registry=args.registry,
            venue=args.venue,
            year=args.year,
            pdf_root=args.pdf_root,
            out=args.out,
            data_dir=_data_dir(args.data_dir, parser),
        )
    elif args.command == "corpus" and args.corpus_command == "mirror":
        result = run_mirror(
            venue=args.venue,
            year=args.year,
            registry=args.registry,
            revision=args.revision,
            workers=args.workers,
            min_free_gb=args.min_free_gb,
            data_dir=_data_dir(args.data_dir, parser),
        )
    elif args.command == "corpus" and args.corpus_command == "run":
        result = run_corpus(
            config=args.config,
            database_url=_database_url(args.database_url),
            data_dir=_data_dir(args.data_dir, parser),
            base_manifest=args.base_manifest,
            parsing_config=args.parsing_config,
            only=args.only,
            keep_pdfs=args.keep_pdfs,
            dry_run=args.dry_run,
            workers=args.workers,
            download_workers=args.download_workers,
            state=args.state,
            fresh=args.fresh,
        )
    elif args.command == "corpus" and args.corpus_command == "export":
        result = run_export(
            run_id=args.run_id,
            out=args.out,
            database_url=_database_url(args.database_url),
            data_dir=_data_dir(args.data_dir, parser),
            artifacts=args.artifacts,
        )
    elif args.command == "corpus" and args.corpus_command == "validate":
        result = run_validate(
            manifest=args.manifest,
            data_dir=_data_dir(args.data_dir, parser),
            artifacts=args.artifacts,
        )
    elif args.command == "corpus" and args.corpus_command == "restore":
        result = run_restore(
            manifest=args.manifest,
            database_url=args.database_url,
            data_dir=_data_dir(args.data_dir, parser),
            artifacts=args.artifacts,
        )
    elif args.command == "corpus" and args.corpus_command == "coverage":
        result = run_coverage(
            database_url=_database_url(args.database_url), artifacts=args.artifacts
        )
    elif args.command == "corpus" and args.corpus_command == "compare-chunkers":
        result = run_compare_chunkers(
            source=args.source,
            sample=args.sample,
            seed=args.seed,
            data_dir=_data_dir(args.data_dir, parser),
            parsing_config=args.parsing_config,
        )
    elif args.command == "corpus" and args.corpus_command == "rechunk":
        result = run_rechunk(
            database_url=_database_url(args.database_url),
            data_dir=_data_dir(args.data_dir, parser),
            parsing_config=args.parsing_config,
            source_overlap=args.source_overlap,
            limit=args.limit,
            shards=args.shards,
            shard=args.shard,
        )
    elif args.command == "corpus" and args.corpus_command == "activate":
        result = run_activate(
            release=args.release,
            models=args.models,
            database_url=_database_url(args.database_url),
            qdrant_url=args.qdrant_url,
        )
    elif args.command == "search" and args.search_command == "fetch-model":
        result = run_fetch_model(models=args.models, data_dir=_data_dir(args.data_dir, parser))
    elif args.command == "search" and args.search_command == "build-index":
        result = run_build_index(
            manifest=args.manifest,
            models=args.models,
            data_dir=_data_dir(args.data_dir, parser),
            database_url=_database_url(args.database_url),
            qdrant_url=args.qdrant_url,
            prefix=args.prefix,
            release=args.release,
            collections=[item.strip() for item in args.collections.split(",") if item.strip()],
            device=args.device,
            limit=args.limit,
            cache=not args.no_cache,
        )
    elif args.command == "search" and args.search_command == "validate-index":
        result = run_validate_index(
            release=args.release,
            models=args.models,
            database_url=_database_url(args.database_url),
            qdrant_url=args.qdrant_url,
        )
    elif args.command == "search" and args.search_command == "drop-index":
        result = run_drop_index(
            release=args.release,
            database_url=_database_url(args.database_url),
            qdrant_url=args.qdrant_url,
            data_dir=_data_dir(args.data_dir, parser),
        )
    elif args.command == "search" and args.search_command == "compare-precision":
        if args.model == "reranker":
            if not args.release:
                parser.error("--model reranker needs --release")
            result = run_compare_reranker_precision(
                release=args.release,
                models=args.models,
                data_dir=_data_dir(args.data_dir, parser),
                database_url=_database_url(args.database_url),
                qdrant_url=args.qdrant_url,
                sample=args.sample or 10,
                seed=args.seed,
                depth=args.depth,
            )
        else:
            if not args.manifest:
                parser.error("--model embedding needs --manifest")
            result = run_compare_precision(
                manifest=args.manifest,
                models=args.models,
                data_dir=_data_dir(args.data_dir, parser),
                database_url=_database_url(args.database_url),
                sample=args.sample or 256,
                seed=args.seed,
            )
    elif args.command == "search" and args.search_command == "pilot":
        result = run_search_pilot(
            models=args.models,
            search_config=args.search_config,
            data_dir=_data_dir(args.data_dir, parser),
            database_url=_database_url(args.database_url),
            qdrant_url=args.qdrant_url,
            mode=args.mode,
            queries=args.query,
            split=args.split,
            dataset=args.dataset,
            limit_queries=args.limit_queries,
            traces=args.traces,
        )
    elif args.command == "search" and args.search_command == "compare-oracle":
        result = run_compare_oracle(
            release=args.release,
            database_url=_database_url(args.database_url),
            qdrant_url=args.qdrant_url,
            sample=args.sample,
            seed=args.seed,
            limit=args.limit,
        )
    elif args.command == "api" and args.api_command == "export-schema":
        result = run_export_schema(args.out)
    elif args.command == "llm" and args.llm_command == "spend":
        result = run_llm_spend(
            database_url=_database_url(args.database_url), day=args.day, run_id=args.run
        )
    elif args.command == "worker" and args.worker_command == "run":
        result = run_worker(
            database_url=_database_url(args.database_url),
            staging_dir=_staging_dir(args.staging_dir, parser),
            data_dir=_data_dir(args.data_dir, parser),
            manifest=args.manifest,
            parsing_config=args.parsing_config,
            worker_id=args.worker_id,
            once=args.once,
            max_jobs=args.max_jobs,
        )
    else:
        raise ValueError("unsupported_command")
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
