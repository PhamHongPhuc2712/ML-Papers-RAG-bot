"""Command-line entry points: fixture replay, pilot ingestion, and the job worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .corpus.dedupe import IdentityConflictError, QuarantineError, resolve_paper
from .db.session import make_engine, migrate_database, session_factory

DEFAULT_FIXTURE_PATH = Path("data/fixtures/metadata.jsonl")
DEFAULT_MANIFEST_PATH = Path("configs/corpus.yaml")
DEFAULT_PARSING_PATH = Path("configs/parsing.yaml")
DEFAULT_ARTIFACTS_PATH = Path("configs/artifacts.yaml")


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
        return {"out": str(destination), **{k: manifest[k] for k in ("counts", "rights")}}
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
        return {"policy": config.chunker.policy, "chunker_version": config.chunker.chunker_version,
                **stats.as_dict()}
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
            sys.executable, "-m", "copilot.cli", "worker", "run",
            "--database-url", database_url,
            "--manifest", str(manifest),
            "--parsing-config", str(parsing_config),
            "--staging-dir", str(data_dir / "sources"),
            "--data-dir", str(data_dir),
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

    evaluation = subparsers.add_parser("eval")
    evaluation_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    validate_dataset_cmd = evaluation_commands.add_parser(
        "validate-dataset", help="refuse a dataset that leaks between splits"
    )
    validate_dataset_cmd.add_argument("--path", type=Path, required=True)

    worker = subparsers.add_parser("worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    run = worker_commands.add_parser("run", help="process leased jobs")
    run.add_argument(
        "--worker-id", default=f"{os.environ.get('COMPUTERNAME', 'worker')}-{os.getpid()}"
    )
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
    elif args.command == "eval" and args.eval_command == "validate-dataset":
        from .evaluation.datasets import validate_dataset

        result = dict(validate_dataset(args.path))
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
