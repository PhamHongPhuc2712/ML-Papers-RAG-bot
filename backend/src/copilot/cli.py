"""Command-line entry points: fixture replay, pilot ingestion, and the job worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .corpus.dedupe import IdentityConflictError, QuarantineError, resolve_paper
from .db.session import make_engine, migrate_database, session_factory

DEFAULT_FIXTURE_PATH = Path("data/fixtures/metadata.jsonl")
DEFAULT_MANIFEST_PATH = Path("configs/corpus.yaml")
DEFAULT_PARSING_PATH = Path("configs/parsing.yaml")
DEFAULT_ARTIFACTS_PATH = Path("configs/artifacts.yaml")
# Same files Settings reads, in the same precedence order: later entries win
# there, so backend/.env is loaded first and .env cannot overwrite it here.
DOTENV_FILES = (Path("backend/.env"), Path(".env"))


def load_dotenv_files() -> None:
    """Copy dotenv values into ``os.environ`` for the lookups Settings cannot serve.

    ``Settings`` parses these files into its own object and never touches the
    process environment, but the OpenReview credentials
    (``corpus/sources/openreview.py``) and ``default_staging_dir`` below read
    ``os.environ`` directly. Without this they read as unset and the pilot falls
    back to guest access, which OpenReview answers with a 403 challenge. An
    explicitly exported shell variable always wins over the file.
    """

    for candidate in DOTENV_FILES:
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def default_staging_dir() -> Path | None:
    """Raw source artifacts live under the configured data root unless overridden."""

    root = os.environ.get("COPILOT_DATA_DIR") or os.environ.get("DATA_DIR") or ""
    root = root.strip()
    if not root:
        return None
    return Path(root.rstrip("\\/") or root) / "sources"


def default_export_dir() -> Path | None:
    """Snapshots stage under the data root so no bulk data enters the repository."""

    root = os.environ.get("COPILOT_DATA_DIR") or os.environ.get("DATA_DIR") or ""
    root = root.strip()
    if not root:
        return None
    return Path(root.rstrip("\\/") or root) / "exports"


def resolve_export_path(value: Path, parser: argparse.ArgumentParser) -> Path:
    """Resolve a relative --out/--manifest against ${DATA_DIR}/exports.

    Relative paths keep the commands free of shell-specific variable syntax; an
    absolute path is honoured as given.
    """

    if value.is_absolute():
        return value
    base = default_export_dir()
    if base is None:
        parser.error("DATA_DIR must be set, or pass an absolute path")
    return base / value


def run_export(
    run_id: str, out: Path, *, database_url: str, public_only: bool
) -> dict[str, object]:
    from .corpus.export import export_snapshot

    engine = make_engine(database_url)
    try:
        manifest = export_snapshot(run_id, out, engine=engine, public_only=public_only)
        return {
            "run_id": run_id,
            "destination": str(out),
            "public_only": public_only,
            "counts": manifest["counts"],
            "withheld_rows": manifest["withheld_rows"],
            "manifest_sha256": manifest["manifest_sha256"],
        }
    finally:
        engine.dispose()


def run_validate(manifest_path: Path) -> dict[str, object]:
    from .corpus.export import validate_manifest

    manifest = validate_manifest(manifest_path)
    return {
        "manifest": str(manifest_path),
        "valid": True,
        "schema_version": manifest["schema_version"],
        "run_id": manifest["run_id"],
        "counts": manifest["counts"],
        "artifacts": len(manifest["artifacts"]),
    }


def run_restore(manifest_path: Path, *, database_url: str) -> dict[str, object]:
    from .corpus.export import restore_snapshot

    engine = make_engine(database_url)
    try:
        migrate_database(engine)
        counts = restore_snapshot(manifest_path, engine)
        return {"restored_from": str(manifest_path), **counts}
    finally:
        engine.dispose()


def run_publish(manifest_path: Path, repo: str, *, artifacts_config: Path) -> dict[str, object]:
    """Public publication is an explicitly authorized mode and is off by default."""

    import yaml

    raw = yaml.safe_load(Path(artifacts_config).read_text(encoding="utf-8")) or {}
    publication = raw.get("publication", {}) if isinstance(raw, dict) else {}
    if not publication.get("enabled", False):
        raise SystemExit(
            "publication_disabled: set publication.enabled in configs/artifacts.yaml and "
            "confirm redistribution rights before publishing"
        )
    raise SystemExit("publication_not_implemented: no authorized publication has been configured")


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


def run_worker(
    *,
    database_url: str,
    staging_dir: Path,
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

    export = corpus_commands.add_parser("export", help="write an immutable local snapshot")
    export.add_argument("--run", dest="run_id", required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument(
        "--public-only",
        action="store_true",
        help="apply the redistribution filter; withholds rows without established rights",
    )
    export.add_argument("--database-url", default=None, help="defaults to DATABASE_URL")

    validate = corpus_commands.add_parser("validate", help="verify a snapshot manifest")
    validate.add_argument("--manifest", type=Path, required=True)

    restore = corpus_commands.add_parser(
        "restore", help="restore a snapshot into an empty database"
    )
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--database-url", required=True)

    publish = corpus_commands.add_parser("publish", help="authorized public publication only")
    publish.add_argument("--manifest", type=Path, required=True)
    publish.add_argument("--repo", required=True)
    publish.add_argument("--artifacts-config", type=Path, default=DEFAULT_ARTIFACTS_PATH)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv_files()
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
    elif args.command == "corpus" and args.corpus_command == "ingest":
        result = run_ingest(
            args.manifest,
            args.limit,
            database_url=_database_url(args.database_url),
            staging_dir=_staging_dir(args.staging_dir, parser),
        )
    elif args.command == "corpus" and args.corpus_command == "export":
        result = run_export(
            args.run_id,
            resolve_export_path(args.out, parser),
            database_url=_database_url(args.database_url),
            public_only=args.public_only,
        )
    elif args.command == "corpus" and args.corpus_command == "validate":
        result = run_validate(resolve_export_path(args.manifest, parser))
    elif args.command == "corpus" and args.corpus_command == "restore":
        result = run_restore(
            resolve_export_path(args.manifest, parser), database_url=args.database_url
        )
    elif args.command == "corpus" and args.corpus_command == "publish":
        result = run_publish(
            resolve_export_path(args.manifest, parser),
            args.repo,
            artifacts_config=args.artifacts_config,
        )
    elif args.command == "worker" and args.worker_command == "run":
        result = run_worker(
            database_url=_database_url(args.database_url),
            staging_dir=_staging_dir(args.staging_dir, parser),
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
