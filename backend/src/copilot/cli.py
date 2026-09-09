"""Small command-line entry point for replaying corpus metadata fixtures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .corpus.dedupe import IdentityConflictError, QuarantineError, resolve_paper
from .db.session import make_engine, migrate_database, session_factory

DEFAULT_FIXTURE_PATH = Path("data/fixtures/metadata.jsonl")


def default_staging_dir() -> Path | None:
    """Raw source artifacts live under the configured data root unless overridden."""

    root = os.environ.get("COPILOT_DATA_DIR") or os.environ.get("DATA_DIR") or ""
    root = root.strip()
    if not root:
        return None
    return Path(root.rstrip("\\/") or root) / "sources"


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "fixtures" and args.fixture_command == "load":
        staging_dir = args.staging_dir or default_staging_dir()
        if staging_dir is None:
            parser.error("--staging-dir is required when DATA_DIR is not set")
        result = load_fixtures(
            args.fixture_path,
            database_url=args.database_url,
            staging_dir=staging_dir,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise ValueError("unsupported_command")


if __name__ == "__main__":
    raise SystemExit(main())
