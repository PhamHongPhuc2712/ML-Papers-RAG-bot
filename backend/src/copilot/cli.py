"""Small command-line entry point for replaying corpus metadata fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus.dedupe import IdentityConflictError, QuarantineError, resolve_paper
from .db.session import make_engine, migrate_database, session_factory


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
                    decoded["_staging_dir"] = str(staging_dir)
                    with session.begin():
                        try:
                            resolve_paper(decoded, session)
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
    load.add_argument(
        "fixture_path", type=Path, nargs="?", default=Path("data/fixtures/metadata.jsonl")
    )
    load.add_argument("--database-url", required=True)
    load.add_argument("--staging-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fixtures" and args.fixture_command == "load":
        result = load_fixtures(
            args.fixture_path,
            database_url=args.database_url,
            staging_dir=args.staging_dir,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise ValueError("unsupported_command")


if __name__ == "__main__":
    raise SystemExit(main())
