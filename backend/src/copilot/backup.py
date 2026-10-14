"""Backups of user data (spec §12): accounts, chats, messages and evidence snapshots.

The corpus is rebuilt from its own snapshots (corpus export/restore); this covers only what
users created. Each table is copied out as CSV with PostgreSQL's COPY, beside a manifest of
row counts, sha256 digests and the schema revision. A restore refuses another revision and,
unless asked to replace, any target table that already has rows.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import psycopg
from sqlalchemy import Engine, text

# Table names are this constant, never input: the f-strings below interpolate only these.
# Foreign-key order, so a table is loaded after the ones it references.
USER_TABLES = ("users", "chats", "chat_messages", "message_evidence")


class BackupError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def backup_dir(data_dir: Path, out: Path | None, now: datetime) -> Path:
    """Where a backup goes: ``out``, or ``<DATA_DIR>/backups/<UTC stamp>``; never outside DATA_DIR.

    Backups hold users' chats, so they stay where deleting DATA_DIR deletes them.
    """

    root = data_dir.resolve()
    target = (
        out if out is not None else root / "backups" / now.strftime("%Y%m%dT%H%M%SZ")
    ).resolve()
    if not target.is_relative_to(root):
        raise BackupError("outside_data_dir", str(target))
    return target


def _revision(engine: Engine) -> str:
    with engine.connect() as connection:
        return str(connection.execute(text("select version_num from alembic_version")).scalar_one())


def _driver(engine: Engine) -> tuple[Any, psycopg.Connection[Any]]:
    raw = engine.raw_connection()
    return raw, cast(psycopg.Connection[Any], raw.driver_connection)


def create_backup(engine: Engine, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    tables: dict[str, Any] = {}
    raw, driver = _driver(engine)
    try:
        with driver.cursor() as cursor:
            # One snapshot for all four tables, so a turn written mid-backup is in all or none.
            cursor.execute("set transaction isolation level repeatable read, read only")
            for table in USER_TABLES:
                target = out / f"{table}.csv"
                with (
                    target.open("wb") as handle,
                    cursor.copy(f"COPY {table} TO STDOUT (FORMAT csv, HEADER)") as copy,
                ):
                    for chunk in copy:
                        handle.write(bytes(chunk))
                data = target.read_bytes()
                # Over-counts a value with a newline: a sanity figure; the digest is the check.
                tables[table] = {
                    "rows": max(0, data.count(b"\n") - 1),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
        driver.commit()
    finally:
        raw.close()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "revision": _revision(engine),
        "tables": tables,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def restore_backup(engine: Engine, source: Path, *, replace: bool) -> dict[str, Any]:
    manifest = json.loads((source / "manifest.json").read_text())
    current = _revision(engine)
    if manifest["revision"] != current:
        raise BackupError("revision_mismatch", f"{manifest['revision']} != {current}")
    if set(manifest["tables"]) != set(USER_TABLES):
        raise BackupError("tables_mismatch", ",".join(sorted(manifest["tables"])))
    for table, facts in manifest["tables"].items():
        if hashlib.sha256((source / f"{table}.csv").read_bytes()).hexdigest() != facts["sha256"]:
            raise BackupError("checksum_mismatch", table)
    raw, driver = _driver(engine)
    try:
        with driver.cursor() as cursor:
            for table in USER_TABLES:
                cursor.execute(f"select exists (select 1 from {table})")
                row = cursor.fetchone()
                if row is not None and row[0] and not replace:
                    raise BackupError("target_not_empty", table)
            if replace:
                cursor.execute(f"truncate {', '.join(USER_TABLES)}")
            for table in USER_TABLES:
                with (
                    (source / f"{table}.csv").open("rb") as handle,
                    cursor.copy(f"COPY {table} FROM STDIN (FORMAT csv, HEADER)") as copy,
                ):
                    while chunk := handle.read(1 << 20):
                        copy.write(chunk)
        driver.commit()
    except Exception:
        driver.rollback()
        raise
    finally:
        raw.close()
    return {"restored": {table: manifest["tables"][table]["rows"] for table in USER_TABLES}}
