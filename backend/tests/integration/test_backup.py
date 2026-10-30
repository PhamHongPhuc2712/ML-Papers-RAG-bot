"""User data backs up to files under DATA_DIR and restores only where it is safe to."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from copilot.backup import BackupError, backup_dir, create_backup, restore_backup
from copilot.chat.store import ChatStore
from copilot.contracts import ChatRequest, Principal

pytestmark = pytest.mark.integration


def test_a_backup_restores_into_empty_tables_and_refuses_live_ones(
    migrated_database, make_user, test_settings, tmp_path
):
    user_id, _ = make_user()
    ChatStore(migrated_database).begin_turn(
        Principal(user_id=user_id),
        ChatRequest(request_id=uuid4(), message="kept"),
        stale_after=timedelta(minutes=4),
    )
    manifest = create_backup(migrated_database, tmp_path / "b1")
    assert manifest["tables"]["users"]["rows"] >= 1 and manifest["revision"]
    with pytest.raises(BackupError, match="target_not_empty"):
        restore_backup(migrated_database, tmp_path / "b1", replace=False)
    restore_backup(migrated_database, tmp_path / "b1", replace=True)
    with migrated_database.connect() as connection:
        kept = connection.execute(
            text("select count(*) from chat_messages where text = 'kept'")
        ).scalar_one()
    assert kept == 1
    (tmp_path / "b1" / "chats.csv").write_bytes(
        (tmp_path / "b1" / "chats.csv").read_bytes() + b"\n"
    )
    with pytest.raises(BackupError, match="checksum_mismatch"):
        restore_backup(migrated_database, tmp_path / "b1", replace=True)
    (tmp_path / "b1" / "manifest.json").write_text(
        (tmp_path / "b1" / "manifest.json").read_text().replace(manifest["revision"], "0000_other")
    )
    with pytest.raises(BackupError, match="revision_mismatch"):
        restore_backup(migrated_database, tmp_path / "b1", replace=True)


def test_a_backup_never_goes_outside_data_dir(tmp_path):
    now = datetime(2026, 10, 14, 9, 30, tzinfo=UTC)
    data = tmp_path / "data"
    assert backup_dir(data, None, now) == (data / "backups" / "20261014T093000Z").resolve()
    assert backup_dir(data, data / "backups" / "x", now) == (data / "backups" / "x").resolve()
    for outside in (tmp_path / "elsewhere", data / ".." / "escape"):
        with pytest.raises(BackupError, match="outside_data_dir"):
            backup_dir(data, outside, now)
