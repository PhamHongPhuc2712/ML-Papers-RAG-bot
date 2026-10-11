"""Chats, their messages and the evidence each answer saw (spec §5, revised 2026-10-11)."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_chats"
down_revision: str | None = "0007_users"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
STAMP = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "chats",
        sa.Column("id", UUID, nullable=False),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("created_at", STAMP, nullable=False),
        sa.Column("updated_at", STAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_chats_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chats"),
    )
    op.create_index("ix_chats_user_updated", "chats", ["user_id", "updated_at"])
    op.create_table(
        "chat_messages",
        sa.Column("id", UUID, nullable=False),
        sa.Column("chat_id", UUID, nullable=False),
        # Denormalized owner: makes a request id unique per user, which is what makes a
        # retried new-chat request find its chat instead of creating a second one.
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("request_id", UUID, nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("claims", postgresql.JSONB(), nullable=True),
        sa.Column("standalone_query", sa.Text(), nullable=True),
        sa.Column(
            "warnings", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("corpus_release_id", sa.String(length=255), nullable=True),
        sa.Column("identity", postgresql.JSONB(), nullable=True),
        sa.Column("trace", postgresql.JSONB(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", STAMP, nullable=False),
        sa.Column("started_at", STAMP, nullable=True),
        sa.Column("completed_at", STAMP, nullable=True),
        sa.CheckConstraint("role in ('user', 'assistant')", name="ck_chat_messages_role"),
        sa.CheckConstraint(
            "status in ('pending', 'complete', 'insufficient', 'failed', 'interrupted')",
            name="ck_chat_messages_status",
        ),
        sa.ForeignKeyConstraint(
            ["chat_id"], ["chats.id"], name="fk_chat_messages_chat_id_chats", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_chat_messages_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chat_messages"),
        sa.UniqueConstraint("chat_id", "ordinal", name="uq_chat_messages_ordinal"),
        sa.UniqueConstraint(
            "user_id", "request_id", "role", name="uq_chat_messages_user_request_role"
        ),
    )
    op.create_table(
        "message_evidence",
        sa.Column("message_id", UUID, nullable=False),
        sa.Column("evidence_id", sa.String(length=16), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        # No foreign key to papers: this is a snapshot, kept even if the corpus changes.
        sa.Column("paper_id", UUID, nullable=False),
        sa.Column("version_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", UUID, nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("venue", sa.String(length=255), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("section", sa.Text(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("abstract_only", sa.Boolean(), nullable=False),
        sa.Column("cited", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["chat_messages.id"],
            name="fk_message_evidence_message_id_chat_messages",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id", "evidence_id", name="pk_message_evidence"),
    )


def downgrade() -> None:
    op.drop_table("message_evidence")
    op.drop_table("chat_messages")
    op.drop_index("ix_chats_user_updated", table_name="chats")
    op.drop_table("chats")
