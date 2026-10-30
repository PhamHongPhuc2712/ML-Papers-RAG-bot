"""Answers used per account and UTC day (sign-up plan, A4), kept apart from the chats.

The daily limit counted stored answers, so deleting a chat gave its answers back. This table
is written when an answer completes and is not touched by deleting chats; it goes with the
account. Filled here from the answers already stored.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_answer_usage"
down_revision: str | None = "0011_login_codes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "answer_usage",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("answers", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_answer_usage_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "day", name="pk_answer_usage"),
        sa.CheckConstraint("answers >= 0", name="ck_answer_usage_answers"),
    )
    op.execute(
        "insert into answer_usage (user_id, day, answers)"
        " select user_id, (started_at at time zone 'UTC')::date, count(*) from chat_messages"
        " where role = 'assistant' and status in ('complete', 'insufficient')"
        " and started_at is not null group by 1, 2"
    )


def downgrade() -> None:
    op.drop_table("answer_usage")
