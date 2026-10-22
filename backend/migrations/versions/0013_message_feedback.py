"""Helpful / Not helpful on an answer (sign-up plan, A5).

A rating and when it was given, both null until the owner rates and again once they clear it.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_message_feedback"
down_revision: str | None = "0012_answer_usage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("feedback", sa.String(16), nullable=True))
    op.add_column(
        "chat_messages", sa.Column("feedback_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_check_constraint(
        "ck_chat_messages_feedback",
        "chat_messages",
        "feedback in ('helpful', 'not_helpful')",
    )
    op.create_check_constraint(
        "ck_chat_messages_feedback_at",
        "chat_messages",
        "(feedback is null) = (feedback_at is null)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_chat_messages_feedback_at", "chat_messages", type_="check")
    op.drop_constraint("ck_chat_messages_feedback", "chat_messages", type_="check")
    op.drop_column("chat_messages", "feedback_at")
    op.drop_column("chat_messages", "feedback")
