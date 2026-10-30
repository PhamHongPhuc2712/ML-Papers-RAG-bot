"""Every search an answer ran (agent in the chat, 2026-10-30).

The research agent may search up to three times per answer; the queries are kept in order so
the answer can say what it searched for after a reload. Messages answered before this have none.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0014_message_search_queries"
down_revision: str | None = "0013_message_feedback"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("search_queries", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "search_queries")
