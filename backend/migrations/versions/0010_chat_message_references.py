"""An answer's References (RAG-only plan, U1): the papers its search ranked, ids only.

``[{paper_id, rank, read}]`` per assistant message; title, venue and year are read from the
corpus when the history is. Named ``paper_references``: ``references`` is reserved in SQL.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_chat_message_references"
down_revision: str | None = "0009_llm_call_purpose_chat"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column(
            "paper_references",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "paper_references")
