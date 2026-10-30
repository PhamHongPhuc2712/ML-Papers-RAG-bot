"""Ledger rows for chat turns (spec §12): the purpose check admits 'chat'."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009_llm_call_purpose_chat"
down_revision: str | None = "0008_chats"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_llm_calls_purpose", "llm_calls", type_="check")
    op.create_check_constraint(
        "ck_llm_calls_purpose", "llm_calls", "purpose in ('search', 'evaluation', 'chat')"
    )


def downgrade() -> None:
    # Fails while chat rows exist: the ledger is never rewritten to fit an older schema.
    op.drop_constraint("ck_llm_calls_purpose", "llm_calls", type_="check")
    op.create_check_constraint(
        "ck_llm_calls_purpose", "llm_calls", "purpose in ('search', 'evaluation')"
    )
