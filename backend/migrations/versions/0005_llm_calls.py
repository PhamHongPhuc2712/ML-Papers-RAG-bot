"""Add the hosted-LLM spend ledger (spec §12): one row per priced call, no text."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_llm_calls"
down_revision: str | None = "0004_search_orderings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Deliberately no column for a prompt, a query or a response: the ledger
    # records what a call cost, never what it said.
    op.create_table(
        "llm_calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("run_id", sa.String(length=255), nullable=True),
        sa.Column("purpose", sa.String(length=16), nullable=False),
        sa.Column("model_key", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("served_model", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("estimated_usd", sa.Numeric(12, 6), nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("purpose in ('search', 'evaluation')", name="ck_llm_calls_purpose"),
        sa.CheckConstraint(
            "status in ('reserved', 'succeeded', 'failed')", name="ck_llm_calls_status"
        ),
        sa.CheckConstraint(
            "estimated_usd >= 0 and (cost_usd is null or cost_usd >= 0)",
            name="ck_llm_calls_nonnegative",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_llm_calls"),
    )
    op.create_index("ix_llm_calls_created_at", "llm_calls", ["created_at"])
    op.create_index("ix_llm_calls_run_id", "llm_calls", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_llm_calls_run_id", table_name="llm_calls")
    op.drop_index("ix_llm_calls_created_at", table_name="llm_calls")
    op.drop_table("llm_calls")
