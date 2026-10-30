"""Sign-in by emailed code (sign-up plan, A1): the codes, and accounts without a password.

A code is stored only as an HMAC of (email, code). Rows name an email rather than a user,
because the first code an address receives is what creates its account.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_login_codes"
down_revision: str | None = "0010_chat_message_references"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "login_codes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("code_hmac", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("email = lower(email)", name="ck_login_codes_email_lower"),
        sa.PrimaryKeyConstraint("id", name="pk_login_codes"),
    )
    op.create_index("ix_login_codes_email_created", "login_codes", ["email", "created_at"])
    op.alter_column("users", "password_hash", existing_type=sa.Text(), nullable=True)


def downgrade() -> None:
    op.execute("delete from users where password_hash is null")
    op.alter_column("users", "password_hash", existing_type=sa.Text(), nullable=False)
    op.drop_index("ix_login_codes_email_created", table_name="login_codes")
    op.drop_table("login_codes")
