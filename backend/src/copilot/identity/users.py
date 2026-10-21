"""Accounts (spec §10). Since the sign-up plan (2026-10-21) anyone can make one with a code
sent to their email (``identity/codes.py``), and sign in only that way; the operator's CLI
still makes them. Accounts made before then keep an Argon2id hash in ``password_hash``,
which nothing reads any more; the column is dropped in a later migration."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AccountError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def normalize_email(email: str) -> str:
    value = email.strip().lower()
    if len(value) > 320 or not _EMAIL.match(value):
        raise AccountError("email_invalid")
    return value


class UserStore:
    def __init__(
        self, engine: Engine, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self._engine = engine
        self._clock = clock

    def create(self, email: str) -> UUID:
        address = normalize_email(email)
        user_id = uuid4()
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    text(
                        "insert into users (id, email, password_hash, created_at)"
                        " values (:id, :email, null, :now)"
                    ),
                    {"id": user_id, "email": address, "now": self._clock()},
                )
        except IntegrityError as error:
            raise AccountError("email_taken") from error
        return user_id

    def for_code(self, email: str) -> UUID | None:
        """The enabled account a verified code signs in to, made now if the email has none;
        None when the account is disabled."""

        address = normalize_email(email)
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    "insert into users (id, email, password_hash, created_at)"
                    " values (:id, :email, null, :now) on conflict (email) do nothing"
                ),
                {"id": uuid4(), "email": address, "now": self._clock()},
            )
            row = connection.execute(
                text("select id, disabled_at from users where email = :email"),
                {"email": address},
            ).one()
        return None if row.disabled_at is not None else UUID(str(row.id))

    def disabled(self, email: str) -> bool:
        with self._engine.connect() as connection:
            return (
                connection.execute(
                    text(
                        "select 1 from users where email = :email and disabled_at is not null"
                    ),
                    {"email": normalize_email(email)},
                ).first()
                is not None
            )

    def active(self, user_id: UUID) -> bool:
        with self._engine.connect() as connection:
            return (
                connection.execute(
                    text("select 1 from users where id = :id and disabled_at is null"),
                    {"id": user_id},
                ).first()
                is not None
            )

    def email(self, user_id: UUID) -> str | None:
        with self._engine.connect() as connection:
            value = connection.execute(
                text("select email from users where id = :id and disabled_at is null"),
                {"id": user_id},
            ).scalar_one_or_none()
        return None if value is None else str(value)

    def disable(self, email: str) -> bool:
        with self._engine.begin() as connection:
            result = connection.execute(
                text(
                    "update users set disabled_at = :now"
                    " where email = :email and disabled_at is null"
                ),
                {"email": normalize_email(email), "now": self._clock()},
            )
        return result.rowcount > 0

    def delete(self, user_id: UUID) -> bool:
        """Delete the account; chats, messages and evidence go with it (ON DELETE CASCADE).

        The spend ledger keeps its rows: they hold no text and no user id.
        """

        with self._engine.begin() as connection:
            result = connection.execute(text("delete from users where id = :id"), {"id": user_id})
        return result.rowcount > 0

    def id_for(self, email: str) -> UUID | None:
        with self._engine.connect() as connection:
            value = connection.execute(
                text("select id from users where email = :email"),
                {"email": normalize_email(email)},
            ).scalar_one_or_none()
        return None if value is None else UUID(str(value))
