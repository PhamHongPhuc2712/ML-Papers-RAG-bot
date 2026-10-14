"""Accounts (spec §10): created by the operator's CLI, since signup is closed, and
authenticated by email and an Argon2id hash in the local database."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from .passwords import hash_password, needs_rehash, verify_password

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

    def create(self, email: str, password: str) -> UUID:
        address = normalize_email(email)
        try:
            stored = hash_password(password)
        except ValueError as error:
            raise AccountError("password_length") from error
        user_id = uuid4()
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    text(
                        "insert into users (id, email, password_hash, created_at)"
                        " values (:id, :email, :hash, :now)"
                    ),
                    {"id": user_id, "email": address, "hash": stored, "now": self._clock()},
                )
        except IntegrityError as error:
            raise AccountError("email_taken") from error
        return user_id

    def authenticate(self, email: str, password: str) -> UUID | None:
        """The account's id when the password matches an enabled account, else None."""

        try:
            address: str | None = normalize_email(email)
        except AccountError:
            address = None
        row = None
        if address is not None:
            with self._engine.connect() as connection:
                row = connection.execute(
                    text(
                        "select id, password_hash from users"
                        " where email = :email and disabled_at is null"
                    ),
                    {"email": address},
                ).first()
        stored = None if row is None else str(row.password_hash)
        if row is None or stored is None or not verify_password(stored, password):
            # verify_password ran against the dummy hash when row is None: same cost.
            if row is None:
                verify_password(None, password)
            return None
        if needs_rehash(stored):
            with self._engine.begin() as connection:
                connection.execute(
                    text("update users set password_hash = :hash where id = :id"),
                    {"hash": hash_password(password), "id": row.id},
                )
        return UUID(str(row.id))

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

    def check_password(self, user_id: UUID, password: str) -> bool:
        """Whether the password is this enabled account's, at the same cost either way."""

        with self._engine.connect() as connection:
            stored = connection.execute(
                text("select password_hash from users where id = :id and disabled_at is null"),
                {"id": user_id},
            ).scalar_one_or_none()
        return verify_password(None if stored is None else str(stored), password)

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
