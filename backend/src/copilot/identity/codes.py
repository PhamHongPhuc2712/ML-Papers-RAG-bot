"""Sign-in codes sent by email (sign-up plan, A1, rules fixed in the plan).

A code is 6 random digits, stored only as an HMAC of (email, code) under a key derived from
SECRET_KEY. It lives 10 minutes, works once and dies on the 5th wrong try; a new code ends
every earlier one. An email may ask once a minute while its newest code is unused (refused,
with the wait; a code already signed in with does not hold the next one back, A6) and 5 times
an hour (beyond that nothing is sent, and the caller is not told). Only the newest code for an
email is ever checked. Requests for one email are serialized by a transaction-scoped
advisory lock, so two at once cannot both pass a limit.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import Connection, Engine, text

CODE_DIGITS = 6
CODE_TTL = timedelta(minutes=10)
MAX_ATTEMPTS = 5
RESEND_AFTER = timedelta(seconds=60)
PER_HOUR = 5


class CodeRateLimited(Exception):
    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"retry after {retry_after} s")


class Verdict(Enum):
    ACCEPTED = "accepted"
    INVALID = "invalid_code"
    EXHAUSTED = "code_exhausted"
    EXPIRED = "code_expired"


@dataclass(frozen=True)
class Issued:
    """A code to send, or None when the hourly limit says to send nothing."""

    code: str | None
    code_id: UUID | None


def code_key(secret_key: str) -> bytes:
    """A key used only for sign-in codes, so the application secret is never used raw."""

    return hmac.new(secret_key.encode(), b"copilot:login-code:v1", hashlib.sha256).digest()


class CodeStore:
    def __init__(
        self,
        engine: Engine,
        secret_key: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._engine = engine
        self._key = code_key(secret_key)
        self._clock = clock

    def _mac(self, email: str, code: str) -> str:
        return hmac.new(self._key, f"{email}\n{code}".encode(), hashlib.sha256).hexdigest()

    def issue(self, email: str) -> Issued:
        """A new code for a normalized email; raises ``CodeRateLimited`` inside a minute."""

        now = self._clock()
        with self._engine.begin() as connection:
            _lock(connection, email)
            newest = connection.execute(
                text(
                    "select created_at, consumed_at from login_codes where email = :email"
                    " order by created_at desc limit 1"
                ),
                {"email": email},
            ).first()
            if (
                newest is not None
                and newest.consumed_at is None
                and now - newest.created_at < RESEND_AFTER
            ):
                wait = (RESEND_AFTER - (now - newest.created_at)).total_seconds()
                raise CodeRateLimited(math.ceil(wait))
            hour = connection.execute(
                text(
                    "select count(*) from login_codes where email = :email and created_at > :since"
                ),
                {"email": email, "since": now - timedelta(hours=1)},
            ).scalar_one()
            if hour >= PER_HOUR:
                return Issued(None, None)
            connection.execute(
                text(
                    "update login_codes set consumed_at = :now"
                    " where email = :email and consumed_at is null"
                ),
                {"email": email, "now": now},
            )
            code = f"{secrets.randbelow(10**CODE_DIGITS):0{CODE_DIGITS}d}"
            code_id = uuid4()
            connection.execute(
                text(
                    "insert into login_codes (id, email, code_hmac, created_at, expires_at)"
                    " values (:id, :email, :mac, :now, :expires)"
                ),
                {
                    "id": code_id,
                    "email": email,
                    "mac": self._mac(email, code),
                    "now": now,
                    "expires": now + CODE_TTL,
                },
            )
        return Issued(code, code_id)

    def discard(self, code_id: UUID) -> None:
        """Forget a code that could not be sent, so it neither works nor counts."""

        with self._engine.begin() as connection:
            connection.execute(text("delete from login_codes where id = :id"), {"id": code_id})

    def verify(self, email: str, code: str) -> Verdict:
        now = self._clock()
        with self._engine.begin() as connection:
            _lock(connection, email)
            row = connection.execute(
                text(
                    "select id, code_hmac, expires_at, attempts, consumed_at from login_codes"
                    " where email = :email order by created_at desc limit 1"
                ),
                {"email": email},
            ).first()
            if row is None or row.consumed_at is not None:
                return Verdict.EXPIRED
            if row.attempts >= MAX_ATTEMPTS:
                return Verdict.EXHAUSTED
            if row.expires_at <= now:
                return Verdict.EXPIRED
            if hmac.compare_digest(str(row.code_hmac), self._mac(email, code)):
                connection.execute(
                    text("update login_codes set consumed_at = :now where id = :id"),
                    {"now": now, "id": row.id},
                )
                return Verdict.ACCEPTED
            attempts = int(row.attempts) + 1
            connection.execute(
                text("update login_codes set attempts = :attempts where id = :id"),
                {"attempts": attempts, "id": row.id},
            )
        return Verdict.EXHAUSTED if attempts >= MAX_ATTEMPTS else Verdict.INVALID


def _lock(connection: Connection, email: str) -> None:
    connection.execute(
        text("select pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"login_codes:{email}"},
    )
