"""Answers a day per account (sign-up plan, A3; 20, the developer's choice, configs/chat.yaml).

An answer counts when its assistant message ends ``complete`` or ``insufficient``; a turn that
failed or was interrupted delivered nothing and does not. A turn still ``pending`` holds its
place, and the check runs under a lock on the account's row, so two turns started together
cannot both take the last answer. Days and months are UTC, like the spend cap, and an answer
belongs to the day its turn started (``started_at``, renewed when a failed turn is retried).
Completed answers are counted in ``answer_usage`` (A4), not read back from the messages, so
deleting chats does not give answers back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import Connection, Engine, text


class DailyLimitReached(Exception):
    def __init__(self, limit: int, resets_at: datetime) -> None:
        self.limit = limit
        self.resets_at = resets_at
        super().__init__(f"{limit} answers used; resets at {resets_at.isoformat()}")


@dataclass(frozen=True)
class Usage:
    today: int
    month: int
    limit: int
    resets_at: datetime


def day_start(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _answers(connection: Connection, user_id: UUID, since: datetime) -> int:
    return int(
        connection.execute(
            text(
                "select coalesce(sum(answers), 0) from answer_usage"
                " where user_id = :user and day >= :since"
            ),
            {"user": user_id, "since": since.date()},
        ).scalar_one()
    )


def _pending(connection: Connection, user_id: UUID, since: datetime) -> int:
    return int(
        connection.execute(
            text(
                "select count(*) from chat_messages where user_id = :user and role = 'assistant'"
                " and status = 'pending' and started_at >= :since"
            ),
            {"user": user_id, "since": since},
        ).scalar_one()
    )


def count_answer(connection: Connection, message_id: UUID) -> None:
    """Count a just-completed answer on its turn's UTC day, in the completing transaction."""

    connection.execute(
        text(
            "insert into answer_usage (user_id, day, answers)"
            " select user_id, (started_at at time zone 'UTC')::date, 1 from chat_messages"
            " where id = :id on conflict (user_id, day)"
            " do update set answers = answer_usage.answers + 1"
        ),
        {"id": message_id},
    )


def lock_account(connection: Connection, user_id: UUID) -> None:
    """Serialize this account's turn starts until the transaction ends. Take it before the
    transaction writes anything that references the account."""

    connection.execute(text("select 1 from users where id = :id for update"), {"id": user_id})


def check_daily_limit(connection: Connection, user_id: UUID, limit: int, now: datetime) -> None:
    """Raise ``DailyLimitReached`` when a new turn would pass the limit; call it inside the
    transaction that writes the turn's pending message, after ``lock_account``."""

    lock_account(connection, user_id)
    start = day_start(now)
    if _answers(connection, user_id, start) + _pending(connection, user_id, start) >= limit:
        raise DailyLimitReached(limit, start + timedelta(days=1))


def usage(engine: Engine, user_id: UUID, *, limit: int, now: datetime | None = None) -> Usage:
    current = now or datetime.now(UTC)
    start = day_start(current)
    with engine.connect() as connection:
        today = _answers(connection, user_id, start)
        month = _answers(connection, user_id, start.replace(day=1))
    return Usage(today=today, month=month, limit=limit, resets_at=start + timedelta(days=1))
