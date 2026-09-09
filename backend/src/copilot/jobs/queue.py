"""Durable job queue on PostgreSQL row leases.

Every state change uses server time and a lease token, so a worker that lost
its lease (crash, pause, expiry) can neither complete nor extend a job that a
newer owner holds. Leasing uses ``FOR UPDATE SKIP LOCKED`` in a short
transaction; external work happens outside it (spec §4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..db.models import Job

LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 15
MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 60
ACTIVE_STATUSES = ("queued", "retry_wait")


@dataclass(frozen=True)
class LeasedJob:
    id: UUID
    kind: str
    payload: dict[str, Any]
    attempt: int
    token: UUID
    worker_id: str


def backoff_seconds(attempt: int, retry_after: int | None = None) -> int:
    """Exponential backoff capped at 60 s, unless the provider asked for longer."""

    base = min(MAX_BACKOFF_SECONDS, 2 ** max(1, attempt))
    if retry_after is not None and retry_after > base:
        return int(retry_after)
    return base


def enqueue(
    session: Session,
    kind: str,
    payload: Mapping[str, Any],
    key: str,
    *,
    max_attempts: int = MAX_ATTEMPTS,
) -> UUID:
    """Insert a job once per idempotency key and return its identifier."""

    statement = (
        pg_insert(Job)
        .values(
            id=uuid4(),
            kind=kind,
            payload=dict(payload),
            idempotency_key=key,
            status="queued",
            max_attempts=max_attempts,
        )
        .on_conflict_do_nothing(index_elements=["idempotency_key"])
        .returning(Job.id)
    )
    job_id = session.execute(statement).scalar_one_or_none()
    if job_id is None:
        job_id = session.execute(select(Job.id).where(Job.idempotency_key == key)).scalar_one()
    return job_id


def lease(
    session: Session,
    worker_id: str,
    *,
    lease_seconds: int = LEASE_SECONDS,
    kinds: Sequence[str] | None = None,
) -> LeasedJob | None:
    """Claim one due job with SKIP LOCKED; the caller commits the short transaction."""

    due = or_(
        and_(
            Job.status.in_(ACTIVE_STATUSES),
            or_(Job.next_attempt_at.is_(None), Job.next_attempt_at <= func.now()),
        ),
        and_(Job.status == "running", Job.lease_until < func.now()),
    )
    statement = select(Job).where(due)
    if kinds:
        statement = statement.where(Job.kind.in_(list(kinds)))
    statement = (
        statement.order_by(Job.created_at, Job.id).limit(1).with_for_update(skip_locked=True)
    )
    job = session.execute(statement).scalar_one_or_none()
    if job is None:
        return None
    token = uuid4()
    next_attempt = job.attempt + 1
    session.execute(
        update(Job)
        .where(Job.id == job.id)
        .values(
            status="running",
            attempt=next_attempt,
            lease_token=token,
            worker_id=worker_id,
            lease_until=func.now() + timedelta(seconds=lease_seconds),
            heartbeat_at=func.now(),
            error_code=None,
            updated_at=func.now(),
        )
    )
    session.flush()
    return LeasedJob(
        id=job.id,
        kind=job.kind,
        payload=dict(job.payload),
        attempt=next_attempt,
        token=token,
        worker_id=worker_id,
    )


def _live(job_id: UUID, token: UUID) -> Any:
    return and_(
        Job.id == job_id,
        Job.lease_token == token,
        Job.status == "running",
        Job.lease_until > func.now(),
    )


def heartbeat(
    session: Session, job_id: UUID, token: UUID, *, lease_seconds: int = LEASE_SECONDS
) -> bool:
    """Extend a live lease; returns False when the lease is no longer this worker's."""

    result = session.execute(
        update(Job)
        .where(_live(job_id, token))
        .values(
            heartbeat_at=func.now(),
            lease_until=func.now() + timedelta(seconds=lease_seconds),
            updated_at=func.now(),
        )
    )
    return result.rowcount == 1


def complete(
    session: Session, job_id: UUID, token: UUID, result: Mapping[str, Any] | None = None
) -> bool:
    """Mark success only while the lease is live; a lost lease cannot mark success."""

    outcome = session.execute(
        update(Job)
        .where(_live(job_id, token))
        .values(
            status="succeeded",
            result=dict(result or {}),
            lease_until=None,
            lease_token=None,
            error_code=None,
            updated_at=func.now(),
        )
    )
    return outcome.rowcount == 1


def fail(
    session: Session,
    job_id: UUID,
    token: UUID,
    error_code: str,
    *,
    retry_after: int | None = None,
    retryable: bool = True,
) -> bool:
    """Schedule a retry with backoff, or fail terminally after the last attempt."""

    job = session.execute(
        select(Job).where(_live(job_id, token)).with_for_update()
    ).scalar_one_or_none()
    if job is None:
        return False
    if not retryable or job.attempt >= job.max_attempts:
        values: dict[str, Any] = {"status": "failed", "next_attempt_at": None}
    else:
        delay = backoff_seconds(job.attempt, retry_after)
        values = {
            "status": "retry_wait",
            "next_attempt_at": func.now() + timedelta(seconds=delay),
        }
    outcome = session.execute(
        update(Job)
        .where(_live(job_id, token))
        .values(
            error_code=error_code[:128],
            lease_until=None,
            lease_token=None,
            updated_at=func.now(),
            **values,
        )
    )
    return outcome.rowcount == 1
