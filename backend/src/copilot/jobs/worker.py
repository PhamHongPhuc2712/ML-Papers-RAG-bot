"""Worker loop: lease, run the handler outside the transaction, commit output with completion."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from ..db.session import session_factory
from .queue import HEARTBEAT_SECONDS, LEASE_SECONDS, LeasedJob, complete, fail, heartbeat, lease

logger = logging.getLogger(__name__)

Heartbeat = Callable[[], bool]
Writes = Callable[[Session], None]


@dataclass
class JobOutcome:
    """What a handler produced: a result record and the writes to commit with it."""

    result: dict[str, Any] = field(default_factory=dict)
    writes: Writes | None = None


class JobError(Exception):
    """Classified handler failure. Retryable errors back off; others fail terminally."""

    def __init__(self, code: str, *, retryable: bool = True, retry_after: int | None = None):
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after
        super().__init__(code)


class ThrottledError(JobError):
    def __init__(self, retry_after: int | None = None) -> None:
        super().__init__("throttled", retryable=True, retry_after=retry_after)


class LeaseLostError(RuntimeError):
    """The lease expired or moved to another worker before completion."""


Handler = Callable[[LeasedJob, Heartbeat], JobOutcome]


class Worker:
    def __init__(
        self,
        engine: Engine,
        handlers: Mapping[str, Handler],
        *,
        lease_seconds: int = LEASE_SECONDS,
        heartbeat_seconds: int = HEARTBEAT_SECONDS,
    ) -> None:
        self.engine = engine
        self.handlers = handlers
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self._factory = session_factory(engine)

    def run_once(self, worker_id: str) -> bool:
        """Process at most one job. Returns False when nothing is due."""

        with self._factory() as session:
            job = lease(
                session, worker_id, lease_seconds=self.lease_seconds, kinds=list(self.handlers)
            )
            session.commit()
        if job is None:
            return False

        def beat() -> bool:
            with self._factory() as session:
                alive = heartbeat(session, job.id, job.token, lease_seconds=self.lease_seconds)
                session.commit()
            return alive

        stop = threading.Event()
        thread = threading.Thread(target=self._heartbeat_loop, args=(beat, stop), daemon=True)
        thread.start()
        try:
            handler = self.handlers.get(job.kind)
            if handler is None:
                self._fail(job, "unknown_kind", retryable=False)
                return True
            try:
                outcome = handler(job, beat)
            except JobError as error:
                self._fail(
                    job, error.code, retryable=error.retryable, retry_after=error.retry_after
                )
                return True
            except Exception:
                logger.exception("job %s (%s) raised", job.id, job.kind)
                self._fail(job, "handler_exception")
                return True
            self._commit(job, outcome)
            return True
        finally:
            stop.set()
            thread.join(timeout=1.0)

    def run_until_idle(self, worker_id: str, *, max_jobs: int | None = None) -> int:
        processed = 0
        while (max_jobs is None or processed < max_jobs) and self.run_once(worker_id):
            processed += 1
        return processed

    def _commit(self, job: LeasedJob, outcome: JobOutcome) -> None:
        with self._factory() as session:
            try:
                if outcome.writes is not None:
                    outcome.writes(session)
                if not complete(session, job.id, job.token, outcome.result):
                    raise LeaseLostError(str(job.id))
                session.commit()
            except LeaseLostError:
                session.rollback()
                logger.warning("job %s lease lost before completion; output discarded", job.id)
            except JobError as error:
                session.rollback()
                self._fail(
                    job, error.code, retryable=error.retryable, retry_after=error.retry_after
                )
            except Exception:
                session.rollback()
                logger.exception("job %s (%s) failed while committing output", job.id, job.kind)
                self._fail(job, "handler_exception")

    def _fail(
        self,
        job: LeasedJob,
        code: str,
        *,
        retryable: bool = True,
        retry_after: int | None = None,
    ) -> None:
        with self._factory() as session:
            fail(session, job.id, job.token, code, retry_after=retry_after, retryable=retryable)
            session.commit()

    def _heartbeat_loop(self, beat: Heartbeat, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            try:
                if not beat():
                    return
            except Exception:
                logger.exception("heartbeat failed")
                return
