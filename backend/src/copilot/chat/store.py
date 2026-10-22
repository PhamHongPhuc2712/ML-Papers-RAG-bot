"""Chats, messages and evidence snapshots in PostgreSQL, owned by one user each.

Every read and write names the principal's user id; a chat that is someone else's is
``ChatNotFound``, exactly like one that does not exist. A turn is a user message and an
assistant message inserted together, the assistant ``pending`` until the turn completes,
fails or is interrupted. A request id is unique per user: the same request again replays a
finished answer, runs a failed or interrupted one again, and is refused while it is running.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import IntegrityError

from ..contracts import (
    ChatMessage,
    ChatRequest,
    ChatSummary,
    Claim,
    Feedback,
    Principal,
    Reference,
    StoredEvidence,
)
from .limits import check_daily_limit, count_answer, lock_account
from .turns import INSUFFICIENT_TEXT, HistoryItem, TurnResult, answer_text, chat_title

ANSWERED = ("complete", "insufficient")


class ChatNotFound(Exception):
    pass


class MessageNotRatable(Exception):
    """Not an answer in this chat, or one still pending, failed or interrupted."""


class TurnConflict(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Turn:
    chat_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    ordinal: int
    replay: ChatMessage | None


_MESSAGE_COLUMNS = (
    "id, ordinal, role, status, text, claims, standalone_query, warnings, error_code,"
    " corpus_release_id, created_at, paper_references, feedback"
)


class ChatStore:
    def __init__(
        self, engine: Engine, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self._engine = engine
        self._clock = clock

    # --- Reads ---------------------------------------------------------------------------

    def list_chats(self, principal: Principal, *, limit: int) -> list[ChatSummary]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    "select id, title, created_at, updated_at from chats where user_id = :user"
                    " order by updated_at desc, id limit :limit"
                ),
                {"user": principal.user_id, "limit": limit},
            ).all()
        return [ChatSummary.model_validate(row._asdict()) for row in rows]

    def get_chat(
        self, principal: Principal, chat_id: UUID
    ) -> tuple[ChatSummary, list[ChatMessage]]:
        with self._engine.connect() as connection:
            row = connection.execute(
                text(
                    "select id, title, created_at, updated_at from chats"
                    " where id = :chat and user_id = :user"
                ),
                {"chat": chat_id, "user": principal.user_id},
            ).first()
            if row is None:
                raise ChatNotFound()
            rows = connection.execute(
                text(
                    f"select {_MESSAGE_COLUMNS} from chat_messages where chat_id = :chat"
                    " order by ordinal"
                ),
                {"chat": chat_id},
            ).all()
            evidence = _evidence_for(connection, [r.id for r in rows])
            papers = _reference_facts(connection, rows)
        messages = [_message(r, evidence.get(r.id, []), papers) for r in rows]
        return ChatSummary.model_validate(row._asdict()), messages

    def history(self, chat_id: UUID, *, before_ordinal: int, turns: int) -> list[HistoryItem]:
        """The last ``turns`` answered exchanges before ``before_ordinal``, oldest first."""

        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    "select ordinal, role, status, text from chat_messages"
                    " where chat_id = :chat and ordinal < :before order by ordinal"
                ),
                {"chat": chat_id, "before": before_ordinal},
            ).all()
        pairs: list[tuple[str, str]] = []
        for user, assistant in zip(rows[0::2], rows[1::2], strict=False):
            if assistant.status not in ANSWERED:
                continue
            reply = assistant.text if assistant.status == "complete" else INSUFFICIENT_TEXT
            pairs.append((user.text, reply))
        kept = pairs[-turns:] if turns > 0 else []
        items: list[HistoryItem] = []
        for question, reply in kept:
            items += [HistoryItem("user", question), HistoryItem("assistant", reply)]
        return items

    # --- Writes --------------------------------------------------------------------------

    def delete_chat(self, principal: Principal, chat_id: UUID) -> None:
        with self._engine.begin() as connection:
            deleted = connection.execute(
                text("delete from chats where id = :chat and user_id = :user"),
                {"chat": chat_id, "user": principal.user_id},
            ).rowcount
        if not deleted:
            raise ChatNotFound()

    def delete_all_chats(self, principal: Principal) -> int:
        """Delete every chat of this account; its answers stay counted (``answer_usage``)."""

        with self._engine.begin() as connection:
            return int(
                connection.execute(
                    text("delete from chats where user_id = :user"), {"user": principal.user_id}
                ).rowcount
            )

    def set_feedback(
        self, principal: Principal, chat_id: UUID, message_id: UUID, rating: Feedback | None
    ) -> None:
        """Rate an answered assistant message of the principal's chat; ``None`` clears it."""

        with self._engine.begin() as connection:
            owned = connection.execute(
                text("select 1 from chats where id = :chat and user_id = :user"),
                {"chat": chat_id, "user": principal.user_id},
            ).first()
            if owned is None:
                raise ChatNotFound()
            updated = connection.execute(
                text(
                    "update chat_messages set feedback = :rating, feedback_at = :at"
                    " where id = :id and chat_id = :chat and role = 'assistant'"
                    " and status in ('complete', 'insufficient')"
                ),
                {
                    "rating": rating,
                    "at": None if rating is None else self._clock(),
                    "id": message_id,
                    "chat": chat_id,
                },
            ).rowcount
        if not updated:
            raise MessageNotRatable()

    def interrupt_stale(self, *, stale_after: timedelta) -> int:
        now = self._clock()
        with self._engine.begin() as connection:
            return _interrupt(connection, now=now, before=now - stale_after, chat_id=None)

    def begin_turn(
        self,
        principal: Principal,
        request: ChatRequest,
        *,
        stale_after: timedelta,
        daily_answers: int | None = None,
    ) -> Turn:
        """Write the turn's pending answer; ``DailyLimitReached`` when ``daily_answers`` are
        used (a replay of an answered request is never refused: it spends nothing)."""

        now = self._clock()
        try:
            with self._engine.begin() as connection:
                return self._begin(
                    connection, principal.user_id, request, now, stale_after, daily_answers
                )
        except IntegrityError as error:
            # A concurrent copy of this request won the insert.
            raise TurnConflict("request_in_progress") from error

    def _begin(
        self,
        connection: Connection,
        user_id: UUID,
        request: ChatRequest,
        now: datetime,
        stale_after: timedelta,
        daily_answers: int | None,
    ) -> Turn:
        if daily_answers is not None:
            # Before anything is written: inserting a chat takes a key-share lock on this row
            # through its foreign key, so locking it later could deadlock two turns at once.
            lock_account(connection, user_id)
        chat_id = request.chat_id
        if chat_id is None:
            chat_id = connection.execute(
                text(
                    "select chat_id from chat_messages where user_id = :user"
                    " and request_id = :request and role = 'user'"
                ),
                {"user": user_id, "request": request.request_id},
            ).scalar_one_or_none()
        if chat_id is None:
            chat_id = uuid4()
            connection.execute(
                text(
                    "insert into chats (id, user_id, title, created_at, updated_at)"
                    " values (:id, :user, :title, :now, :now)"
                ),
                {"id": chat_id, "user": user_id, "title": chat_title(request.message), "now": now},
            )
        owned = connection.execute(
            text("select id from chats where id = :chat and user_id = :user for update"),
            {"chat": chat_id, "user": user_id},
        ).first()
        if owned is None:
            raise ChatNotFound()
        _interrupt(connection, now=now, before=now - stale_after, chat_id=chat_id)
        existing = connection.execute(
            text(
                "select id, ordinal, role, status, text from chat_messages"
                " where chat_id = :chat and user_id = :user and request_id = :request"
                " order by ordinal"
            ),
            {"chat": chat_id, "user": user_id, "request": request.request_id},
        ).all()
        if existing:
            user_row, assistant_row = existing[0], existing[1]
            if user_row.text != request.message:
                raise TurnConflict("request_id_reused")
            if assistant_row.status in ANSWERED:
                replay = _load_message(connection, assistant_row.id)
                return Turn(chat_id, user_row.id, assistant_row.id, assistant_row.ordinal, replay)
            if assistant_row.status == "pending":
                raise TurnConflict("request_in_progress")
            if daily_answers is not None:
                check_daily_limit(connection, user_id, daily_answers, now)
            connection.execute(
                text(
                    "update chat_messages set status = 'pending', error_code = null, text = '',"
                    " claims = null, standalone_query = null, warnings = '[]'::jsonb,"
                    " paper_references = '[]'::jsonb, started_at = :now, completed_at = null"
                    " where id = :id"
                ),
                {"id": assistant_row.id, "now": now},
            )
            connection.execute(
                text("delete from message_evidence where message_id = :id"),
                {"id": assistant_row.id},
            )
            _touch(connection, chat_id, now)
            return Turn(chat_id, user_row.id, assistant_row.id, assistant_row.ordinal, None)
        busy = connection.execute(
            text(
                "select 1 from chat_messages where chat_id = :chat and role = 'assistant'"
                " and status = 'pending' limit 1"
            ),
            {"chat": chat_id},
        ).first()
        if busy is not None:
            raise TurnConflict("chat_busy")
        if daily_answers is not None:
            check_daily_limit(connection, user_id, daily_answers, now)
        last = connection.execute(
            text("select coalesce(max(ordinal), -1) from chat_messages where chat_id = :chat"),
            {"chat": chat_id},
        ).scalar_one()
        user_message, assistant_message = uuid4(), uuid4()
        insert = text(
            "insert into chat_messages (id, chat_id, user_id, ordinal, role, request_id, status,"
            " text, created_at, started_at) values (:id, :chat, :user, :ordinal, :role,"
            " :request, :status, :text, :now, :started)"
        )
        common = {"chat": chat_id, "user": user_id, "request": request.request_id, "now": now}
        connection.execute(
            insert,
            {
                **common,
                "id": user_message,
                "ordinal": last + 1,
                "role": "user",
                "status": "complete",
                "text": request.message,
                "started": None,
            },
        )
        connection.execute(
            insert,
            {
                **common,
                "id": assistant_message,
                "ordinal": last + 2,
                "role": "assistant",
                "status": "pending",
                "text": "",
                "started": now,
            },
        )
        _touch(connection, chat_id, now)
        return Turn(chat_id, user_message, assistant_message, last + 2, None)

    def complete_turn(self, turn: Turn, result: TurnResult) -> ChatMessage:
        answer = result.answer
        status = "insufficient" if answer.insufficient_evidence or not answer.claims else "complete"
        cited = {label for claim in answer.claims for label in claim.evidence_ids}
        now = self._clock()
        with self._engine.begin() as connection:
            updated = connection.execute(
                text(
                    "update chat_messages set status = :status, text = :text,"
                    " claims = cast(:claims as jsonb), standalone_query = :standalone,"
                    " warnings = cast(:warnings as jsonb), corpus_release_id = :release,"
                    " identity = cast(:identity as jsonb), trace = cast(:trace as jsonb),"
                    " cost_usd = :cost, paper_references = cast(:references as jsonb),"
                    " completed_at = :now where id = :id and status = 'pending'"
                ),
                {
                    "id": turn.assistant_message_id,
                    "status": status,
                    "text": answer_text(answer) if status == "complete" else "",
                    "claims": json.dumps([c.model_dump(mode="json") for c in answer.claims]),
                    "standalone": result.standalone_query,
                    "warnings": json.dumps(list(result.warnings)),
                    "release": result.corpus_release_id,
                    "identity": json.dumps(dict(result.identity), default=str),
                    "trace": json.dumps(dict(result.trace)),
                    "cost": round(result.cost_usd, 6),
                    # Ids, ranks and the read flag only: the metadata is the corpus's own.
                    "references": json.dumps(
                        [
                            {"paper_id": str(r.paper_id), "rank": r.rank, "read": r.read}
                            for r in result.references
                        ]
                    ),
                    "now": now,
                },
            ).rowcount
            if not updated:
                raise TurnConflict("turn_superseded")
            count_answer(connection, turn.assistant_message_id)
            papers = _paper_facts(connection, sorted({str(e.source_id) for e in result.evidence}))
            for position, item in enumerate(result.evidence):
                facts = papers.get(str(item.source_id), {})
                connection.execute(
                    text(
                        "insert into message_evidence (message_id, evidence_id, position,"
                        " paper_id, version_id, chunk_id, title, venue, year, section,"
                        " page_start, page_end, text, abstract_only, cited) values (:message,"
                        " :evidence, :position, :paper, :version, :chunk, :title, :venue, :year,"
                        " :section, :page_start, :page_end, :text, :abstract_only, :cited)"
                    ),
                    {
                        "message": turn.assistant_message_id,
                        "evidence": item.id,
                        "position": position,
                        "paper": item.source_id,
                        "version": item.version_id,
                        "chunk": item.chunk_id,
                        "title": facts.get("title") or "Untitled paper",
                        "venue": facts.get("venue"),
                        "year": facts.get("year"),
                        "section": item.section,
                        "page_start": item.page_start,
                        "page_end": item.page_end,
                        "text": item.text,
                        "abstract_only": item.abstract_only,
                        "cited": item.id in cited,
                    },
                )
            _touch(connection, turn.chat_id, now)
            message = _load_message(connection, turn.assistant_message_id)
        assert message is not None
        return message

    def fail_turn(
        self, turn: Turn, *, status: Literal["failed", "interrupted"], error_code: str
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    "update chat_messages set status = :status, error_code = :code,"
                    " completed_at = :now where id = :id and status = 'pending'"
                ),
                {
                    "id": turn.assistant_message_id,
                    "status": status,
                    "code": error_code,
                    "now": self._clock(),
                },
            )


def chat_stats(engine: Engine, *, day: date) -> dict[str, Any]:
    """One UTC day of assistant turns: counts by status, error and rating, stage timings, cost.

    Reads only status, error code, trace and cost: never a message's text or its user.
    """

    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    params = {"start": start, "end": start + timedelta(days=1)}
    window = "role = 'assistant' and created_at >= :start and created_at < :end"
    with engine.connect() as connection:
        statuses = {
            str(row.status): int(row.turns)
            for row in connection.execute(
                text(
                    f"select status, count(*) as turns from chat_messages where {window}"
                    " group by status order by status"
                ),
                params,
            )
        }
        error_codes = {
            str(row.error_code): int(row.turns)
            for row in connection.execute(
                text(
                    f"select error_code, count(*) as turns from chat_messages where {window}"
                    " and error_code is not null group by error_code order by error_code"
                ),
                params,
            )
        }
        stages = {
            str(row.stage): {"turns": int(row.turns), "p50": float(row.p50), "p95": float(row.p95)}
            for row in connection.execute(
                text(
                    "select stage.key as stage, count(*) as turns,"
                    " percentile_cont(0.5) within group (order by (stage.value #>> '{}')::float)"
                    " as p50,"
                    " percentile_cont(0.95) within group (order by (stage.value #>> '{}')::float)"
                    " as p95"
                    " from chat_messages, jsonb_each(coalesce(trace, '{}'::jsonb)) as stage"
                    f" where {window} and jsonb_typeof(stage.value) = 'number'"
                    " group by stage.key order by stage.key"
                ),
                params,
            )
        }
        cost = connection.execute(
            text(f"select coalesce(sum(cost_usd), 0) from chat_messages where {window}"), params
        ).scalar_one()
        ratings = connection.execute(
            text(
                "select count(*) filter (where feedback = 'helpful') as helpful,"
                " count(*) filter (where feedback = 'not_helpful') as not_helpful,"
                " count(*) filter (where feedback is null) as unrated"
                f" from chat_messages where {window} and status in ('complete', 'insufficient')"
            ),
            params,
        ).one()
    return {
        "day": day.isoformat(),
        "turns": sum(statuses.values()),
        "statuses": statuses,
        "error_codes": error_codes,
        "stages": stages,
        "cost_usd": round(float(cost), 6),
        # Answered turns of the day by their owner's rating (sign-up plan, A5).
        "feedback": {
            "helpful": int(ratings.helpful),
            "not_helpful": int(ratings.not_helpful),
            "unrated": int(ratings.unrated),
        },
    }


def _touch(connection: Connection, chat_id: UUID, now: datetime) -> None:
    connection.execute(
        text("update chats set updated_at = :now where id = :chat"), {"chat": chat_id, "now": now}
    )


def _interrupt(
    connection: Connection, *, now: datetime, before: datetime, chat_id: UUID | None
) -> int:
    scope = "" if chat_id is None else " and chat_id = :chat"
    return connection.execute(
        text(
            "update chat_messages set status = 'interrupted', error_code = 'turn_interrupted',"
            f" completed_at = :now where status = 'pending' and started_at < :before{scope}"
        ),
        {"now": now, "before": before, "chat": chat_id},
    ).rowcount


def _paper_facts(connection: Connection, paper_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not paper_ids:
        return {}
    rows = connection.execute(
        text(
            "select p.id::text as paper_id, p.title, p.publication_year as year, v.name as venue"
            " from papers p left join venues v on v.id = p.venue_id"
            " where p.id = any(cast(:ids as uuid[]))"
        ),
        {"ids": paper_ids},
    ).mappings()
    return {row["paper_id"]: dict(row) for row in rows}


def _evidence_for(
    connection: Connection, message_ids: list[UUID]
) -> dict[UUID, list[StoredEvidence]]:
    if not message_ids:
        return {}
    rows = connection.execute(
        text(
            "select message_id, evidence_id as id, paper_id, title, venue, year, section,"
            " page_start, page_end, text, abstract_only, cited from message_evidence"
            " where message_id = any(:ids) order by message_id, position"
        ),
        {"ids": message_ids},
    ).mappings()
    grouped: dict[UUID, list[StoredEvidence]] = {}
    for row in rows:
        values = dict(row)
        grouped.setdefault(values.pop("message_id"), []).append(
            StoredEvidence.model_validate(values)
        )
    return grouped


def _reference_facts(connection: Connection, rows: Sequence[Any]) -> dict[str, dict[str, Any]]:
    ids = {str(item["paper_id"]) for row in rows for item in (row.paper_references or [])}
    return _paper_facts(connection, sorted(ids))


def _references(row: Any, papers: Mapping[str, Mapping[str, Any]]) -> list[Reference]:
    """The stored ranking with the corpus's metadata; a paper no longer in it has no page to
    link to and is left out, its rank kept by the rest."""

    references = []
    for item in row.paper_references or []:
        facts = papers.get(str(item["paper_id"]))
        if facts is None:
            continue
        references.append(
            Reference(
                paper_id=item["paper_id"],
                rank=item["rank"],
                read=item["read"],
                title=facts["title"],
                venue=facts["venue"] or None,
                year=facts["year"],
            )
        )
    return references


def _message(
    row: Any, evidence: list[StoredEvidence], papers: Mapping[str, Mapping[str, Any]]
) -> ChatMessage:
    return ChatMessage(
        id=row.id,
        ordinal=row.ordinal,
        role=row.role,
        status=row.status,
        text=row.text,
        claims=[Claim.model_validate(item) for item in (row.claims or [])],
        evidence=evidence,
        references=_references(row, papers),
        standalone_query=row.standalone_query,
        warnings=list(row.warnings or []),
        error_code=row.error_code,
        corpus_release_id=row.corpus_release_id,
        created_at=row.created_at,
        feedback=row.feedback,
    )


def _load_message(connection: Connection, message_id: UUID) -> ChatMessage | None:
    row = connection.execute(
        text(f"select {_MESSAGE_COLUMNS} from chat_messages where id = :id"), {"id": message_id}
    ).first()
    if row is None:
        return None
    evidence = _evidence_for(connection, [message_id]).get(message_id, [])
    return _message(row, evidence, _reference_facts(connection, [row]))
