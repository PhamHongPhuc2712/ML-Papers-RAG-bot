"""One chat turn as server-sent events, persisted before its answer is sent.

The turn is begun (and refused with an HTTP status) before streaming starts. Then frames
follow the graph: ``chat``, ``status`` as each step starts, ``references`` (the query searched
and the papers it ranked) and ``evidence`` once passages are found, and ``final`` with the
stored, validated message — or ``error``. The answer is stored before the ``final`` frame is
yielded, so a client that disconnects after the answer exists
still finds it in the history; one that disconnects earlier leaves the turn ``interrupted``.
GPU and network work runs in Starlette's thread pool, never on the event loop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from datetime import timedelta
from typing import Any

from sqlalchemy.exc import OperationalError

from ..contracts import ChatRequest, Evidence, Principal
from ..evidence.retrieve import EvidenceError
from ..models.generation import GenerationOutcome
from ..models.llm import LlmError
from ..search.api import ApiError
from ..search.service import SearchUnavailable
from .graph import ChatGraph
from .store import ChatStore, Turn
from .turns import TurnResult

logger = logging.getLogger(__name__)

MESSAGES = {
    "llm_spend_cap": "Today's answer budget is used up. It resets at 00:00 UTC.",
    "llm_timeout": "The answer took too long. Try again.",
    "llm_rate_limited": "The model is busy right now. Try again in a minute.",
    "llm_server_error": "The model provider failed. Try again.",
    "database_unavailable": "The database is unavailable. Try again.",
}
DEFAULT_MESSAGE = "Something went wrong while answering. Try again."


def frame(event: str, data: Mapping[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


def classify(error: Exception) -> tuple[str, bool]:
    if isinstance(error, LlmError | SearchUnavailable | ApiError):
        return error.code, error.retryable
    if isinstance(error, EvidenceError):
        return error.code, False
    if isinstance(error, OperationalError):
        return "database_unavailable", True
    return "chat_internal_error", False


def _preview(item: Evidence) -> dict[str, Any]:
    return {
        "id": item.id,
        "paper_id": str(item.source_id),
        "section": item.section,
        "page_start": item.page_start,
        "page_end": item.page_end,
    }


class ChatService:
    def __init__(self, store: ChatStore, graph: ChatGraph, *, stale_after: timedelta) -> None:
        self._store = store
        self._graph = graph
        self._stale_after = stale_after

    def stream(self, principal: Principal, body: ChatRequest) -> Iterator[str]:
        """Begin the turn now (errors raise here, before any frame), then stream it."""

        turn = self._store.begin_turn(principal, body, stale_after=self._stale_after)
        return self._events(turn, body)

    def _events(self, turn: Turn, body: ChatRequest) -> Iterator[str]:
        settled = False
        try:
            yield frame(
                "chat",
                {"chat_id": str(turn.chat_id), "message_id": str(turn.assistant_message_id)},
            )
            if turn.replay is not None:
                settled = True
                yield frame("final", turn.replay.model_dump(mode="json"))
                return
            history = self._store.history(
                turn.chat_id, before_ordinal=turn.ordinal - 1, turns=self._graph.history_turns
            )
            first = "condensing" if history and self._graph.condense else "searching"
            yield frame("status", {"stage": first})
            initial = {
                "request_id": body.request_id,
                "message": body.message,
                "history": history,
                "warnings": [],
                "timings": {},
                "cost_usd": 0.0,
            }
            state: dict[str, Any] = {}
            query = body.message
            for mode, chunk in self._graph.graph.stream(initial, stream_mode=["updates", "values"]):
                if mode == "values":
                    state = chunk
                    continue
                for node, update in chunk.items():
                    if node == "condense":
                        query = update["standalone"]
                        if first == "condensing":
                            yield frame("status", {"stage": "searching", "query": query})
                    elif node == "retrieve":
                        # What was searched and the papers it ranked, before the answer exists.
                        yield frame(
                            "references",
                            {
                                "query": query,
                                "items": [
                                    item.model_dump(mode="json")
                                    for item in update.get("references", [])
                                ],
                            },
                        )
                        yield frame(
                            "evidence",
                            {
                                "items": [_preview(item) for item in update["evidence"]],
                                "warnings": update.get("warnings", []),
                            },
                        )
                        yield frame("status", {"stage": "writing"})
            outcome: GenerationOutcome = state["outcome"]
            message = self._store.complete_turn(
                turn,
                TurnResult(
                    answer=outcome.answer,
                    evidence=state.get("evidence", []),
                    standalone_query=state["standalone"],
                    corpus_release_id=state.get("release_id"),
                    warnings=state.get("warnings", []),
                    identity=self._graph.identity,
                    trace=state.get("timings", {}),
                    cost_usd=float(state.get("cost_usd", 0.0)),
                    references=state.get("references", []),
                ),
            )
            settled = True
            yield frame("final", message.model_dump(mode="json"))
        except Exception as error:  # noqa: BLE001 - every failure ends the turn typed
            code, retryable = classify(error)
            if code == "chat_internal_error":
                logger.exception("chat turn failed", extra={"chat_id": str(turn.chat_id)})
            self._store.fail_turn(turn, status="failed", error_code=code)
            settled = True
            yield frame(
                "error",
                {
                    "error": {
                        "code": code,
                        "message": MESSAGES.get(code, DEFAULT_MESSAGE),
                        "retryable": retryable,
                    }
                },
            )
        finally:
            # GeneratorExit (the client went away) is not an Exception: it lands here.
            if not settled:
                self._store.fail_turn(turn, status="interrupted", error_code="client_disconnected")
