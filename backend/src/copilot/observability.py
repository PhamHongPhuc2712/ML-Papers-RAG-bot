"""Request logs (spec §12): one JSON line per request, never anything a user typed.

The route is the matched template (``/v1/chats/{chat_id}``), not the path, so ids and query
strings stay out of the log; bodies are never read. A plain ASGI middleware, not
``@app.middleware("http")``, so a streamed chat turn passes through untouched and the line is
written when the stream ends: ``ms`` is the whole turn.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("copilot.requests")


class RequestLog:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        status = 500

        async def capture(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            state: dict[str, Any] = scope.get("state") or {}
            request = state.get("request_id")
            logger.info(
                json.dumps(
                    {
                        "request_id": None if request is None else str(request),
                        "method": scope["method"],
                        "route": getattr(scope.get("route"), "path", "unmatched"),
                        "status": status,
                        "ms": round((time.monotonic() - started) * 1000, 1),
                    },
                    separators=(",", ":"),
                )
            )


def install_request_log(app: FastAPI) -> None:
    app.add_middleware(RequestLog)


def send_request_log_to_stdout() -> None:
    """Print request lines on stdout, one bare JSON object each, for the served app.

    Uvicorn configures only its own loggers; without this the root logger stays at WARNING
    and every INFO request line is dropped. Idempotent, and not propagated, so the line
    is printed once.
    """

    if any(getattr(h, "_copilot_requests", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._copilot_requests = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def send_app_logs_to_stderr() -> None:
    """Print the application's own INFO lines (``copilot.*``) on stderr in the served app.

    Uvicorn configures only its own loggers, so lines such as startup's warm-up report or
    the stale-turn sweep were dropped (R3). Request lines keep their own stdout handler.
    """

    app_logger = logging.getLogger("copilot")
    if any(getattr(h, "_copilot_app", False) for h in app_logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler._copilot_app = True  # type: ignore[attr-defined]
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
