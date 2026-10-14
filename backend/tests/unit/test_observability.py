"""A request log line says what happened and nothing a user typed."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from copilot.observability import install_request_log, send_request_log_to_stdout
from copilot.search.api import request_id


def test_one_line_per_request_with_the_route_template_not_the_path(caplog):
    app = FastAPI()
    install_request_log(app)

    @app.get("/v1/chats/{chat_id}")
    def chat(chat_id: str) -> dict[str, str]:
        return {"ok": chat_id}

    with caplog.at_level(logging.INFO, logger="copilot.requests"):
        TestClient(app).get("/v1/chats/secret-looking-id?q=private+question")
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "copilot.requests"]
    assert len(lines) == 1
    line = lines[0]
    assert line["route"] == "/v1/chats/{chat_id}" and line["status"] == 200
    assert set(line) == {"request_id", "method", "route", "status", "ms"}
    assert "secret-looking-id" not in caplog.text and "private" not in caplog.text


def _lines(caplog) -> list[dict[str, object]]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "copilot.requests"]


def test_the_logged_request_id_is_the_one_the_endpoint_returned(caplog):
    app = FastAPI()
    install_request_log(app)

    @app.get("/v1/thing")
    def thing(request: Request) -> dict[str, str]:
        return {"request_id": str(request_id(request))}

    with caplog.at_level(logging.INFO, logger="copilot.requests"):
        body = TestClient(app).get("/v1/thing").json()
        TestClient(app).get("/v1/nowhere/private-path")
    found, missing = _lines(caplog)
    assert found["request_id"] == body["request_id"]
    assert missing == missing | {"route": "unmatched", "status": 404, "request_id": None}
    assert "private-path" not in caplog.text


def test_the_served_app_prints_request_lines_once_on_stdout(capsys):
    logger = logging.getLogger("copilot.requests")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    try:
        send_request_log_to_stdout()
        send_request_log_to_stdout()
        app = FastAPI()
        install_request_log(app)

        @app.get("/health/live")
        def live() -> dict[str, str]:
            return {"status": "ok"}

        TestClient(app).get("/health/live")
        printed = capsys.readouterr().out.splitlines()
        assert len(printed) == 1 and json.loads(printed[0])["route"] == "/health/live"
    finally:
        logger.handlers[:] = saved[0]
        logger.setLevel(saved[1])
        logger.propagate = saved[2]
