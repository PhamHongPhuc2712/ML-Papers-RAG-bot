"""A chat turn over HTTP: streamed, stored, idempotent, owned, and honest when it fails."""

from __future__ import annotations

import json
from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from copilot.app import create_app
from copilot.chat.condense import Condenser, load_condense_prompt
from copilot.chat.graph import ChatGraph, build_chat_graph
from copilot.chat.retriever import CorpusRetriever, PaperHits
from copilot.chat.service import ChatService
from copilot.chat.store import ChatStore
from copilot.contracts import ChatRequest, Evidence, Principal, Reference
from copilot.models.generation import FakeGenerator
from copilot.models.llm import LlmError

pytestmark = pytest.mark.integration
TITLE = "Contrastive sentence embeddings"


class Calls:
    def __init__(self) -> None:
        self.generations = 0


class CountingGenerator(FakeGenerator):
    def __init__(self, calls: Calls, fail: Exception | None = None) -> None:
        super().__init__()
        self._counter = calls
        self._fail = fail

    def generate(self, question, evidence, *, request_id=None, run_id=None):
        self._counter.generations += 1
        if self._fail is not None:
            raise self._fail
        return super().generate(question, evidence, request_id=request_id, run_id=run_id)


def _graph(paper_id: str, calls: Calls, fail: Exception | None = None) -> ChatGraph:
    evidence = [
        Evidence(
            id="E1",
            source_kind="paper",
            source_id=paper_id,
            version_id="v1",
            chunk_id=uuid4(),
            text="Sentence pairs are contrasted.",
            section="Method",
            page_start=1,
            page_end=1,
        )
    ]
    retriever = CorpusRetriever(
        search=lambda query: PaperHits(
            "search-a",
            [paper_id],
            [],
            [
                Reference(paper_id=paper_id, rank=1, read=True, title=TITLE),
                Reference(paper_id=uuid4(), rank=2, read=False, title="Gone since"),
            ],
        ),
        evidence=lambda query, papers, release: list(evidence),
    )
    graph = build_chat_graph(
        condenser=Condenser(
            None, load_condense_prompt("prompts/chat/condense-v1.yaml"), max_chars=6000
        ),
        retriever=retriever,
        generator=CountingGenerator(calls, fail),
        condense_enabled=True,
    )
    return ChatGraph(graph=graph, identity={"chat_config": "test"}, condense=True, history_turns=3)


@pytest.fixture
def chat_client(test_settings, search_corpus, test_qdrant):
    def make(graph: ChatGraph) -> TestClient:
        app = create_app(
            test_settings,
            {"db_engine": search_corpus.engine, "qdrant_client": test_qdrant, "chat_graph": graph},
        )
        return TestClient(app)

    return make


def _frames(response) -> list[tuple[str, dict]]:
    frames = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        frames.append((lines["event"], json.loads(lines["data"])))
    return frames


def _post(client, headers, **body):
    payload = {"request_id": str(uuid4()), "message": "How are sentence pairs used?", **body}
    return client.post("/v1/chats/stream", json=payload, headers=headers), payload


def test_a_turn_streams_then_is_in_the_history(chat_client, search_corpus, make_user, auth_headers):
    calls = Calls()
    alice, _ = make_user()
    with chat_client(_graph(search_corpus.ids[TITLE], calls)) as client:
        response, _ = _post(client, auth_headers(alice))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _frames(response)
        assert [name for name, _ in frames] == [
            "chat",
            "status",
            "references",
            "evidence",
            "status",
            "final",
        ]
        assert frames[1][1] == {"stage": "searching"}
        sent = frames[2][1]
        assert sent["query"] == "How are sentence pairs used?"
        assert [(r["rank"], r["read"], r["title"]) for r in sent["items"]] == [
            (1, True, TITLE),
            (2, False, "Gone since"),
        ]
        final = frames[-1][1]
        assert [(r["rank"], r["title"]) for r in final["references"]] == [(1, TITLE)]
        assert final["status"] == "complete" and final["claims"][0]["evidence_ids"] == ["E1"]
        assert final["evidence"][0]["title"] == TITLE and final["evidence"][0]["cited"]
        chat_id = frames[0][1]["chat_id"]
        detail = client.get(f"/v1/chats/{chat_id}", headers=auth_headers(alice)).json()
        assert [m["status"] for m in detail["messages"]] == ["complete", "complete"]
        # Stored as ids and ranks; read back with the corpus's metadata. A paper no longer
        # in the corpus has no page to link to and is left out of the stored list.
        stored = detail["messages"][-1]["references"]
        assert stored == final["references"]
        assert stored[0]["paper_id"] == search_corpus.ids[TITLE] and stored[0]["read"]


def test_the_same_request_replays_without_running_again(
    chat_client, search_corpus, make_user, auth_headers
):
    calls = Calls()
    alice, _ = make_user()
    with chat_client(_graph(search_corpus.ids[TITLE], calls)) as client:
        first, payload = _post(client, auth_headers(alice))
        again = client.post("/v1/chats/stream", json=payload, headers=auth_headers(alice))
        assert [n for n, _ in _frames(again)] == ["chat", "final"]
        assert _frames(again)[-1][1] == _frames(first)[-1][1]
        assert calls.generations == 1


def test_the_spend_cap_is_a_typed_error_and_the_turn_can_run_again(
    chat_client, search_corpus, make_user, auth_headers, migrated_database
):
    alice, _ = make_user()
    calls = Calls()
    with chat_client(
        _graph(search_corpus.ids[TITLE], calls, fail=LlmError("llm_spend_cap"))
    ) as client:
        response, payload = _post(client, auth_headers(alice))
        name, data = _frames(response)[-1]
        assert name == "error" and data["error"]["code"] == "llm_spend_cap"
        assert data["error"]["retryable"] is False and "00:00 UTC" in data["error"]["message"]
        chat_id = _frames(response)[0][1]["chat_id"]
        messages = client.get(f"/v1/chats/{chat_id}", headers=auth_headers(alice)).json()[
            "messages"
        ]
        assert messages[-1]["status"] == "failed" and messages[-1]["claims"] == []
    with chat_client(_graph(search_corpus.ids[TITLE], calls)) as client:
        retry = client.post("/v1/chats/stream", json=payload, headers=auth_headers(alice))
        assert _frames(retry)[-1][0] == "final"


def test_a_closed_stream_leaves_the_turn_interrupted(search_corpus, make_user):
    alice = Principal(user_id=make_user()[0])
    store = ChatStore(search_corpus.engine)
    service = ChatService(
        store, _graph(search_corpus.ids[TITLE], Calls()), stale_after=timedelta(minutes=4)
    )
    events = service.stream(alice, ChatRequest(request_id=uuid4(), message="Hello?"))
    chat_id = json.loads(next(events).split("data: ", 1)[1])["chat_id"]
    events.close()
    _, messages = store.get_chat(alice, chat_id)
    assert messages[-1].status == "interrupted" and messages[-1].error_code == "client_disconnected"


def test_another_users_chat_is_404_and_a_running_request_is_409(
    chat_client, search_corpus, make_user, auth_headers
):
    alice, _ = make_user()
    bob, _ = make_user()
    with chat_client(_graph(search_corpus.ids[TITLE], Calls())) as client:
        response, _ = _post(client, auth_headers(alice))
        chat_id = _frames(response)[0][1]["chat_id"]
        stolen, _ = _post(client, auth_headers(bob), chat_id=chat_id)
        assert stolen.status_code == 404 and stolen.json()["error"]["code"] == "chat_not_found"
    store = ChatStore(search_corpus.engine)
    request = ChatRequest(request_id=uuid4(), message="Pending one")
    store.begin_turn(Principal(user_id=alice), request, stale_after=timedelta(minutes=4))
    with chat_client(_graph(search_corpus.ids[TITLE], Calls())) as client:
        busy = client.post(
            "/v1/chats/stream",
            json={"request_id": str(request.request_id), "message": "Pending one"},
            headers=auth_headers(alice),
        )
        assert busy.status_code == 409 and busy.json()["error"]["code"] == "request_in_progress"


def test_bad_input_is_rejected_before_anything_runs(
    chat_client, search_corpus, make_user, auth_headers
):
    alice, _ = make_user()
    calls = Calls()
    with chat_client(_graph(search_corpus.ids[TITLE], calls)) as client:
        too_long, _ = _post(client, auth_headers(alice), message="x" * 2001)
        assert too_long.status_code == 422
        blank, _ = _post(client, auth_headers(alice), message="   ")
        assert blank.status_code == 422
        anonymous, _ = _post(client, {})
        assert anonymous.status_code == 401
    assert calls.generations == 0
