"""Search, paper metadata and related papers over HTTP, against a real release."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from copilot.app import create_app, openapi_schema
from copilot.corpus.dedupe import merge_papers
from copilot.db.session import session_scope
from copilot.models.embeddings import FixtureEmbedding
from copilot.search.cache import OrderingCache
from copilot.search.dense import DenseRetriever
from copilot.search.index import CHUNKS, PAPERS, collection_names
from copilot.search.rerank import FixtureReranker
from copilot.search.service import SearchConfig, SearchService
from copilot.search.sparse import SparseRetriever

pytestmark = pytest.mark.integration


class Clock:
    """Wall-clock seconds the test moves by hand; nothing here ever sleeps."""

    def __init__(self, now: int = 1_800_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


@pytest.fixture(scope="module")
def clock() -> Clock:
    return Clock()


@pytest.fixture(scope="module")
def api_client(search_corpus, test_settings, clock) -> Iterator[TestClient]:
    """The app over the seeded release; module-scoped, so it never tears the release down."""

    app = create_app(
        test_settings,
        {"db_engine": search_corpus.engine, "qdrant_client": search_corpus.client, "clock": clock},
    )
    with TestClient(app) as client:
        yield client


def test_search_validation_and_metadata(api_client):
    invalid = api_client.post("/v1/search", json={"query": "", "mode": "hybrid"})
    assert invalid.status_code == 422
    response = api_client.post(
        "/v1/search",
        json={
            "query": "contrastive learning",
            "mode": "hybrid",
            "filters": {"venues": [], "fulltext_only": False},
            "limit": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) <= 2
    assert body["corpus_release_id"]
    assert all("paper_id" in item and "rank" in item for item in body["items"])


# --- helpers -------------------------------------------------------------------------------

QUERY = {"query": "graph contrastive learning", "mode": "hybrid", "filters": {}}


def _search(client, body=None, **changes):
    return client.post("/v1/search", json={**(body or QUERY), **changes})


def _all_pages(client, body, limit):
    pages = [_search(client, body, limit=limit).json()]
    while pages[-1]["next_cursor"]:
        response = _search(client, body, limit=limit, cursor=pages[-1]["next_cursor"])
        assert response.status_code == 200, response.text
        pages.append(response.json())
    return pages


def _assert_error(response, status, code, *, retryable=False):
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["retryable"] is retryable
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    UUID(body["request_id"])
    return body


def _insert_paper(engine, title, *, pdf_url=None, landing_url=None, year=2024) -> str:
    paper_id = str(uuid4())
    with engine.begin() as connection:
        connection.execute(
            text(
                "insert into papers (id, title, abstract, publication_year, first_seen_at,"
                " pdf_url, acceptance_type, metadata_status) values (:id, :title, 'An abstract.',"
                " :year, now(), :pdf, 'poster', 'active')"
            ),
            {"id": paper_id, "title": title, "year": year, "pdf": pdf_url},
        )
        if landing_url is not None:
            connection.execute(
                text(
                    "insert into paper_versions (id, paper_id, source, source_url,"
                    " source_revision, content_sha256, redistribution, parser_version,"
                    " parse_status) values (gen_random_uuid(), :paper, 'papercli', :url, 'rev',"
                    " :sha, 'unknown', 'pypdf-text-v2', 'not_pdf')"
                ),
                {"paper": paper_id, "url": landing_url, "sha": "e" * 64},
            )
        connection.execute(
            text(
                "insert into paper_identifiers (id, paper_id, namespace, value)"
                " values (gen_random_uuid(), :paper, 'openreview', :value)"
            ),
            {"paper": paper_id, "value": f"or-{paper_id[:8]}"},
        )
    return paper_id


def _delete_papers(engine, *paper_ids) -> None:
    with engine.begin() as connection:
        for table, column in (
            ("paper_redirects", "from_paper_id"),
            ("paper_redirects", "to_paper_id"),
            ("paper_identifiers", "paper_id"),
            ("paper_versions", "paper_id"),
        ):
            connection.execute(
                text(f"delete from {table} where {column} = any(cast(:ids as uuid[]))"),
                {"ids": list(paper_ids)},
            )
        connection.execute(
            text("update papers set merged_into = null where id = any(cast(:ids as uuid[]))"),
            {"ids": list(paper_ids)},
        )
        connection.execute(
            text("delete from papers where id = any(cast(:ids as uuid[]))"),
            {"ids": list(paper_ids)},
        )


@contextmanager
def _active(engine, release_id, paper_collection, chunk_collection):
    """Point the active release elsewhere for one test, then put the seeded one back."""

    with engine.begin() as connection:
        connection.execute(
            text(
                "insert into corpus_releases (id, manifest_sha256, paper_collection,"
                " chunk_collection, model_revision, status, counts) values (:id, :sha, :papers,"
                " :chunks, 'fixture/hashed-2d@v1#l2', 'staged', '{}')"
            ),
            {
                "id": release_id,
                "sha": "f" * 64,
                "papers": paper_collection,
                "chunks": chunk_collection,
            },
        )
        previous = connection.execute(
            text("select corpus_release_id from active_release where singleton_key = 'active'")
        ).scalar_one()
        connection.execute(
            text(
                "update active_release set corpus_release_id = :id where singleton_key = 'active'"
            ),
            {"id": release_id},
        )
    try:
        yield
    finally:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "update active_release set corpus_release_id = :id"
                    " where singleton_key = 'active'"
                ),
                {"id": previous},
            )
            connection.execute(
                text("delete from corpus_releases where id = :id"), {"id": release_id}
            )


# --- validation and the error shape --------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"query": "   "}, "query_blank"),
        ({"query": "x" * 2001}, "query"),
        ({"limit": 0}, "limit"),
        ({"limit": 51}, "limit"),
        ({"filters": {"year_from": 2025, "year_to": 2023}}, "invalid_year_interval"),
        ({"mode": "semantic"}, "mode"),
    ],
)
def test_invalid_requests_are_422_in_the_error_shape(api_client, changes, fragment):
    body = _assert_error(_search(api_client, **changes), 422, "validation_error")
    assert fragment in body["error"]["message"]


def test_a_validation_message_never_echoes_the_submitted_text(api_client):
    secret = "my unpublished idea " * 120
    response = _search(api_client, query=secret)
    _assert_error(response, 422, "validation_error")
    assert "unpublished" not in response.text


def test_successes_carry_a_fresh_request_id(api_client):
    first = _search(api_client).json()["request_id"]
    second = _search(api_client).json()["request_id"]
    assert UUID(first) != UUID(second)


# --- pagination ----------------------------------------------------------------------------


def test_pages_are_contiguous_stable_and_never_repeat(api_client):
    whole = _search(api_client, limit=50).json()
    pages = _all_pages(api_client, QUERY, limit=3)
    assert len(pages) > 1
    items = [item for page in pages for item in page["items"]]
    ids = [item["paper_id"] for item in items]
    assert len(ids) == len(set(ids))
    assert ids == [item["paper_id"] for item in whole["items"]]
    assert [item["rank"] for item in items] == list(range(1, len(items) + 1))
    assert pages[-1]["next_cursor"] is None
    assert all(set(page["papers"]) == {i["paper_id"] for i in page["items"]} for page in pages)
    assert len({page["request_id"] for page in pages}) == len(pages)


def test_a_repeated_search_reuses_its_cached_ordering(api_client, search_corpus):
    body = {**QUERY, "query": "diffusion image synthesis"}
    first = _search(api_client, body).json()
    second = _search(api_client, body).json()
    assert [i["paper_id"] for i in first["items"]] == [i["paper_id"] for i in second["items"]]
    with search_corpus.engine.connect() as connection:
        stored = connection.execute(
            text("select count(*) from search_orderings where id = :id"),
            {"id": first["request_id"]},
        ).scalar_one()
        repeats = connection.execute(
            text("select count(*) from search_orderings where id = :id"),
            {"id": second["request_id"]},
        ).scalar_one()
    assert (stored, repeats) == (1, 0)


def test_the_cache_stores_no_query_text(api_client, search_corpus):
    _search(api_client, query="a remarkably specific phrase about lemurs")
    with search_corpus.engine.connect() as connection:
        dump = connection.execute(
            text("select string_agg(t::text, ' ') from search_orderings t")
        ).scalar_one()
    assert "lemurs" not in dump


def test_a_degraded_ordering_pages_but_is_never_reused(search_corpus):
    cache = OrderingCache(search_corpus.engine, ttl_seconds=600)
    now = 1_900_000_000
    common = {
        "cache_key": "d" * 64,
        "query_key": "q" * 64,
        "release_id": search_corpus.release,
        "mode": "hybrid_rerank",
        "items": [(search_corpus.ids["Graph transformers at scale"], 0.5)],
        "scores": {},
        "now": now,
    }
    degraded = cache.put(entry_id=uuid4(), warnings=["rerank_timeout"], **common)
    assert cache.get(degraded.id, now + 1) is not None
    assert cache.find("d" * 64, now + 1) is None
    healthy = cache.put(entry_id=uuid4(), warnings=[], **common)
    assert cache.find("d" * 64, now + 1).id == healthy.id
    assert cache.get(healthy.id, now + 600) is None


def test_the_llm_mode_without_an_llm_degrades_and_is_never_reused(api_client, search_corpus):
    body = {**QUERY, "mode": "hybrid_rerank_llm"}

    def stored() -> int:
        with search_corpus.engine.connect() as connection:
            return connection.execute(text(
                "select count(*) from search_orderings where mode = 'hybrid_rerank_llm'"
            )).scalar_one()

    before = stored()
    first = _search(api_client, body)
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["degraded"] and "llm_rerank_unavailable" in payload["warnings"]
    assert payload["items"]
    second = _search(api_client, body)
    assert second.status_code == 200, second.text
    # Degraded orderings are stored for paging but never reused: each search ranks afresh.
    assert stored() == before + 2


def test_a_tampered_or_foreign_cursor_is_refused(api_client):
    cursor = _search(api_client, limit=2).json()["next_cursor"]
    payload, signature = cursor.split(".")
    flipped = payload[:-1] + ("A" if payload[-1] != "A" else "B")
    _assert_error(
        _search(api_client, limit=2, cursor=f"{flipped}.{signature}"), 422, "cursor_invalid"
    )
    _assert_error(_search(api_client, limit=2, cursor="garbage"), 422, "cursor_invalid")
    other = _search(api_client, limit=2, query="scaling laws", cursor=cursor)
    _assert_error(other, 422, "cursor_mismatch")


def test_an_expired_cursor_is_410_and_asks_for_a_fresh_search(api_client, clock):
    cursor = _search(api_client, limit=2).json()["next_cursor"]
    started = clock.now
    try:
        clock.now = started + 599
        assert _search(api_client, limit=2, cursor=cursor).status_code == 200
        clock.now = started + 600
        _assert_error(_search(api_client, limit=2, cursor=cursor), 410, "cursor_expired")
    finally:
        clock.now = started


def test_a_release_switch_leaves_open_pages_on_their_release(
    api_client, search_corpus, test_settings
):
    first = _search(api_client, limit=2).json()
    names = collection_names(test_settings.qdrant_collection_prefix, "search-b")
    with _active(search_corpus.engine, "search-b", names[PAPERS], names[CHUNKS]):
        page = _search(api_client, limit=2, cursor=first["next_cursor"])
        assert page.status_code == 200, page.text
        assert page.json()["corpus_release_id"] == search_corpus.release
        assert [i["rank"] for i in page.json()["items"]] == [3, 4]
        # A new search reads the new release, whose collections do not exist.
        _assert_error(_search(api_client, limit=2), 503, "candidates_unavailable", retryable=True)


def test_hybrid_rerank_pages_only_through_its_reranked_head(api_client):
    pages = _all_pages(api_client, {**QUERY, "mode": "hybrid_rerank"}, limit=5)
    items = [item for page in pages for item in page["items"]]
    assert items and all("rerank" in item["scores"] for item in items)


# --- private and foreign indexes -----------------------------------------------------------


@pytest.mark.parametrize(
    ("papers", "chunks"),
    [
        ("dev_paper_abstracts_search-x", "dev_paper_chunks_search-x"),
        ("test_user_documents_bge-m3", "test_user_documents_bge-m3-chunks"),
    ],
)
def test_only_this_deployments_public_collections_are_served(
    api_client, search_corpus, papers, chunks
):
    seed = search_corpus.ids["Graph neural networks for molecules"]
    with _active(search_corpus.engine, "search-x", papers, chunks):
        _assert_error(_search(api_client), 503, "release_outside_namespace")
        _assert_error(
            api_client.get(f"/v1/papers/{seed}/related"), 503, "release_outside_namespace"
        )


# --- paper metadata ------------------------------------------------------------------------


def test_paper_metadata_is_canonical_and_typed(api_client, search_corpus):
    paper_id = search_corpus.ids["Graph transformers at scale"]
    response = api_client.get(f"/v1/papers/{paper_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["paper_id"] == paper_id and body["redirected_from"] is None
    assert body["paper"]["title"] == "Graph transformers at scale"
    assert body["paper"]["authors"] == ["Ada Lovelace"]
    assert (body["paper"]["venue"], body["paper"]["year"]) == ("ICLR", 2024)
    assert body["paper"]["fulltext_indexed"] is True and body["parse_status"] == "parsed"
    assert body["source"] == {"pdf_url": None, "landing_url": "u", "available": True}
    UUID(body["request_id"])


def test_an_unknown_paper_is_404(api_client):
    _assert_error(api_client.get(f"/v1/papers/{uuid4()}"), 404, "paper_not_found")
    _assert_error(api_client.get(f"/v1/papers/{uuid4()}/related"), 404, "paper_not_found")
    _assert_error(api_client.get("/v1/papers/not-a-uuid"), 422, "validation_error")


def test_a_merged_paper_resolves_to_its_survivor(api_client, search_corpus):
    engine = search_corpus.engine
    survivor = _insert_paper(engine, "Survivor", pdf_url="https://example.org/s.pdf")
    loser = _insert_paper(engine, "Merged away")
    try:
        with session_scope(engine) as session:
            merge_papers(UUID(survivor), UUID(loser), session, reason="test")
        body = api_client.get(f"/v1/papers/{loser}").json()
        assert (body["paper_id"], body["redirected_from"]) == (survivor, loser)
        assert body["paper"]["title"] == "Survivor"
        # The loser's identifiers moved with the merge.
        assert {i["value"] for i in body["identifiers"]} == {
            f"or-{survivor[:8]}",
            f"or-{loser[:8]}",
        }
        assert api_client.get(f"/v1/papers/{survivor}").json()["redirected_from"] is None
    finally:
        _delete_papers(engine, loser, survivor)


def test_a_paper_without_any_source_link_says_so(api_client, search_corpus):
    paper_id = _insert_paper(search_corpus.engine, "Nowhere to read it")
    try:
        body = api_client.get(f"/v1/papers/{paper_id}").json()
        assert body["source"] == {"pdf_url": None, "landing_url": None, "available": False}
        assert body["parse_status"] is None and body["paper"]["fulltext_indexed"] is False
        linked = _insert_paper(
            search_corpus.engine, "Blocked PDF", landing_url="https://example.org/abs"
        )
        detail = api_client.get(f"/v1/papers/{linked}").json()
        assert detail["source"]["available"] is True and detail["parse_status"] == "not_pdf"
        _delete_papers(search_corpus.engine, linked)
    finally:
        _delete_papers(search_corpus.engine, paper_id)


# --- related papers ------------------------------------------------------------------------


def test_related_papers_exclude_the_seed_and_rank_by_dense_score(api_client, search_corpus):
    seed = search_corpus.ids["Graph neural networks for molecules"]
    body = api_client.get(f"/v1/papers/{seed}/related").json()
    ids = [item["paper_id"] for item in body["items"]]
    assert seed not in ids and len(ids) == len(search_corpus.ids) - 1
    scores = [item["scores"]["dense"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert body["corpus_release_id"] == search_corpus.release and not body["degraded"]


def test_related_papers_honour_filters_and_limit(api_client, search_corpus):
    seed = search_corpus.ids["Graph neural networks for molecules"]
    body = api_client.get(
        f"/v1/papers/{seed}/related", params={"year_from": 2024, "venues": ["ACL", "NeurIPS"]}
    ).json()
    assert body["items"]
    assert all(
        paper["year"] >= 2024 and paper["venue"] in {"ACL", "NeurIPS"}
        for paper in body["papers"].values()
    )
    assert (
        len(api_client.get(f"/v1/papers/{seed}/related", params={"limit": 2}).json()["items"]) == 2
    )
    invalid = api_client.get(
        f"/v1/papers/{seed}/related", params={"year_from": 2025, "year_to": 2024}
    )
    assert (
        "invalid_year_interval"
        in _assert_error(invalid, 422, "validation_error")["error"]["message"]
    )
    _assert_error(
        api_client.get(f"/v1/papers/{seed}/related", params={"limit": 51}), 422, "validation_error"
    )


def test_a_seed_outside_the_release_is_reported_not_guessed(api_client, search_corpus):
    paper_id = _insert_paper(search_corpus.engine, "Added after the release")
    try:
        body = api_client.get(f"/v1/papers/{paper_id}/related").json()
        assert body["items"] == [] and body["warnings"] == ["seed_not_indexed"]
    finally:
        _delete_papers(search_corpus.engine, paper_id)


# --- the published contract ----------------------------------------------------------------

SCHEMA = Path(__file__).resolve().parents[3] / "frontend" / "openapi.json"


def _structure(schema):
    """Paths, methods, statuses and each schema's fields — not generator ordering or prose."""

    paths = {
        path: {method: sorted(operation["responses"]) for method, operation in item.items()}
        for path, item in schema["paths"].items()
    }
    schemas = {
        name: {
            "required": sorted(definition.get("required", [])),
            "properties": sorted(definition.get("properties", {})),
        }
        for name, definition in schema["components"]["schemas"].items()
    }
    return {"version": schema["info"]["version"], "paths": paths, "schemas": schemas}


def test_the_published_schema_matches_the_running_contract():
    live = _structure(openapi_schema())
    assert live == _structure(json.loads(SCHEMA.read_text(encoding="utf-8"))), (
        "run: uv run --project backend python -m copilot.cli api export-schema"
        " --out frontend/openapi.json"
    )
    assert live["paths"]["/v1/search"] == {"post": ["200", "410", "422", "503"]}
    assert live["paths"]["/v1/papers/{paper_id}"] == {"get": ["200", "404", "422", "503"]}
    assert live["paths"]["/v1/papers/{paper_id}/related"] == {"get": ["200", "404", "422", "503"]}
    assert live["schemas"]["ErrorResponse"]["required"] == ["error", "request_id"]
    assert live["schemas"]["SearchResponse"]["required"] == sorted(
        ["request_id", "corpus_release_id", "items", "papers", "degraded", "warnings"]
    )


def test_startup_warms_every_stage_on_the_active_release(search_corpus, test_settings):
    """Found by the real-corpus smoke: unwarmed, the first request lost its dense branch."""

    seen: list[tuple[str, str]] = []

    class Seen:
        def __init__(self, name, inner) -> None:
            self.name, self.inner = name, inner

        def search(self, query, filters, limit, release_id):
            seen.append((self.name, release_id))
            return self.inner.search(query, filters, limit, release_id)

    class SeenReranker(FixtureReranker):
        def score(self, query, texts):
            seen.append(("rerank", search_corpus.release))
            return super().score(query, texts)

    engine, client = search_corpus.engine, search_corpus.client
    service = SearchService(
        engine=engine,
        lexical=Seen("lexical", SparseRetriever(engine, client)),
        dense=Seen("dense", DenseRetriever(engine, client, FixtureEmbedding())),
        reranker=SeenReranker(),
        config=SearchConfig(),
    )
    app = create_app(
        test_settings, {"db_engine": engine, "qdrant_client": client, "search_service": service}
    )
    assert seen == []
    with TestClient(app):
        assert sorted(seen) == sorted(
            (name, search_corpus.release) for name in ("lexical", "dense", "rerank")
        )
