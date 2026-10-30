"""The Qdrant client factory keeps its connections (found 2026-10-07)."""

from __future__ import annotations

from qdrant_client import QdrantClient

from copilot.search.client import MAX_KEEPALIVE_CONNECTIONS, qdrant_client


def _pool(client: QdrantClient):
    # The library's own layering: QdrantClient -> remote -> ApiClient -> httpx.Client.
    return client.http.client._client._transport._pool


def test_a_localhost_client_keeps_connections_alive():
    client = qdrant_client("http://localhost:6333", timeout=5)
    try:
        assert _pool(client)._max_keepalive_connections == MAX_KEEPALIVE_CONNECTIONS
    finally:
        client.close()


def test_the_library_default_would_not():
    client = QdrantClient(url="http://localhost:6333", timeout=5)
    try:
        assert _pool(client)._max_keepalive_connections == 0
    finally:
        client.close()
