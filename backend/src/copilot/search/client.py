"""The one way this project opens a Qdrant client.

``qdrant-client`` turns keep-alive off for any ``localhost`` URL, so every request
opens a new TCP connection. Here Qdrant always is localhost, in Docker, and on this
WSL host a burst of a few thousand such connections exhausts the port proxy: the
peer resets connections for ten to twenty seconds. Found on 2026-10-07, when an
evaluation over a small corpus lost 485 to 751 queries of one variant to
``Connection reset by peer`` while the sequential variants before it ran clean.
The API serves from the same URL, so the same burst would have hit it.

A bounded keep-alive pool is passed explicitly, which the library accepts in place
of its default. Nothing else about the client changes.
"""

from __future__ import annotations

import httpx
from qdrant_client import QdrantClient

# Two candidate branches plus the reranker's metadata reads and a build's upserts
# never need more than a handful of concurrent connections.
MAX_CONNECTIONS = 16
MAX_KEEPALIVE_CONNECTIONS = 8
KEEPALIVE_EXPIRY_SECONDS = 30.0


def connection_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=MAX_CONNECTIONS,
        max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS,
    )


def qdrant_client(url: str, *, api_key: str | None = None, timeout: int = 60) -> QdrantClient:
    """A REST client that reuses its connections, whatever host the URL names."""

    return QdrantClient(url=url, api_key=api_key, timeout=timeout, limits=connection_limits())
