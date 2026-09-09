"""Hardened PDF download: scheme, host, resolved-IP and redirect checks, size cap, magic bytes."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

from .sources.base import (
    ProviderError,
    ProviderStream,
    ProviderThrottled,
    Transport,
    header,
    retry_after_seconds,
)

DEFAULT_ALLOWED_HOSTS = frozenset(
    {
        "openreview.net",
        "arxiv.org",
        "export.arxiv.org",
        "proceedings.mlr.press",
        "papers.nips.cc",
        "proceedings.neurips.cc",
        "openaccess.thecvf.com",
        "aclanthology.org",
    }
)
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
PDF_CONTENT_TYPES = frozenset(
    {"application/pdf", "application/x-pdf", "application/octet-stream", ""}
)

Resolver = Callable[[str], Sequence[str]]


class DownloadPolicyError(Exception):
    """Non-retryable policy violation; ``reason`` is one short classification token."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class Target:
    url: str
    scheme: str
    host: str


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    size: int
    content_type: str
    final_url: str


def default_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise DownloadPolicyError("address", f"cannot resolve {host}") from exc
    return sorted({str(info[4][0]) for info in infos})


def _check_address(text: str, host: str) -> None:
    try:
        address = ipaddress.ip_address(text)
    except ValueError as exc:
        raise DownloadPolicyError("address", f"{host} resolved to invalid address") from exc
    if not address.is_global:
        raise DownloadPolicyError("address", f"{host} resolves to non-public {address}")


def validate_download_url(
    url: str, allowed_hosts: Sequence[str] | frozenset[str], resolver: Resolver = default_resolver
) -> Target:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise DownloadPolicyError("scheme", parts.scheme or "missing")
    if parts.username or parts.password:
        raise DownloadPolicyError("credentials", "userinfo in URL")
    host = (parts.hostname or "").lower()
    if not host:
        raise DownloadPolicyError("host", "missing host")
    if host not in {item.lower() for item in allowed_hosts}:
        raise DownloadPolicyError("host", host)
    try:
        ipaddress.ip_address(host)
    except ValueError:
        for resolved in resolver(host) or ():
            _check_address(resolved, host)
        else:
            if not resolver(host):
                raise DownloadPolicyError("address", f"{host} did not resolve")
    else:
        _check_address(host, host)
    return Target(url=url, scheme=parts.scheme, host=host)


def download_pdf(
    url: str,
    dest_dir: Path,
    transport: Transport,
    *,
    allowed_hosts: Sequence[str] | frozenset[str] = DEFAULT_ALLOWED_HOSTS,
    max_bytes: int,
    resolver: Resolver = default_resolver,
    max_redirects: int = 5,
    timeout: float = 60.0,
    headers: Mapping[str, str] | None = None,
) -> DownloadResult:
    """Stream a PDF to ``dest_dir/<sha256>.pdf`` after vetting every redirect hop."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    origin_host = (urlsplit(url).hostname or "").lower()
    current = url
    for _ in range(max_redirects + 1):
        target = validate_download_url(current, allowed_hosts, resolver)
        # Credentials never follow a redirect to another host.
        hop_headers = dict(headers or {}) if target.host == origin_host else {}
        stream = transport.stream(target.url, headers=hop_headers, timeout=timeout)
        try:
            if stream.status_code in REDIRECT_CODES:
                location = header(stream.headers, "location")
                if not location:
                    raise DownloadPolicyError("redirects", "redirect without location")
                current = urljoin(target.url, location)
                continue
            if stream.status_code in (429, 503):
                raise ProviderThrottled(
                    retry_after_seconds(stream.headers), str(stream.status_code)
                )
            if stream.status_code != 200:
                raise ProviderError("download_http", str(stream.status_code))
            content_type = (
                (header(stream.headers, "content-type") or "").split(";")[0].strip().lower()
            )
            if content_type not in PDF_CONTENT_TYPES:
                raise DownloadPolicyError("content", content_type)
            return _store(stream, dest_dir, max_bytes, content_type, target.url)
        finally:
            stream.close()
    raise DownloadPolicyError("redirects", "too many redirects")


def _store(
    stream: ProviderStream, dest_dir: Path, max_bytes: int, content_type: str, final_url: str
) -> DownloadResult:
    hasher = hashlib.sha256()
    size = 0
    head = b""
    temporary = dest_dir / f".download-{uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            for chunk in stream.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise DownloadPolicyError("size", f"exceeds {max_bytes} bytes")
                if len(head) < 1024:
                    head += chunk[: 1024 - len(head)]
                hasher.update(chunk)
                handle.write(chunk)
        if b"%PDF-" not in head:
            raise DownloadPolicyError("content", "missing PDF signature")
        digest = hasher.hexdigest()
        final = dest_dir / f"{digest}.pdf"
        if not final.exists():
            os.replace(temporary, final)
        return DownloadResult(final, digest, size, content_type, final_url)
    finally:
        if temporary.exists():
            temporary.unlink()
