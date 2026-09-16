"""Shared source-adapter contracts: eligibility, decisions, transports, manifests."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

DECISIONS = ("accepted", "rejected", "desk_rejected", "withdrawn", "under_review", "unknown")
PRESENTATION_TYPES = ("oral", "spotlight", "poster")

# Canonical tracks. Only "main" is the accepted main conference a manifest asks
# for; everything else is a real publication in a track this corpus excludes.
TRACKS = (
    "main",
    "findings",
    "workshop",
    "industry",
    "demo",
    "shared_task",
    "tutorial",
    "other",
    "unknown",
)

# The ACL Anthology encodes the track in the volume title, and a venue-year's
# rows mix the main conference with Findings, workshops, industry and demo
# volumes, and whole co-located conferences (WMT, IWSLT, ArabicNLP) that the
# registry still files under venue ACL or EMNLP. Main is therefore matched
# positively: an unrecognized label is "other", never silently the main track.
_ACL_MAIN_VOLUMES = (
    re.compile(
        r"^proceedings of the \d+(st|nd|rd|th) annual meeting of the association for "
        r"computational linguistics \(volume [12]: (long|short) papers\)$"
    ),
    re.compile(
        r"^proceedings of the \d{4} conference on empirical methods in natural language "
        r"processing$"
    ),
    re.compile(
        r"^proceedings of the \d{4} conference of the (north american chapter|nations of the "
        r"americas chapter) of the association for computational linguistics: human language "
        r"technologies \(volume [12]: (long|short) papers\)$"
    ),
)
_SUBMISSION = re.compile(r"submitted to|under review")
_PRESENTATION = re.compile(r"poster|oral|spotlight|notable top|^accept")


def classify_track(label: object) -> tuple[str, str]:
    """Map a venue's own label onto (canonical track, decision).

    The label is the only membership evidence the registry carries, and it is
    specific: OpenReview venues publish a presentation type for accepted work
    and "Submitted to <venue>" for the rest, while proceedings venues publish a
    volume title. Order matters — 163 of the 165 workshop labels begin with
    "Proceedings of", so the workshop test has to precede the volume patterns
    or 4,847 workshop papers would read as main-conference ones.
    """

    text = " ".join(str(label or "").split())
    if not text:
        return ("unknown", "unknown")
    lowered = text.lower()
    if _SUBMISSION.search(lowered):
        # An unaccepted submission still belongs to the venue's main track.
        return ("main", "under_review")
    if "findings" in lowered:
        return ("findings", "accepted")
    if "workshop" in lowered:
        return ("workshop", "accepted")
    if "shared task" in lowered:
        return ("shared_task", "accepted")
    if "industry track" in lowered:
        return ("industry", "accepted")
    if "system demonstration" in lowered or "demonstrations" in lowered:
        return ("demo", "accepted")
    if "tutorial" in lowered:
        return ("tutorial", "accepted")
    if _PRESENTATION.search(lowered):
        return ("main", "accepted")
    # Proceedings venues (AAAI, the CVF conferences, IJCAI, Interspeech, JMLR)
    # publish only accepted main-conference papers, and the registry labels
    # them "main"; their pdf_url hosts are the official proceedings sites.
    if lowered == "main":
        return ("main", "accepted")
    if any(pattern.match(lowered) for pattern in _ACL_MAIN_VOLUMES):
        return ("main", "accepted")
    return ("other", "accepted")


def normalize_decision(value: object) -> str:
    """Map an official decision or venue label onto the project's decision enum."""

    if value is None:
        return "unknown"
    text = str(value).strip().lower().replace("_", " ")
    if not text:
        return "unknown"
    if "withdraw" in text:
        return "withdrawn"
    if "desk" in text and "reject" in text:
        return "desk_rejected"
    if "reject" in text:
        return "rejected"
    if text.startswith("accept") or any(
        token in text for token in ("accepted", *PRESENTATION_TYPES)
    ):
        return "accepted"
    if any(token in text for token in ("under review", "submitted", "pending", "in review")):
        return "under_review"
    return "unknown"


def presentation_type(value: object) -> str | None:
    text = str(value or "").lower()
    return next((kind for kind in PRESENTATION_TYPES if kind in text), None)


def is_eligible(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    """Membership rule: accepted, not withdrawn, and in the manifest's venue-years/track."""

    # Normalized records carry venue as {"name", "track"}; flat rows carry strings.
    venue = record.get("venue")
    venue_name = venue.get("name") if isinstance(venue, Mapping) else venue
    track = record.get("track")
    if track is None and isinstance(venue, Mapping):
        track = venue.get("track")
    return (
        venue_name == manifest["venue"]
        and record.get("year") in manifest["years"]
        and track == manifest["track"]
        and normalize_decision(record.get("decision")) == "accepted"
        and not bool(record.get("withdrawn", False))
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def retry_after_seconds(headers: Mapping[str, str]) -> int | None:
    value = header(headers, "retry-after")
    if value is None:
        return None
    try:
        return max(0, int(float(value.strip())))
    except ValueError:
        return None


class ProviderError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


class ProviderThrottled(ProviderError):
    def __init__(self, retry_after: int | None = None, detail: str = "") -> None:
        self.retry_after = retry_after
        super().__init__("throttled", detail)


@dataclass
class ProviderResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        import json

        return json.loads(self.body)


class ProviderStream(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class Transport(Protocol):
    """Injected HTTP boundary so tests never touch the network."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ProviderResponse: ...

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ProviderResponse: ...

    def stream(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ProviderStream: ...


def check_response(response: ProviderResponse) -> None:
    if response.status_code in (429, 503):
        raise ProviderThrottled(retry_after_seconds(response.headers), str(response.status_code))
    if response.status_code >= 400:
        raise ProviderError("provider_error", str(response.status_code))


class _HttpxStream:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status_code = response.status_code
        self.headers: Mapping[str, str] = dict(response.headers)

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        return self._response.iter_bytes(chunk_size)

    def close(self) -> None:
        self._response.close()


class HttpxTransport:
    """Real transport: no automatic redirects, so download.py can vet every hop."""

    def __init__(self, *, timeout: float = 30.0, user_agent: str = "ml-research-copilot/0.1"):
        self.timeout = timeout
        self._client = httpx.Client(
            follow_redirects=False, timeout=timeout, headers={"User-Agent": user_agent}
        )

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ProviderResponse:
        response = self._client.get(
            url, params=dict(params or {}), headers=dict(headers or {}), timeout=timeout
        )
        return ProviderResponse(response.status_code, dict(response.headers), response.content)

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ProviderResponse:
        response = self._client.post(
            url, json=dict(json or {}), headers=dict(headers or {}), timeout=timeout
        )
        return ProviderResponse(response.status_code, dict(response.headers), response.content)

    def stream(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ProviderStream:
        request = self._client.build_request(
            "GET", url, headers=dict(headers or {}), timeout=timeout
        )
        return _HttpxStream(self._client.send(request, stream=True))

    def close(self) -> None:
        self._client.close()


class SourceAdapter(Protocol):
    source: str

    def fetch_page(self, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        """Return normalized records for one page and the next cursor, or None at the end."""


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load the venue-year source manifest and validate its required shape."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("manifest_invalid:root")
    manifest = dict(raw)
    if not isinstance(manifest.get("venue"), str) or not manifest["venue"]:
        raise ValueError("manifest_invalid:venue")
    years = manifest.get("years")
    if not isinstance(years, list) or not years or not all(isinstance(y, int) for y in years):
        raise ValueError("manifest_invalid:years")
    if not isinstance(manifest.get("track"), str):
        raise ValueError("manifest_invalid:track")
    if not isinstance(manifest.get("sources"), Mapping) or not manifest["sources"]:
        raise ValueError("manifest_invalid:sources")
    return manifest


def partition_key(manifest: Mapping[str, Any], year: int) -> str:
    return f"{manifest['venue']}:{year}:{manifest['track']}"
