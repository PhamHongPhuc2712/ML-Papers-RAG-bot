"""OpenReview API v2 adapter for accepted-paper metadata and PDF links.

Membership: accepted notes carry the conference ``venueid`` (for example
``ICLR.cc/2024/Conference``) while rejected and withdrawn submissions carry
``.../Rejected_Submission`` and ``.../Withdrawn_Submission``. The adapter still
maps every note's venue label onto the decision enum so eligibility is checked
explicitly rather than assumed from the query.

Guest requests to the API now receive ``ChallengeRequiredError`` (HTTP 403), a
browser challenge a CLI cannot answer. A free OpenReview account avoids it:
``POST /login`` returns a bearer token that the adapter attaches to every
request. Credentials are read from the environment variables the manifest
names (default ``OPENREVIEW_USERNAME`` / ``OPENREVIEW_PASSWORD``) and are never
stored in records, logs or checkpoints.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from .base import (
    ProviderError,
    Transport,
    check_response,
    normalize_decision,
    presentation_type,
    utc_now,
)

DEFAULT_API_BASE = "https://api2.openreview.net"
DEFAULT_USERNAME_ENV = "OPENREVIEW_USERNAME"
DEFAULT_PASSWORD_ENV = "OPENREVIEW_PASSWORD"


def _ms_to_iso(value: object) -> str | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat()
    return None


def credentials_from_env(config: Mapping[str, Any]) -> tuple[str, str] | None:
    auth = config.get("auth", {})
    auth = auth if isinstance(auth, Mapping) else {}
    username = os.environ.get(str(auth.get("username_env", DEFAULT_USERNAME_ENV)), "").strip()
    password = os.environ.get(str(auth.get("password_env", DEFAULT_PASSWORD_ENV)), "")
    if username and password:
        return username, password
    return None


class OpenReviewSource:
    source = "openreview"

    def __init__(
        self,
        config: Mapping[str, Any],
        transport: Transport,
        *,
        venue: str,
        year: int,
        track: str,
        now: Callable[[], datetime] = utc_now,
        credentials: tuple[str, str] | None = None,
    ) -> None:
        self.api_base = str(config.get("api_base", DEFAULT_API_BASE)).rstrip("/")
        self.venue_id = str(config["venue_id"])
        self.page_size = int(config.get("page_size", 1000))
        self.pdf_base = str(config.get("pdf_base", "https://openreview.net/pdf?id="))
        self.forum_base = str(config.get("forum_base", "https://openreview.net/forum?id="))
        self.transport = transport
        self.venue = venue
        self.year = year
        self.track = track
        self.now = now
        self.credentials = credentials if credentials is not None else credentials_from_env(config)
        self._token: str | None = None

    def auth_headers(self) -> dict[str, str]:
        """Bearer header from a cached login; empty when running as a guest."""

        if self.credentials is None:
            return {}
        if self._token is None:
            username, password = self.credentials
            response = self.transport.post(
                f"{self.api_base}/login", json={"id": username, "password": password}
            )
            check_response(response)
            data = response.json()
            token = data.get("token") if isinstance(data, Mapping) else None
            if not isinstance(token, str) or not token:
                raise ProviderError("provider_error", "openreview login returned no token")
            self._token = token
        return {"Authorization": f"Bearer {self._token}"}

    def fetch_page(self, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        offset = int(cursor) if cursor else 0
        response = self.transport.get(
            f"{self.api_base}/notes",
            params={"content.venueid": self.venue_id, "limit": self.page_size, "offset": offset},
            headers=self.auth_headers(),
        )
        if response.status_code == 403 and self.credentials is None:
            raise ProviderError(
                "challenge_required",
                "OpenReview requires a logged-in session; set OPENREVIEW_USERNAME and "
                "OPENREVIEW_PASSWORD in .env",
            )
        check_response(response)
        data = response.json()
        notes = data.get("notes", []) if isinstance(data, Mapping) else []
        count = data.get("count") if isinstance(data, Mapping) else None
        records = [self.normalize(note) for note in notes if isinstance(note, Mapping)]
        next_offset = offset + len(notes)
        more = bool(notes) and (not isinstance(count, int) or next_offset < count)
        return records, (str(next_offset) if more else None)

    def normalize(self, note: Mapping[str, Any]) -> dict[str, Any]:
        content = note.get("content", {})
        if not isinstance(content, Mapping):
            content = {}

        def value(key: str) -> Any:
            item = content.get(key)
            return item.get("value") if isinstance(item, Mapping) else item

        note_id = str(note["id"])
        venue_label = value("venue")
        venue_id = str(value("venueid") or "")
        decision = normalize_decision(venue_label or venue_id)
        withdrawn = "withdrawn" in venue_id.lower() or decision == "withdrawn"
        authors = value("authors")
        revision = str(note.get("mdate") or note.get("tmdate") or note.get("cdate") or "unknown")
        return {
            "source": self.source,
            "source_item_id": note_id,
            "source_revision": revision,
            "external_ids": {"openreview": note_id},
            "title": str(value("title") or ""),
            "authors": [str(name) for name in authors] if isinstance(authors, list) else [],
            "abstract": value("abstract"),
            "venue": {"name": self.venue, "track": self.track},
            "venue_detail": venue_label,
            "year": self.year,
            "track": self.track,
            "decision": decision,
            "withdrawn": withdrawn,
            "acceptance_decision": presentation_type(venue_label) or decision,
            "retrieved_at": self.now().isoformat(),
            "source_url": f"{self.forum_base}{note_id}",
            "pdf_url": f"{self.pdf_base}{note_id}",
            "first_published_at": _ms_to_iso(note.get("cdate") or note.get("tcdate")),
            "redistribution": "unknown",
            "openreview_pdf": value("pdf"),
        }
