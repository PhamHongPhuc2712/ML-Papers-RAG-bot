"""Pure eligibility, decision mapping, backoff and download-policy tests."""

from __future__ import annotations

import pytest

from copilot.corpus.download import DownloadPolicyError, validate_download_url
from copilot.corpus.sources.base import is_eligible, normalize_decision
from copilot.jobs.queue import backoff_seconds


def test_under_review_is_not_accepted():
    manifest = {"venue": "ICLR", "years": [2024], "track": "main"}
    row = {
        "venue": "ICLR",
        "year": 2024,
        "track": "main",
        "decision": "under_review",
        "withdrawn": False,
    }
    assert not is_eligible(row, manifest)
    assert is_eligible({**row, "decision": "accepted"}, manifest)
    assert not is_eligible({**row, "decision": "accepted", "withdrawn": True}, manifest)


def test_wrong_venue_year_or_track_is_ineligible():
    manifest = {"venue": "ICLR", "years": [2024], "track": "main"}
    accepted = {"venue": "ICLR", "year": 2024, "track": "main", "decision": "accepted"}
    assert is_eligible(accepted, manifest)
    assert not is_eligible({**accepted, "venue": "ICML"}, manifest)
    assert not is_eligible({**accepted, "year": 2023}, manifest)
    assert not is_eligible({**accepted, "track": "workshop"}, manifest)
    assert not is_eligible({**accepted, "decision": "rejected"}, manifest)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Accept (poster)", "accepted"),
        ("Accept (oral)", "accepted"),
        ("Accept (spotlight)", "accepted"),
        ("ICLR 2024 poster", "accepted"),
        ("ICLR 2024 oral", "accepted"),
        ("Reject", "rejected"),
        ("ICLR 2024 Conference Rejected Submission", "rejected"),
        ("Desk Reject", "desk_rejected"),
        ("Withdrawn", "withdrawn"),
        ("ICLR 2024 Conference Withdrawn Submission", "withdrawn"),
        ("Under review", "under_review"),
        ("Submitted to ICLR 2024", "under_review"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_official_decisions_map_to_the_enum(raw, expected):
    assert normalize_decision(raw) == expected


def test_backoff_is_exponential_capped_and_honours_longer_retry_after():
    assert [backoff_seconds(attempt) for attempt in (1, 2, 3, 4, 5, 6, 10)] == [
        2,
        4,
        8,
        16,
        32,
        60,
        60,
    ]
    assert backoff_seconds(1, retry_after=120) == 120
    assert backoff_seconds(6, retry_after=5) == 60


def test_download_policy_rejects_unsafe_targets():
    allowed = {"openreview.net", "arxiv.org"}

    def public(host: str) -> list[str]:
        return ["93.184.216.34"]

    def private(host: str) -> list[str]:
        return ["10.0.0.5"]

    def mixed(host: str) -> list[str]:
        return ["93.184.216.34", "169.254.169.254"]

    target = validate_download_url("https://openreview.net/pdf?id=x", allowed, public)
    assert target.host == "openreview.net"

    with pytest.raises(DownloadPolicyError, match="scheme"):
        validate_download_url("ftp://openreview.net/x.pdf", allowed, public)
    with pytest.raises(DownloadPolicyError, match="host"):
        validate_download_url("https://evil.example/x.pdf", allowed, public)
    with pytest.raises(DownloadPolicyError, match="address"):
        validate_download_url("https://arxiv.org/pdf/1.pdf", allowed, private)
    with pytest.raises(DownloadPolicyError, match="address"):
        validate_download_url("https://arxiv.org/pdf/1.pdf", allowed, mixed)
    with pytest.raises(DownloadPolicyError, match="address"):
        validate_download_url("http://169.254.169.254/latest", {"169.254.169.254"}, public)
    with pytest.raises(DownloadPolicyError, match="credentials"):
        validate_download_url("https://user:pw@openreview.net/x.pdf", allowed, public)
