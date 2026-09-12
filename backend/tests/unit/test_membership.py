"""Pure eligibility, decision mapping, backoff and download-policy tests."""

from __future__ import annotations

from pathlib import Path

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


def test_download_pacing_is_configurable_and_off_by_default():
    """Spec 4 requires provider throttling to be configurable.

    Three concurrent workers with no inter-request delay had OpenReview return a
    long Retry-After for 8,665 of 8,905 downloads. The pacing below is what keeps
    a fleet under a provider's limit instead of discovering it.
    """

    from copilot.corpus.ingest import IngestContext

    def context(**download):
        return IngestContext(
            manifest={"venue": "ICLR", "years": [2024], "track": "main", "download": download},
            transport=None,
            staging_dir=Path("."),
            parsing_config=None,
        )

    assert context().min_interval_seconds == 0.0
    assert context(min_interval_seconds=1.5).min_interval_seconds == 1.5


def test_pacing_sleeps_only_for_the_remaining_interval(monkeypatch):
    from copilot.corpus import ingest as ingest_module
    from copilot.corpus.ingest import IngestContext

    slept: list[float] = []
    # Two reads per paced call (one to measure, one to re-mark), one for the first.
    clock = iter([100.0, 100.2, 100.2, 105.0, 105.0])
    monkeypatch.setattr(ingest_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(ingest_module.time, "sleep", slept.append)

    context = IngestContext(
        manifest={"download": {"min_interval_seconds": 1.0}},
        transport=None,
        staging_dir=Path("."),
        parsing_config=None,
    )
    context.pace()          # first call sets the mark, never sleeps
    context.pace()          # 0.2 s elapsed of a 1.0 s interval
    context.pace()          # 4.8 s elapsed, already past the interval

    assert slept == [pytest.approx(0.8)]
