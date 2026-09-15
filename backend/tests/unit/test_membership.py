"""Pure eligibility, decision mapping, backoff and download-policy tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.corpus.download import DownloadPolicyError, validate_download_url
from copilot.corpus.sources.base import is_eligible, normalize_decision
from copilot.corpus.sources.papercli import PapercliSource
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


def _mirror_records(tmp_path, count: int = 3) -> Path:
    path = tmp_path / "records.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "forum_id": f"id{index:02d}",
                        "venue": "ICLR",
                        "year": 2024,
                        "title": f"Paper number {index}",
                        "abstract": f"Abstract for paper {index}.",
                        "forum_url": f"https://openreview.net/forum?id=id{index:02d}",
                        "openreview_pdf_url": (
                            f"https://api2.openreview.net/attachment?name=pdf&id=id{index:02d}"
                        ),
                        "pdf_path": f"pdfs/{index:02d}.pdf",
                        "bytes": 1024,
                        "sha256": "ab" * 32,
                    }
                )
                + "\n"
            )
    return path


def _mirror_config(records: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "records": str(records),
        "root_dir": str(records.parent),
        "dataset": "GenAI4ELab/papercli-papers",
        "dataset_revision": "90a1fbd",
        "membership_is_acceptance": True,
        "page_size": 2,
    }
    config.update(overrides)
    return config


def test_mirror_listing_is_ineligible_until_acceptance_is_declared(tmp_path):
    """A mirror index carries no decision field, so membership must be asserted."""

    manifest = {"venue": "ICLR", "years": [2024], "track": "main"}
    records = _mirror_records(tmp_path)

    undeclared = PapercliSource(
        _mirror_config(records, membership_is_acceptance=False),
        venue="ICLR",
        year=2024,
        track="main",
    )
    rows, _ = undeclared.fetch_page(None)
    assert rows and all(row["decision"] == "unknown" for row in rows)
    assert not any(is_eligible(row, manifest) for row in rows)

    declared = PapercliSource(_mirror_config(records), venue="ICLR", year=2024, track="main")
    rows, _ = declared.fetch_page(None)
    assert all(row["decision"] == "accepted" for row in rows)
    assert all(is_eligible(row, manifest) for row in rows)


def test_mirror_records_carry_identity_and_local_pdf(tmp_path):
    records = _mirror_records(tmp_path, count=1)
    source = PapercliSource(_mirror_config(records), venue="ICLR", year=2024, track="main")
    (row,), _ = source.fetch_page(None)

    assert row["source"] == "papercli"
    assert row["source_item_id"] == "id00"
    assert row["source_revision"] == "90a1fbd"
    assert row["external_ids"] == {"openreview": "id00"}
    assert row["title"] == "Paper number 0"
    assert row["abstract"] == "Abstract for paper 0."
    assert row["venue"] == {"name": "ICLR", "track": "main"}
    assert row["year"] == 2024
    assert row["source_url"] == "https://openreview.net/forum?id=id00"
    assert row["pdf_path"] == str(tmp_path / "pdfs" / "00.pdf")
    assert row["pdf_sha256"] == "ab" * 32
    assert row["authors"] == []


def test_mirror_pages_every_record_without_repeating(tmp_path):
    records = _mirror_records(tmp_path, count=5)
    source = PapercliSource(_mirror_config(records), venue="ICLR", year=2024, track="main")
    seen: list[str] = []
    cursor = None
    while True:
        rows, cursor = source.fetch_page(cursor)
        seen.extend(row["source_item_id"] for row in rows)
        if cursor is None:
            break
    assert seen == [f"id{index:02d}" for index in range(5)]


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
