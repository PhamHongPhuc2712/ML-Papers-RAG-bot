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
    """The enriched venue-year index `corpus mirror-index` writes."""

    path = tmp_path / "records.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "id": f"id{index:02d}",
                        "venue": "ICLR",
                        "year": 2024,
                        "track": "ICLR 2024 poster",
                        "source": "iclr",
                        "title": f"Paper number {index}",
                        "authors": ["Ada Lovelace", "Grace Hopper"],
                        "abstract": f"Abstract for paper {index}.",
                        "forum_url": f"https://openreview.net/forum?id=forum{index:02d}",
                        "pdf_url": f"https://openreview.net/pdf?id=forum{index:02d}",
                        "hf_pdf_path": f"pdfs/iclr/2024/{index:02d}/id{index:02d}.pdf",
                        "pdf_path": str(tmp_path / "pdfs" / f"{index:02d}.pdf"),
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
    # The registry id is the mirror's own key; the forum id comes from the URL
    # and is the identifier the upstream venue actually published.
    assert row["external_ids"] == {"papercli": "id00", "openreview": "forum00"}
    assert row["title"] == "Paper number 0"
    assert row["abstract"] == "Abstract for paper 0."
    assert row["venue"] == {"name": "ICLR", "track": "main"}
    assert row["year"] == 2024
    assert row["source_url"] == "https://openreview.net/forum?id=forum00"
    assert row["pdf_path"] == str(tmp_path / "pdfs" / "00.pdf")
    assert row["pdf_sha256"] == "ab" * 32
    # The registry publishes authors; the five-column browse view did not.
    assert row["authors"] == ["Ada Lovelace", "Grace Hopper"]
    # The venue's own label is carried through for the decision mapper.
    assert row["source_track"] == "ICLR 2024 poster"


def test_proceedings_rows_carry_no_openreview_alias(tmp_path):
    path = tmp_path / "cvpr.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "abc123",
                "venue": "CVPR",
                "year": 2024,
                "track": "main",
                "title": "A vision paper",
                "authors": ["Ada Lovelace"],
                "abstract": "An abstract.",
                "forum_url": "https://openaccess.thecvf.com/content/CVPR2024/html/x.html",
                "pdf_url": "https://openaccess.thecvf.com/content/CVPR2024/papers/x.pdf",
                "pdf_path": None,
                "sha256": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = PapercliSource(_mirror_config(path), venue="CVPR", year=2024, track="main")
    (row,), _ = source.fetch_page(None)
    assert row["external_ids"] == {"papercli": "abc123"}
    assert row["pdf_path"] is None
    assert row["pdf_url"].endswith(".pdf")


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
