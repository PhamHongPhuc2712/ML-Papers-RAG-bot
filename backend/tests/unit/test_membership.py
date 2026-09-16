"""Pure eligibility, decision mapping, backoff and download-policy tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.corpus.download import DownloadPolicyError, validate_download_url
from copilot.corpus.sources.base import (
    DECISIONS,
    TRACKS,
    classify_track,
    is_eligible,
    normalize_decision,
)
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
        "page_size": 2,
    }
    config.update(overrides)
    return config


def test_mirror_listing_without_a_label_is_ineligible(tmp_path):
    """No label means no membership evidence, so the record is not admitted."""

    manifest = {"venue": "ICLR", "years": [2024], "track": "main"}
    path = tmp_path / "unlabelled.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "id00",
                "venue": "ICLR",
                "year": 2024,
                "title": "Paper number 0",
                "abstract": "Abstract.",
                "forum_url": "https://openreview.net/forum?id=forum00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = PapercliSource(_mirror_config(path), venue="ICLR", year=2024, track="main")
    rows, _ = source.fetch_page(None)
    assert [row["decision"] for row in rows] == ["unknown"]
    assert [row["track"] for row in rows] == ["unknown"]
    assert not any(is_eligible(row, manifest) for row in rows)


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


REGISTRY_TRACKS = (
    Path(__file__).resolve().parents[3] / "data" / "fixtures" / "tracks" / "registry-tracks.jsonl"
)

# Row-weighted classification of every label the registry carries for 2023-2026,
# measured 2026-09-16. These are the corpus sizes the manifest decisions imply.
EXPECTED_ROWS = {
    ("main", "accepted"): 85732,
    ("findings", "accepted"): 7499,
    ("workshop", "accepted"): 4883,
    ("main", "under_review"): 2219,
    ("other", "accepted"): 1118,
    ("industry", "accepted"): 706,
    ("demo", "accepted"): 414,
    ("shared_task", "accepted"): 157,
    ("tutorial", "accepted"): 61,
}


def _registry_labels() -> list[dict]:
    return [json.loads(line) for line in REGISTRY_TRACKS.read_text(encoding="utf-8").splitlines()]


def test_every_registry_label_classifies_to_a_known_track():
    """All 262 labels the registry uses, weighted by the rows carrying them."""

    labels = _registry_labels()
    assert len(labels) == 262
    totals: dict[tuple[str, str], int] = {}
    for row in labels:
        result = classify_track(row["label"])
        assert result[0] in TRACKS, row["label"]
        assert result[1] in DECISIONS, row["label"]
        totals[result] = totals.get(result, 0) + row["rows"]
    assert totals == EXPECTED_ROWS
    assert sum(totals.values()) == 102789


@pytest.mark.parametrize(
    "label",
    [
        "main",
        "ICLR 2024 poster",
        "ICLR 2024 oral",
        "NeurIPS 2025 spotlight",
        "ICML 2023 OralPoster",
        "ICLR 2023 notable top 5%",
        "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics"
        " (Volume 1: Long Papers)",
        "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    ],
)
def test_main_conference_labels_are_accepted_main_track(label):
    assert classify_track(label) == ("main", "accepted")


@pytest.mark.parametrize(
    ("label", "track"),
    [
        ("Findings of the Association for Computational Linguistics: ACL 2024", "findings"),
        ("Proceedings of the 8th Workshop on Representation Learning for NLP", "workshop"),
        (
            "Proceedings of the 63rd Annual Meeting of the Association for Computational"
            " Linguistics (Volume 6: Industry Track)",
            "industry",
        ),
        (
            "Proceedings of the 63rd Annual Meeting of the Association for Computational"
            " Linguistics (Volume 3: System Demonstrations)",
            "demo",
        ),
        ("Proceedings of The Third Arabic Natural Language Processing Conference: Shared Tasks",
         "shared_task"),
        # A whole co-located conference the registry files under venue EMNLP.
        ("Proceedings of the Ninth Conference on Machine Translation", "other"),
    ],
)
def test_non_main_tracks_are_separated_from_the_main_conference(label, track):
    assert classify_track(label) == (track, "accepted")


def test_submission_pool_is_not_accepted():
    """ICLR 2023 carries 2,219 unaccepted submissions under this one label."""

    assert classify_track("Submitted to ICLR 2023") == ("main", "under_review")


def test_an_unrecognized_label_is_never_main():
    """Main is matched positively, so a new venue label fails closed."""

    track, decision = classify_track("Proceedings of Some Future Colocated Thing 2027")
    assert track == "other"
    assert decision == "accepted"


def test_eligibility_follows_the_observed_label(tmp_path):
    """Membership is read off each record, not asserted for the whole listing."""

    manifest = {"venue": "ICLR", "years": [2024], "track": "main"}
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"id{index}",
                    "venue": "ICLR",
                    "year": 2024,
                    "track": label,
                    "title": f"Paper {index}",
                    "authors": ["Ada Lovelace"],
                    "abstract": "An abstract.",
                    "forum_url": f"https://openreview.net/forum?id=f{index}",
                }
            )
            for index, label in enumerate(
                ["ICLR 2024 poster", "Submitted to ICLR 2023", "ICLR 2024 Workshop on Things"]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    source = PapercliSource(
        {"records": str(path), "root_dir": str(tmp_path), "dataset_revision": "90a1fbd"},
        venue="ICLR",
        year=2024,
        track="main",
    )
    rows, _ = source.fetch_page(None)
    assert [row["decision"] for row in rows] == ["accepted", "under_review", "accepted"]
    assert [row["track"] for row in rows] == ["main", "main", "workshop"]
    assert [is_eligible(row, manifest) for row in rows] == [True, False, False]
