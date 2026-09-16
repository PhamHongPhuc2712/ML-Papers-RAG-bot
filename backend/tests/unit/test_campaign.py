"""Venue-by-venue driver: ordering, resumption, and the destructive PDF sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.corpus.venues import (
    CampaignError,
    VenueYear,
    default_state_path,
    load_venue_plan,
    run_campaign,
    sweep_pdfs,
)


def _index(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "iclr-2024.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _mirror(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    pdf_root = tmp_path / "pdfs"
    parsed = pdf_root / "pdfs" / "iclr" / "2024" / "aa" / "parsed.pdf"
    failed = pdf_root / "pdfs" / "iclr" / "2024" / "bb" / "failed.pdf"
    for path in (parsed, failed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 x")
    index = _index(
        tmp_path,
        [
            {"id": "a", "sha256": "aa" * 32, "pdf_path": str(parsed)},
            {"id": "b", "sha256": "bb" * 32, "pdf_path": str(failed)},
        ],
    )
    return index, pdf_root, parsed, failed


def test_sweep_deletes_parsed_pdfs_and_keeps_the_rest(tmp_path):
    """A PDF whose paper never parsed is the only copy left to retry from."""

    index, pdf_root, parsed, failed = _mirror(tmp_path)
    result = sweep_pdfs(index, {"aa" * 32}, pdf_root)
    assert result == {"deleted": 1, "kept": 1, "missing": 0, "bytes_freed": 10}
    assert not parsed.exists()
    assert failed.exists()


def test_sweep_dry_run_reports_without_deleting(tmp_path):
    index, pdf_root, parsed, failed = _mirror(tmp_path)
    result = sweep_pdfs(index, {"aa" * 32, "bb" * 32}, pdf_root, dry_run=True)
    assert result["deleted"] == 2
    assert parsed.exists() and failed.exists()


def test_sweep_refuses_paths_outside_the_mirror(tmp_path):
    outside = tmp_path / "elsewhere" / "secret.pdf"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"%PDF-1.4 x")
    index = _index(tmp_path, [{"id": "a", "sha256": "aa" * 32, "pdf_path": str(outside)}])
    with pytest.raises(CampaignError, match="pdf_outside_mirror"):
        sweep_pdfs(index, {"aa" * 32}, tmp_path / "pdfs")
    assert outside.exists()


def test_plan_is_read_in_configured_order(tmp_path):
    config = tmp_path / "venues.yaml"
    config.write_text(
        "defaults:\n  workers: 9\n  min_free_gb: 80\n"
        "venue_years:\n  - {venue: WACV, year: 2023}\n  - {venue: ICLR, year: 2024}\n",
        encoding="utf-8",
    )
    plan, defaults = load_venue_plan(config)
    assert plan == [VenueYear("WACV", 2023), VenueYear("ICLR", 2024)]
    assert defaults["workers"] == 9


class _Steps:
    """Records the order the driver calls each stage in."""

    def __init__(self, failing: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.failing = failing or set()

    def mirror(self, vy: VenueYear) -> dict[str, object]:
        self.calls.append(f"mirror:{vy.venue}:{vy.year}")
        return {"rows": 2, "with_pdf": 2, "failed": 0, "index": "x.jsonl"}

    def ingest(self, vy: VenueYear) -> str:
        self.calls.append(f"ingest:{vy.venue}:{vy.year}")
        return f"run-{vy.venue}-{vy.year}"

    def work(self, vy: VenueYear, run_id: str) -> dict[str, object]:
        self.calls.append(f"work:{vy.venue}:{vy.year}")
        unfinished = 1 if f"{vy.venue}:{vy.year}" in self.failing else 0
        return {"processed": 2, "unfinished": unfinished, "parsed_sha256": set()}

    def sweep(self, vy: VenueYear, shas: set[str]) -> dict[str, int]:
        self.calls.append(f"sweep:{vy.venue}:{vy.year}")
        return {"deleted": 2, "kept": 0, "missing": 0, "bytes_freed": 20}


def test_campaign_runs_each_stage_in_order_and_records_state(tmp_path):
    steps = _Steps()
    state = tmp_path / "state.json"
    result = run_campaign(
        [VenueYear("WACV", 2023), VenueYear("ICLR", 2024)],
        mirror=steps.mirror,
        ingest=steps.ingest,
        work=steps.work,
        sweep=steps.sweep,
        state_path=state,
    )
    assert steps.calls == [
        "mirror:WACV:2023", "ingest:WACV:2023", "work:WACV:2023", "sweep:WACV:2023",
        "mirror:ICLR:2024", "ingest:ICLR:2024", "work:ICLR:2024", "sweep:ICLR:2024",
    ]
    assert result["completed"] == ["WACV:2023", "ICLR:2024"]
    assert json.loads(state.read_text())["completed"] == ["WACV:2023", "ICLR:2024"]


def test_campaign_resumes_and_skips_completed_venue_years(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"completed": ["WACV:2023"], "results": {}}), encoding="utf-8")
    steps = _Steps()
    run_campaign(
        [VenueYear("WACV", 2023), VenueYear("ICLR", 2024)],
        mirror=steps.mirror,
        ingest=steps.ingest,
        work=steps.work,
        sweep=steps.sweep,
        state_path=state,
    )
    assert not any(call.endswith("WACV:2023") for call in steps.calls)
    assert steps.calls[0] == "mirror:ICLR:2024"


def test_unfinished_jobs_stop_the_sweep_and_the_campaign(tmp_path):
    """PDFs are the only retry source, so nothing is deleted while work is pending."""

    steps = _Steps(failing={"WACV:2023"})
    result = run_campaign(
        [VenueYear("WACV", 2023), VenueYear("ICLR", 2024)],
        mirror=steps.mirror,
        ingest=steps.ingest,
        work=steps.work,
        sweep=steps.sweep,
        state_path=tmp_path / "state.json",
        stop_on_error=True,
    )
    assert "sweep:WACV:2023" not in steps.calls
    assert result["completed"] == []
    assert result["failed"] == ["WACV:2023"]
    assert not any("ICLR" in call for call in steps.calls)


def test_state_path_is_scoped_to_the_plan_not_the_invocation(tmp_path):
    """Two runs of the same plan must share progress, or a restart redoes it all."""

    first = default_state_path(tmp_path, Path("configs/venues.yaml"))
    second = default_state_path(tmp_path, Path("configs/venues.yaml"))
    assert first == second == tmp_path / "runs" / "venues-state.json"
    assert default_state_path(tmp_path, Path("configs/pilot.yaml")) != first
