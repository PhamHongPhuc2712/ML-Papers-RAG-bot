"""Venue-by-venue corpus run: mirror, ingest, work, verify, then reclaim disk.

The whole corpus does not fit on this machine at once — roughly 450 GB of PDFs
against 522 GB free, with the database and index still to come — so a run takes
one venue-year at a time and deletes its PDFs once their text is safely in
PostgreSQL. Peak disk is then the largest single venue-year rather than the sum.

Deleting source material is the one irreversible step here, so it is fenced:
nothing is swept while any job for that venue-year is unfinished, a paper whose
parse did not succeed keeps its PDF because that copy is the only thing a retry
could read, and every path is checked to be inside the mirror before it is
unlinked.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class CampaignError(RuntimeError):
    """A venue-year cannot proceed safely."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class VenueYear:
    venue: str
    year: int

    @property
    def key(self) -> str:
        return f"{self.venue}:{self.year}"


def load_venue_plan(path: str | Path) -> tuple[list[VenueYear], dict[str, Any]]:
    """Read the ordered venue-year plan and its shared defaults."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise CampaignError("plan_invalid", "root")
    entries = raw.get("venue_years")
    if not isinstance(entries, list) or not entries:
        raise CampaignError("plan_invalid", "venue_years")
    plan: list[VenueYear] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or "venue" not in entry or "year" not in entry:
            raise CampaignError("plan_invalid", str(entry))
        plan.append(VenueYear(str(entry["venue"]), int(entry["year"])))
    defaults = raw.get("defaults")
    return plan, dict(defaults) if isinstance(defaults, Mapping) else {}


def _index_records(index: str | Path) -> list[dict[str, Any]]:
    path = Path(index)
    if not path.is_file():
        raise CampaignError("index_missing", str(path))
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def sweep_pdfs(
    index: str | Path,
    parsed_sha256: set[str],
    pdf_root: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Delete the PDFs whose text is stored, keeping every other copy.

    Paths are all validated before anything is unlinked, so a record pointing
    outside the mirror stops the sweep rather than taking part of it with it.
    """

    root = Path(pdf_root).resolve()
    records = _index_records(index)
    doomed: list[Path] = []
    kept = 0
    missing = 0
    for record in records:
        raw = record.get("pdf_path")
        if not raw or str(record.get("sha256") or "") not in parsed_sha256:
            kept += 1
            continue
        path = Path(str(raw)).resolve()
        if not path.is_relative_to(root):
            raise CampaignError("pdf_outside_mirror", str(path))
        if not path.is_file():
            missing += 1
            continue
        doomed.append(path)

    freed = 0
    for path in doomed:
        freed += path.stat().st_size
        if not dry_run:
            path.unlink()
    return {"deleted": len(doomed), "kept": kept, "missing": missing, "bytes_freed": freed}


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"completed": [], "results": {}}
    state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault("completed", [])
    state.setdefault("results", {})
    return state


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_campaign(
    plan: list[VenueYear],
    *,
    mirror: Callable[[VenueYear], dict[str, Any]],
    ingest: Callable[[VenueYear], str],
    work: Callable[[VenueYear, str], dict[str, Any]],
    sweep: Callable[[VenueYear, set[str]], dict[str, int]],
    state_path: str | Path,
    keep_pdfs: bool = False,
    stop_on_error: bool = True,
) -> dict[str, Any]:
    """Drive the plan, recording each venue-year so a restart resumes."""

    path = Path(state_path)
    state = _load_state(path)
    completed: list[str] = list(state["completed"])
    failed: list[str] = []
    for venue_year in plan:
        if venue_year.key in completed:
            continue
        outcome: dict[str, Any] = {"mirror": mirror(venue_year)}
        run_id = ingest(venue_year)
        outcome["run_id"] = run_id
        worked = work(venue_year, run_id)
        outcome["work"] = {k: v for k, v in worked.items() if k != "parsed_sha256"}
        if int(worked.get("unfinished", 0)) > 0:
            # Unfinished work means a retry may still need the PDFs.
            outcome["swept"] = None
            failed.append(venue_year.key)
            state["results"][venue_year.key] = outcome
            _save_state(path, state)
            if stop_on_error:
                break
            continue
        if not keep_pdfs:
            outcome["swept"] = sweep(venue_year, set(worked.get("parsed_sha256") or ()))
        completed.append(venue_year.key)
        state["completed"] = completed
        state["results"][venue_year.key] = outcome
        _save_state(path, state)
    return {"completed": completed, "failed": failed, "results": state["results"]}


# --- concrete steps -------------------------------------------------------
# Kept below the pure driver above so the orchestration stays testable without
# a database, a network or a subprocess.


def venue_manifest(
    base: Mapping[str, Any],
    *,
    venue: str,
    year: int,
    index: Path,
    mirror_root: Path,
    pdf_revision: str,
) -> dict[str, Any]:
    """The base manifest retargeted at one venue-year's captured index."""

    from .mirror import shard_repo

    manifest = dict(base)
    manifest["venue"] = venue
    manifest["years"] = [year]
    sources = {
        name: dict(config)
        for name, config in dict(base.get("sources", {})).items()
        if isinstance(config, Mapping)
    }
    papercli = sources.get("papercli")
    if papercli is None:
        raise CampaignError("plan_invalid", "base manifest has no papercli source")
    papercli.update(
        {
            "enabled": True,
            "root_dir": str(mirror_root),
            "records": str(index),
            "pdf_dataset": shard_repo(venue),
            "pdf_dataset_revision": pdf_revision,
        }
    )
    manifest["sources"] = sources
    manifest.pop("sample", None)  # a venue-year run takes the whole listing
    return manifest


@dataclass
class CampaignPaths:
    data_dir: Path
    run_dir: Path

    @property
    def mirror_root(self) -> Path:
        return self.data_dir / "sources" / "papercli"

    @property
    def pdf_root(self) -> Path:
        return self.mirror_root / "pdfs"

    @property
    def registry(self) -> Path:
        return self.mirror_root / "papers.parquet"

    def index(self, venue: str, year: int) -> Path:
        return self.mirror_root / f"{venue.lower()}-{year}.jsonl"

    def manifest(self, venue: str, year: int) -> Path:
        return self.run_dir / f"{venue.lower()}-{year}.yaml"


def pending_jobs(engine: Any, run_id: str) -> tuple[int, int]:
    """(unfinished, terminally failed) jobs for one run."""

    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "select status, count(*) from jobs where payload->>'run_id' = :run_id "
                "group by status"
            ),
            {"run_id": run_id},
        ).all()
    counts = {status: int(count) for status, count in rows}
    unfinished = sum(counts.get(status, 0) for status in ("queued", "running", "retry_wait"))
    return unfinished, counts.get("failed", 0)


def parsed_checksums(engine: Any) -> set[str]:
    """Checksums whose text is stored, so their PDF is no longer the only copy."""

    from sqlalchemy import text

    with engine.connect() as connection:
        rows = connection.execute(
            text("select content_sha256 from paper_versions where parse_status = 'parsed'")
        ).scalars()
        return {str(value) for value in rows}
