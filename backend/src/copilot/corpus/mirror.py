"""Build a venue-year index from the pinned paper registry.

The papercli dataset publishes two artifacts: a registry parquet holding every
venue-year's metadata, and per-venue PDF shards. The registry is the record of
what exists; this module projects one venue-year out of it and pairs each row
with the mirrored PDF already on disk, checksumming what it finds.

The resulting JSONL is what the ingestion adapter reads, so the parquet reader
stays out of the ingest path and every downstream record carries a checksum
this module actually computed rather than one a remote index claimed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .sources.base import classify_track

# The registry columns this project depends on. The per-venue `browse/` views
# publish only five of them, which is why the root registry is the source.
REGISTRY_COLUMNS = (
    "id",
    "title",
    "authors",
    "abstract",
    "venue",
    "year",
    "track",
    "source",
    "pdf_url",
    "forum_url",
    "hf_pdf_path",
)
_READ_BLOCK = 1 << 20
# Each venue's PDFs live in their own shard repository (dataset card, §Hub).
_SHARD_PREFIX = "GenAI4ELab/papercli-papers-"
# Refuse to start a venue-year that could fill the disk mid-run. The largest
# single venue-year measured is ICLR 2026 at roughly 48 GB.
DEFAULT_MIN_FREE_BYTES = 80 * 1024**3
DEFAULT_WORKERS = 8

# (repo, revision, member path, destination root) -> local path
Fetch = Callable[[str, str, str, Path], Path]


class MirrorIndexError(RuntimeError):
    """The registry cannot produce a usable index for this venue-year."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def split_authors(value: object) -> list[str]:
    """The registry stores authors as one semicolon-separated string."""

    if not value:
        return []
    return [name.strip() for name in str(value).split(";") if name.strip()]


def checksum(path: Path) -> tuple[str, int]:
    """Stream a local file to a sha256 and byte count."""

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def read_registry(registry: str | Path, *, venue: str, year: int) -> list[dict[str, Any]]:
    """Read one venue-year out of the registry parquet, sorted for replayability."""

    try:
        table = pq.read_table(
            registry,
            columns=list(REGISTRY_COLUMNS),
            filters=[("venue", "=", venue), ("year", "=", year)],
        )
    except (KeyError, ValueError) as error:  # missing column or unreadable file
        raise MirrorIndexError("registry_unreadable", str(error)) from error
    rows: list[dict[str, Any]] = table.to_pylist()
    if not rows:
        raise MirrorIndexError("no_rows", f"{venue}:{year}")
    rows.sort(key=lambda row: str(row.get("id")))
    return rows


def eligible_rows(rows: list[dict[str, Any]], track: str | None) -> list[dict[str, Any]]:
    """Keep only the rows a manifest would admit, by the venue's own label.

    Mirroring everything a venue-year lists would fetch 17,057 papers across the
    corpus that ingestion then rejects — the ACL family's Findings, workshop,
    industry and demo volumes, and ICLR 2023's unaccepted submissions, roughly
    53 GB — so the filter belongs before the download, not after it.
    """

    if track is None:
        return rows
    return [row for row in rows if classify_track(row.get("track")) == (track, "accepted")]


def build_index(
    registry: str | Path,
    *,
    venue: str,
    year: int,
    pdf_root: str | Path,
    out: str | Path,
    mirror_revision: str = "",
    track: str | None = "main",
) -> dict[str, int]:
    """Write the venue-year index and report how many rows have a local PDF.

    A row whose PDF is absent keeps a null path and is counted, never invented:
    a missing full text leaves a searchable abstract (spec §4).
    """

    rows = eligible_rows(read_registry(registry, venue=venue, year=year), track)
    root = Path(pdf_root)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_suffix(target.suffix + ".partial")
    counts = {"rows": len(rows), "with_pdf": 0, "missing_pdf": 0}
    with staged.open("w", encoding="utf-8") as handle:
        for row in rows:
            relative = row.get("hf_pdf_path")
            local: Path | None = None
            digest: str | None = None
            size: int | None = None
            if relative:
                candidate = root / str(relative)
                if candidate.is_file():
                    digest, size = checksum(candidate)
                    local = candidate
            counts["with_pdf" if local is not None else "missing_pdf"] += 1
            record = {
                **{column: row.get(column) for column in REGISTRY_COLUMNS},
                "authors": split_authors(row.get("authors")),
                "pdf_path": str(local) if local is not None else None,
                "sha256": digest,
                "bytes": size,
                "mirror_revision": mirror_revision,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    staged.replace(target)
    return counts


def shard_repo(venue: str) -> str:
    """The dataset repository holding one venue's PDF shard."""

    return f"{_SHARD_PREFIX}{venue.strip().lower()}"


def safe_member_path(member: object) -> Path:
    """Validate a repository-relative path before it is joined to a local root.

    ``hf_pdf_path`` is data, so it is checked rather than trusted: an absolute
    path, a drive letter or any parent traversal would let a registry row write
    outside the mirror.
    """

    text = str(member or "").strip().replace("\\", "/")
    if not text:
        raise MirrorIndexError("unsafe_member_path", "empty")
    path = Path(text)
    if path.is_absolute() or (len(text) > 1 and text[1] == ":"):
        raise MirrorIndexError("unsafe_member_path", text)
    if any(part in ("..", "") for part in path.parts):
        raise MirrorIndexError("unsafe_member_path", text)
    return path


def hf_fetch(repo: str, revision: str, member: str, root: Path) -> Path:
    """Download one shard member at a pinned revision into the mirror.

    This is the offline capture step, never a request-time dependency: nothing
    in the serving path calls it (spec §2).
    """

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=member,
            revision=revision,
            repo_type="dataset",
            local_dir=str(root),
        )
    )


def _hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


_RETRY_SLEEP = time.sleep
REVISION_ATTEMPTS = 5


def resolve_revision(repo: str) -> str:
    """The shard's current commit, so a run records the revision it captured.

    Retried, because this one metadata call gates a whole venue-year. A 503
    from the hub ended a 42-venue-year campaign at venue 31 on 2026-09-21: the
    lookup is momentary and the run behind it is a day long, so a transient
    failure here must never be the thing that stops it.
    """

    last = ""
    for attempt in range(REVISION_ATTEMPTS):
        try:
            sha = _hf_api().dataset_info(repo).sha
        except Exception as error:  # noqa: BLE001 - any hub failure is retryable here
            last = f"{type(error).__name__}: {error}"
        else:
            if sha:
                return str(sha)
            last = "empty_sha"
        if attempt < REVISION_ATTEMPTS - 1:
            _RETRY_SLEEP(2**attempt)
    raise MirrorIndexError("revision_unresolved", f"{repo} ({last})")


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def mirror_venue_year(
    registry: str | Path,
    *,
    venue: str,
    year: int,
    mirror_root: str | Path,
    revision: str,
    fetch: Fetch | None = None,
    workers: int = DEFAULT_WORKERS,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    free_bytes: Callable[[Path], int] = _free_bytes,
    out: str | Path | None = None,
    track: str | None = "main",
) -> dict[str, Any]:
    """Download one venue-year's PDFs, then write its verified index.

    Resumable by construction: a member already on disk is skipped, so an
    interrupted run continues where it stopped. Failures are counted and
    reported rather than swallowed — a paper whose PDF never arrives keeps a
    null path in the index and stays searchable by abstract.
    """

    rows = eligible_rows(read_registry(registry, venue=venue, year=year), track)
    root = Path(mirror_root)
    pdf_root = root / "pdfs"
    pdf_root.mkdir(parents=True, exist_ok=True)
    available = free_bytes(pdf_root)
    if available < min_free_bytes:
        raise MirrorIndexError("insufficient_free_space", f"{available} < {min_free_bytes}")

    repo = shard_repo(venue)
    download = fetch or hf_fetch
    pending: list[str] = []
    already = 0
    for row in rows:
        member = row.get("hf_pdf_path")
        if not member:
            continue
        relative = safe_member_path(member)
        if (pdf_root / relative).is_file():
            already += 1
            continue
        pending.append(relative.as_posix())

    downloaded = 0
    failures: list[tuple[str, str]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(download, repo, revision, member, pdf_root): member
                for member in pending
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - every failure is reported
                    failures.append((futures[future], str(error)))
                else:
                    downloaded += 1

    index = Path(out) if out else root / f"{venue.lower()}-{year}.jsonl"
    counts = build_index(
        registry,
        venue=venue,
        year=year,
        pdf_root=pdf_root,
        out=index,
        mirror_revision=revision,
        track=track,
    )
    return {
        "venue": venue,
        "year": year,
        "repo": repo,
        "revision": revision,
        "index": str(index),
        "downloaded": downloaded,
        "already_present": already,
        "failed": len(failures),
        "failures": [f"{member}: {reason}" for member, reason in failures[:10]],
        **counts,
    }
