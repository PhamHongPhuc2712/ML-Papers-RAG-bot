"""Frozen evaluation splits, and the validation that keeps them honest.

Three files make up a retrieval dataset, one record per line:

``queries.jsonl``
    the query, its provenance, the workflow it exercises, and whether our
    corpus holds every gold paper for it.
``qrels.jsonl``
    one row per gold paper, carrying the benchmark's own document id
    (``corpusid``) and our paper id.
``splits.jsonl``
    query to family to split.

Two rules are enforced rather than documented. A family never spans two splits,
because a paraphrase in development and its twin in test leaks the answer. And
a query counts as in-domain only when **every** gold paper is present: recall
computed against two of three golds understates the system silently.

Builders live with their benchmark: ``orb.py`` (Open RAG Bench) and
``section_queries.py`` (S1's passage questions over our corpus).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPLITS = ("development", "validation", "test")
# A benchmark read as a whole rather than tuned on: every query in one split that
# never produces a decision (Open RAG Bench's text slice). Its answering sample is a
# separate assignment beside the dataset, not a split of it.
RETRIEVAL_SPLIT = "retrieval"

class DatasetError(RuntimeError):
    """A dataset cannot be built, read or trusted."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True, slots=True)
class QueryRecord:
    query_id: str
    query: str
    query_set: str
    workflow: str
    specificity: int
    quality: int
    in_domain: bool
    source: str
    author: str


@dataclass(frozen=True, slots=True)
class QrelRecord:
    query_id: str
    corpusid: str
    grade: int
    paper_id: str | None


@dataclass(frozen=True, slots=True)
class SplitRecord:
    query_id: str
    family_id: str
    split: str


@dataclass(frozen=True, slots=True)
class Dataset:
    queries: tuple[QueryRecord, ...]
    qrels: tuple[QrelRecord, ...]
    splits: tuple[SplitRecord, ...]


def _write(path: Path, rows: Sequence[Any], *, drop: frozenset[str] = frozenset()) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {k: v for k, v in asdict(row).items() if k not in drop}
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetError("missing_file", str(path))
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise DatasetError("malformed_line", f"{path}:{number}") from error
    return rows


def write_dataset(dataset: Dataset, directory: Path, *, include_query_text: bool = True) -> None:
    """Write the three files.

    ``include_query_text=False`` omits the query strings. The repository copy is
    written that way on purpose: ORB's text is CC-BY-NC and S1's is derived from
    corpus papers, so it stays under ``DATA_DIR`` while the split assignment — the
    thing that must be frozen, reviewable and diffable — lives in git. ``hydrate``
    puts the text back at run time.
    """

    directory.mkdir(parents=True, exist_ok=True)
    drop = frozenset() if include_query_text else frozenset({"query"})
    _write(directory / "queries.jsonl", dataset.queries, drop=drop)
    _write(directory / "qrels.jsonl", dataset.qrels)
    _write(directory / "splits.jsonl", dataset.splits)


def read_dataset(directory: Path) -> Dataset:
    """Read a dataset. A record written without its query text loads with ``""``."""

    return Dataset(
        queries=tuple(
            QueryRecord(**{"query": "", **row}) for row in _read(directory / "queries.jsonl")
        ),
        qrels=tuple(QrelRecord(**row) for row in _read(directory / "qrels.jsonl")),
        splits=tuple(SplitRecord(**row) for row in _read(directory / "splits.jsonl")),
    )


def hydrate(dataset: Dataset, texts: Mapping[str, str]) -> Dataset:
    """Re-attach query text to a dataset read from the text-free repository copy."""

    missing = [record.query_id for record in dataset.queries if not texts.get(record.query_id)]
    if missing:
        raise DatasetError("missing_query_text", missing[0])
    return Dataset(
        queries=tuple(
            QueryRecord(**{**asdict(record), "query": texts[record.query_id]})
            for record in dataset.queries
        ),
        qrels=dataset.qrels,
        splits=dataset.splits,
    )


def validate_dataset(directory: Path) -> dict[str, Any]:
    """Refuse a dataset that would leak between splits or reference nothing.

    Returns the dataset's shape so a caller can print what it accepted rather
    than only that it accepted something.
    """

    dataset = read_dataset(directory)
    query_ids = {record.query_id for record in dataset.queries}
    if len(query_ids) != len(dataset.queries):
        raise DatasetError("duplicate_query_id")

    families: dict[str, str] = {}
    split_ids: set[str] = set()
    for record in dataset.splits:
        if record.split not in (*SPLITS, RETRIEVAL_SPLIT):
            raise DatasetError("unknown_split", record.split)
        if record.query_id not in query_ids:
            raise DatasetError("unknown_query", record.query_id)
        held = families.setdefault(record.family_id, record.split)
        if held != record.split:
            raise DatasetError("family_spans_splits", record.family_id)
        split_ids.add(record.query_id)

    judged: set[str] = set()
    for qrel in dataset.qrels:
        if qrel.query_id not in query_ids:
            raise DatasetError("unknown_query", qrel.query_id)
        if qrel.grade < 0:
            raise DatasetError("invalid_grade", f"{qrel.query_id}={qrel.grade}")
        judged.add(qrel.query_id)

    if missing := query_ids - judged:
        raise DatasetError("query_without_qrel", sorted(missing)[0])
    if unsplit := query_ids - split_ids:
        raise DatasetError("query_without_split", sorted(unsplit)[0])

    counts: defaultdict[str, int] = defaultdict(int)
    for record in dataset.splits:
        counts[record.split] += 1
    return {
        "queries": len(dataset.queries),
        "qrels": len(dataset.qrels),
        "in_domain": sum(1 for record in dataset.queries if record.in_domain),
        "splits": dict(sorted(counts.items())),
        "families": len(families),
    }
