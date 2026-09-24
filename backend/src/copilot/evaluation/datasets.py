"""Frozen evaluation splits, and the validation that keeps them honest.

Three files make up a retrieval dataset, one record per line:

``queries.jsonl``
    the query, its provenance, the workflow it exercises, and whether our
    corpus holds every gold paper for it.
``qrels.jsonl``
    one row per gold paper, carrying both the benchmark's corpus id and — where
    matching found one — our paper id, so the same judgements serve a run
    against LitSearch's corpus and a run against ours.
``splits.jsonl``
    query to family to split.

Two rules are enforced rather than documented. A family never spans two splits,
because a paraphrase in development and its twin in test leaks the answer. And
a query counts as in-domain only when **every** gold paper is present: recall
computed against two of three golds understates the system silently, which is
the failure mode this whole module exists to prevent.

Splits are stratified by ``query_set`` and nothing else. LitSearch's
``specificity`` and ``quality`` annotations have no documented scale in the
published artifacts, so they travel as metadata for slicing results rather than
as something to balance on (see ``reports/e1-litsearch-coverage.md``).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .litsearch import LITSEARCH_REVISION, LitSearchQuery

SPLITS = ("development", "validation", "test")
SPLIT_SHARES = {"development": 0.6, "validation": 0.2, "test": 0.2}
DEFAULT_SEED = 42

# LitSearch builds its query sets two ways, and the two exercise different
# behaviour. An author describing their own paper is a known-item search; a
# question generated from a citation sentence is a related-work search. Neither
# is the "keeping up" workflow spec §11 also names — that gap is what the
# hand-authored in-domain queries in E4 exist to fill.
WORKFLOWS = {
    "manual_acl": "known_item",
    "manual_iclr": "known_item",
    "inline_acl": "related_work",
    "inline_nonacl": "related_work",
}

LITSEARCH_AUTHOR = "Ajith et al. 2024 (LitSearch)"


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


def workflow_for(query_set: str) -> str:
    """The workflow a query set exercises, from how it was constructed."""

    try:
        return WORKFLOWS[query_set]
    except KeyError as error:
        raise DatasetError("unknown_query_set", query_set) from error


def assign_splits(
    queries: Sequence[LitSearchQuery], *, seed: int = DEFAULT_SEED
) -> dict[str, str]:
    """Stratify by query set, then cut 60/20/20 inside each stratum.

    Sorted before shuffling so the assignment depends on the seed and the query
    ids, never on the order rows happened to arrive in. Stratifying inside each
    set matters because the sets are wildly different sizes and one of them
    (``inline_acl``) has no corpus coverage at all.
    """

    strata: defaultdict[str, list[str]] = defaultdict(list)
    for query in queries:
        workflow_for(query.query_set)  # reject an unknown set early
        strata[query.query_set].append(query.query_id)

    assignment: dict[str, str] = {}
    for query_set in sorted(strata):
        ids = sorted(strata[query_set])
        random.Random(f"{seed}:{query_set}").shuffle(ids)
        total = len(ids)
        development = round(total * SPLIT_SHARES["development"])
        validation = round(total * SPLIT_SHARES["validation"])
        for index, query_id in enumerate(ids):
            if index < development:
                assignment[query_id] = "development"
            elif index < development + validation:
                assignment[query_id] = "validation"
            else:
                assignment[query_id] = "test"
    return assignment


def build_dataset(
    queries: Sequence[LitSearchQuery],
    *,
    matched: Mapping[str, str],
    seed: int = DEFAULT_SEED,
    revision: str = LITSEARCH_REVISION,
) -> Dataset:
    """Assemble the three record sets from the benchmark and the match report.

    ``matched`` maps a LitSearch corpus id to our paper id. A query is
    in-domain only when every one of its gold papers appears there.
    """

    splits = assign_splits(queries, seed=seed)
    query_records: list[QueryRecord] = []
    qrel_records: list[QrelRecord] = []
    split_records: list[SplitRecord] = []
    for query in queries:
        golds = query.corpusids
        in_domain = bool(golds) and all(gold in matched for gold in golds)
        query_records.append(
            QueryRecord(
                query_id=query.query_id,
                query=query.query,
                query_set=query.query_set,
                workflow=workflow_for(query.query_set),
                specificity=query.specificity,
                quality=query.quality,
                in_domain=in_domain,
                source=f"litsearch@{revision}",
                author=LITSEARCH_AUTHOR,
            )
        )
        for gold in golds:
            qrel_records.append(
                QrelRecord(
                    query_id=query.query_id,
                    corpusid=gold,
                    # LitSearch labels are binary: the cited paper is relevant
                    # and nothing else is labelled at all.
                    grade=1,
                    paper_id=matched.get(gold),
                )
            )
        split_records.append(
            SplitRecord(
                query_id=query.query_id,
                # LitSearch has no rewrite families. Inventing them would fake
                # a grouping the data does not have, so a query is its own.
                family_id=query.query_id,
                split=splits[query.query_id],
            )
        )
    return Dataset(
        queries=tuple(query_records),
        qrels=tuple(qrel_records),
        splits=tuple(split_records),
    )


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
    written that way on purpose: LitSearch declares no license, so its text
    stays under ``DATA_DIR`` while the split assignment — the thing that must be
    frozen, reviewable and diffable — lives in git. ``hydrate`` puts the text
    back at run time from the pinned parquet.
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
        if record.split not in SPLITS:
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
