"""Where retrieval loses the gold paper: offline analysis of recorded runs (P2.6).

Everything here reads a run's ``per_query.parquet`` and its manifest, and nothing
else — no service, no model, no GPU — so it can be rerun on any recorded run and
regenerated after every new one. Two questions are asked of it:

* **Where is the gold paper lost?** Every gold is either returned within the depth,
  in the candidate pool (the union of each branch's top ``per_branch``) but cut, or
  never in the pool. The three shares add to one.
* **Which queries lost it?** Query ids per slice; never query text.

The test split is refused outright: it is read once, for a release decision, and
diagnosing it would turn it into tuning data.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import read_dataset
from .metrics import recall_at_k
from .retrieval import dataset_digest

# The retrieval split is a benchmark read as a whole (Open RAG Bench): diagnosed like
# development, never chosen on.
DIAGNOSTIC_SPLITS = ("development", "validation", "retrieval")
RANK_BUCKETS: tuple[tuple[int, int], ...] = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100))
ABSENT = "absent"
# What every recorded run uses (spec §11); a manifest that names its own wins.
DEFAULT_BOOTSTRAP: Mapping[str, Any] = {"resamples": 1000, "seed": 42, "confidence": 0.95}


class GapError(ValueError):
    """A run that cannot be analysed, or a split that must not be."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def _bucket(low: int, high: int) -> str:
    return str(low) if low == high else f"{low}-{high}"


@dataclass(frozen=True)
class Outcome:
    """One variant's recorded result for one query, as far as this analysis needs it."""

    variant: str
    query_id: str
    query_set: str
    specificity: int | None
    ranked: tuple[str, ...]
    relevant: frozenset[str]
    candidate_recall: float
    failure: str | None

    def recall(self, k: int) -> float:
        return recall_at_k(self.ranked, self.relevant, k)

    def gold_ranks(self) -> list[int | None]:
        """Each gold's one-based rank in the deduplicated ranking, or None if absent."""

        position = {item: rank for rank, item in enumerate(dict.fromkeys(self.ranked), start=1)}
        return [position.get(gold) for gold in sorted(self.relevant)]


def load_outcomes(run: str | Path, split: str) -> tuple[dict[str, Any], list[Outcome]]:
    """A recorded split's manifest and outcomes, with specificity joined from the dataset.

    Runs recorded before specificity was a per-query column (P2.5) take it from the
    frozen dataset the manifest names — refused if that dataset has changed since.
    """

    if split not in DIAGNOSTIC_SPLITS:
        raise GapError("held_out_split" if split == "test" else "unknown_split", split)
    import pyarrow.parquet as pq

    directory = Path(run) / split
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text("utf-8"))
    dataset = Path(str(manifest["dataset"]["path"]))
    if dataset_digest(dataset) != manifest["dataset"]["digest"]:
        raise GapError("dataset_changed", str(dataset))
    specificity = {query.query_id: query.specificity for query in read_dataset(dataset).queries}
    outcomes = []
    for row in pq.read_table(directory / "per_query.parquet").to_pylist():
        if row.get("candidate_recall") is None:
            raise GapError("candidate_recall_missing", f"{row['variant']}:{row['query_id']}")
        recorded = row.get("specificity")
        outcomes.append(
            Outcome(
                variant=str(row["variant"]),
                query_id=str(row["query_id"]),
                query_set=str(row["query_set"]),
                specificity=int(recorded)
                if recorded is not None
                else specificity.get(str(row["query_id"])),
                ranked=tuple(str(item) for item in row["ranked"] or ()),
                relevant=frozenset(str(item) for item in row["relevant"]),
                candidate_recall=float(row["candidate_recall"]),
                failure=row.get("failure"),
            )
        )
    return manifest, outcomes


def _by_variant(outcomes: Iterable[Outcome]) -> dict[str, list[Outcome]]:
    grouped: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.variant, []).append(outcome)
    return grouped


def _facets(
    outcomes: Sequence[Outcome], summarize: Callable[[Sequence[Outcome]], Any]
) -> dict[str, Any]:
    """The summary over every query, then per query set and per specificity."""

    result: dict[str, Any] = {"all": summarize(outcomes)}
    for facet in ("query_set", "specificity"):
        groups: dict[str, list[Outcome]] = {}
        for outcome in outcomes:
            if facet == "query_set":
                value = outcome.query_set
            else:
                value = "unknown" if outcome.specificity is None else str(outcome.specificity)
            groups.setdefault(value, []).append(outcome)
        result[facet] = {value: summarize(subset) for value, subset in sorted(groups.items())}
    return result


def stage_losses(outcomes: Iterable[Outcome], *, depth: int) -> dict[str, dict[str, Any]]:
    """Per variant: the share of gold papers returned within ``depth``, cut, or never pooled.

    Means over queries, each query weighted once however many golds it has — the
    same weighting as recall. "Beyond depth" is what a deeper rerank could at most
    recover; "not in pool" is what no reranker can.
    """

    def summarize(rows: Sequence[Outcome]) -> dict[str, float]:
        within = [row.recall(depth) for row in rows]
        pooled = [row.candidate_recall for row in rows]
        return {
            "queries": len(rows),
            "within_depth": float(np.mean(within)),
            "beyond_depth": float(np.mean([p - w for p, w in zip(pooled, within, strict=True)])),
            "not_in_pool": float(np.mean([1 - p for p in pooled])),
        }

    return {variant: _facets(rows, summarize) for variant, rows in _by_variant(outcomes).items()}


def gold_rank_buckets(outcomes: Iterable[Outcome]) -> dict[str, dict[str, Any]]:
    """Per variant: how many gold papers sit in each rank band, or are absent."""

    def summarize(rows: Sequence[Outcome]) -> dict[str, int]:
        counts = {_bucket(low, high): 0 for low, high in RANK_BUCKETS}
        counts[ABSENT] = 0
        for row in rows:
            for rank in row.gold_ranks():
                if rank is None:
                    counts[ABSENT] += 1
                    continue
                for low, high in RANK_BUCKETS:
                    if low <= rank <= high:
                        counts[_bucket(low, high)] += 1
                        break
        # A band past every stored ranking is noise in a table; drop it when empty.
        return {band: count for band, count in counts.items() if count or band != "51-100"}

    return {variant: _facets(rows, summarize) for variant, rows in _by_variant(outcomes).items()}


def misses(outcomes: Iterable[Outcome], variant: str, *, depth: int) -> dict[str, dict[str, Any]]:
    """Query ids that lost a gold paper, per ``query_set / specificity`` slice.

    A query with several golds can appear under both headings: one gold never
    pooled, another pooled and cut.
    """

    slices: dict[str, dict[str, Any]] = {}
    for row in sorted(
        (row for row in outcomes if row.variant == variant), key=lambda row: row.query_id
    ):
        entry = slices.setdefault(
            f"{row.query_set} / {row.specificity}",
            {"queries": 0, "not_in_pool": [], "beyond_depth": []},
        )
        entry["queries"] += 1
        golds = len(row.relevant)
        pooled = round(row.candidate_recall * golds)
        within = round(row.recall(depth) * golds)
        if pooled < golds:
            entry["not_in_pool"].append(row.query_id)
        if pooled > within:
            entry["beyond_depth"].append(row.query_id)
    return dict(sorted(slices.items()))


# --- Rendering ----------------------------------------------------------------------------


def _percent(value: float) -> str:
    return f"{100 * value:.1f}"


def _loss_row(label: str, cell: Mapping[str, float]) -> str:
    return (
        f"| {label} | {int(cell['queries'])} | {_percent(cell['within_depth'])} | "
        f"{_percent(cell['beyond_depth'])} | {_percent(cell['not_in_pool'])} |"
    )


def _rank_row(label: str, counts: Mapping[str, int], bands: Sequence[str]) -> str:
    total = sum(counts.values())
    return f"| {label} | {total} | " + " | ".join(str(counts.get(b, 0)) for b in bands) + " |"


def _split_lines(
    split: str,
    manifest: Mapping[str, Any],
    outcomes: Sequence[Outcome],
    *,
    focus: str,
    depth: int,
) -> tuple[list[str], dict[str, Any]]:
    losses = stage_losses(outcomes, depth=depth)
    ranks = gold_rank_buckets(outcomes)
    missed = misses(outcomes, focus, depth=depth)
    per_branch = (manifest.get("search") or {}).get("values", {}).get("candidates_per_branch")
    queries = len({outcome.query_id for outcome in outcomes})

    lines = [f"## {split.capitalize()} — {queries} queries", ""]
    lines += [
        f"### Where the gold paper is lost (depth {depth})",
        "",
        f"Share of gold papers, averaged per query. The pool is the union of each branch's "
        f"top {per_branch}; for a single-branch mode it is that branch's list.",
        "",
        f"| Variant | Queries | In top {depth} | In pool, beyond {depth} | Not in pool |",
        "|---|---|---|---|---|",
    ]
    lines += [_loss_row(f"`{variant}`", facets["all"]) for variant, facets in losses.items()]
    if focus in losses:
        lines += [
            "",
            f"`{focus}` by slice:",
            "",
            f"| Slice | Queries | In top {depth} | In pool, beyond {depth} | Not in pool |",
            "|---|---|---|---|---|",
        ]
        for facet in ("query_set", "specificity"):
            lines += [
                _loss_row(f"{facet}: `{value}`", cell)
                for value, cell in losses[focus][facet].items()
            ]

    bands = list(next(iter(ranks.values()))["all"]) if ranks else []
    header = "| Variant | Golds | " + " | ".join(bands) + " |"
    lines += [
        "",
        "### Gold rank distribution",
        "",
        f"Counts of gold papers by one-based rank; `{ABSENT}` means not in the stored ranking.",
        "",
        header,
        "|---|---|" + "---|" * len(bands),
    ]
    lines += [_rank_row(f"`{variant}`", facets["all"], bands) for variant, facets in ranks.items()]
    if focus in ranks:
        lines += [
            "",
            f"`{focus}` by slice:",
            "",
            header.replace("Variant", "Slice"),
            "|---|---|" + "---|" * len(bands),
        ]
        for facet in ("query_set", "specificity"):
            lines += [
                _rank_row(f"{facet}: `{value}`", counts, bands)
                for value, counts in ranks[focus][facet].items()
            ]

    if missed:
        lines += [
            "",
            f"### Queries whose gold paper `{focus}` lost",
            "",
            "Query ids only; the text stays under `DATA_DIR`.",
            "",
        ]
        for name, entry in missed.items():
            never = ", ".join(entry["not_in_pool"]) or "none"
            cut = ", ".join(entry["beyond_depth"]) or "none"
            lines.append(
                f"- **{name}** ({entry['queries']} queries) — not in pool: {never}; "
                f"in pool, beyond {depth}: {cut}"
            )
    data = {
        "queries": queries,
        "stage_losses": losses,
        "gold_ranks": ranks,
        "misses": {focus: missed},
    }
    return lines, data


def render_gaps(
    run: str | Path,
    *,
    splits: Sequence[str] | None = None,
    focus: str = "hybrid_rerank",
    depth: int = 50,
) -> Path:
    """Write ``<run>/gaps.md`` and ``<run>/gaps.json`` from the recorded splits."""

    directory = Path(run)
    chosen = list(splits) if splits else [
        split for split in DIAGNOSTIC_SPLITS if (directory / split / "per_query.parquet").is_file()
    ]
    if not chosen:
        raise GapError("no_recorded_runs", str(directory))
    loaded = {split: load_outcomes(directory, split) for split in chosen}
    first = next(iter(loaded.values()))[0]
    lines = [
        f"# Retrieval gaps — {first['experiment']['name']}",
        "",
        f"Rendered by `eval gaps` from `{directory}/<split>/per_query.parquet`; nothing here "
        "was rerun. Corpus release "
        f"`{first['corpus']['release_id']}`, labels "
        # Runs recorded before E3 scored only our own paper ids and did not say so.
        f"`{first['dataset'].get('labels', 'paper_id')}`.",
    ]
    report: dict[str, Any] = {
        "run": str(directory),
        "depth": depth,
        "focus": focus,
        "splits": {},
    }
    for split, (manifest, outcomes) in loaded.items():
        section, data = _split_lines(split, manifest, outcomes, focus=focus, depth=depth)
        lines += ["", *section]
        report["splits"][split] = data
    path = directory / "gaps.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "gaps.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
