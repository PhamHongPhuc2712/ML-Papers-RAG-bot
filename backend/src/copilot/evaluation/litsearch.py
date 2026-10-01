"""The LitSearch retrieval benchmark, pinned, and matched to our corpus.

LitSearch is 597 expert-checked literature-search queries over a 64,183
document corpus (Ajith et al., EMNLP 2024). It replaces the 150 hand-judged
query families spec §11 assumed, because those queries already exist and were
checked by their authors.

Two facts about matching it to our corpus shape this module.

The benchmark carries strong identifiers — ``externalids`` holds ``acl``,
``arxiv``, ``doi``, ``dblp`` — but **our corpus carries none of them**. Papers
are keyed by the papercli registry id and, for OpenReview venues, a forum id.
The one bridge is the ACL Anthology: 9,128 papers record an
``aclanthology.org`` PDF URL with the anthology id inside it, which is the same
id LitSearch stores. So strong-ID matching covers the ACL-family venues and
normalized-title matching covers the rest, and every match records which method
found it so a reader can discount the weaker one.

And a gold set that is only *partly* present is not coverage. Recall computed
against two of a query's three gold papers understates the system and does it
silently, so ``gold_coverage`` counts covered, partial and missing separately
and ``fraction`` counts only the fully covered.

The dataset repo declares no license. Until that is resolved this data is
local-use-only: it lives under ``DATA_DIR``, never in the repository, and never
enters a corpus export.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..corpus.mirror import hf_fetch

LITSEARCH_REPO = "princeton-nlp/LitSearch"
# Verified 2026-09-21: 597 queries, 64,183 documents.
LITSEARCH_REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"

QUERY_MEMBER = "query/full-00000-of-00001.parquet"
CORPUS_CLEAN_MEMBERS = tuple(f"corpus_clean/full-0000{i}-of-00006.parquet" for i in range(6))
CORPUS_S2ORC_MEMBERS = tuple(f"corpus_s2orc/full-0000{i}-of-00008.parquet" for i in range(8))

# Fixed for good: a LitSearch document's id in our snapshot format is derived from
# its corpusid, so gold labels and indexed points meet without a lookup table.
LITSEARCH_NAMESPACE = uuid.UUID("2b6c3f1e-8d47-4f0a-9c55-3e1d7a9b4c20")

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ACL_URL = re.compile(r"aclanthology\.org/([0-9]{4}\.[a-z0-9-]+\.[0-9]+)")


class LitSearchError(RuntimeError):
    """The benchmark cannot be loaded or matched."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


class UnpinnedRevisionError(LitSearchError):
    """A benchmark fetched from a moving ref is not a reproducible benchmark."""

    def __init__(self, revision: str) -> None:
        super().__init__("unpinned_revision", revision or "<empty>")


@dataclass(frozen=True, slots=True)
class LitSearchPaths:
    root: Path
    query: Path
    corpus_clean: tuple[Path, ...]
    corpus_s2orc: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class LitSearchQuery:
    query_id: str
    query: str
    query_set: str
    specificity: int
    quality: int
    corpusids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LitSearchDoc:
    corpusid: str
    title: str
    acl_id: str | None
    arxiv_id: str | None
    doi: str | None


@dataclass(frozen=True, slots=True)
class CorpusPaper:
    paper_id: str
    title: str
    pdf_url: str | None
    year: int | None


@dataclass(frozen=True, slots=True)
class Match:
    corpusid: str
    paper_id: str
    method: str


@dataclass(slots=True)
class MatchReport:
    matches: dict[str, Match] = field(default_factory=dict)
    matched_by_strong_id: int = 0
    matched_by_title: int = 0
    unmatched: int = 0
    ambiguous_titles: int = 0

    @property
    def matched(self) -> set[str]:
        return set(self.matches)


@dataclass(frozen=True, slots=True)
class CoverageCount:
    covered: int
    partial: int
    missing: int

    @property
    def total(self) -> int:
        return self.covered + self.partial + self.missing

    @property
    def fraction(self) -> float:
        """Share of queries whose gold set is **entirely** in our corpus."""

        return self.covered / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class GoldCoverage:
    covered: int
    partial: int
    missing: int
    by_query_set: dict[str, CoverageCount]

    @property
    def total(self) -> int:
        return self.covered + self.partial + self.missing

    @property
    def fraction(self) -> float:
        return self.covered / self.total if self.total else 0.0


def normalize_title(title: str) -> str:
    """Case, punctuation and whitespace removed; NFKC first."""

    return _NON_ALNUM.sub("", unicodedata.normalize("NFKC", title or "").lower())


def acl_id_from_url(url: str | None) -> str | None:
    """The ACL Anthology id embedded in a PDF URL, if there is one."""

    if not url:
        return None
    found = _ACL_URL.search(url)
    return found.group(1) if found else None


def fetch_litsearch(dest: Path, revision: str = LITSEARCH_REVISION) -> LitSearchPaths:
    """Download the three configurations at a pinned commit.

    Refuses a moving ref. A benchmark fetched from ``main`` cannot be compared
    with a run from last month, and the failure is silent.
    """

    if not _HEX40.match(revision or ""):
        raise UnpinnedRevisionError(revision)
    dest.mkdir(parents=True, exist_ok=True)
    query = hf_fetch(LITSEARCH_REPO, revision, QUERY_MEMBER, dest)
    clean = tuple(hf_fetch(LITSEARCH_REPO, revision, m, dest) for m in CORPUS_CLEAN_MEMBERS)
    s2orc = tuple(hf_fetch(LITSEARCH_REPO, revision, m, dest) for m in CORPUS_S2ORC_MEMBERS)
    return LitSearchPaths(root=dest, query=query, corpus_clean=clean, corpus_s2orc=s2orc)


def load_queries(paths: LitSearchPaths) -> list[LitSearchQuery]:
    """The 597 queries, with their annotations and gold corpus ids."""

    import pyarrow.parquet as pq

    rows = pq.read_table(paths.query).to_pylist()
    return [
        LitSearchQuery(
            query_id=f"litsearch-{index:04d}",
            query=str(row["query"]),
            query_set=str(row["query_set"]),
            specificity=int(row["specificity"]),
            quality=int(row["quality"]),
            corpusids=tuple(str(c) for c in row["corpusids"]),
        )
        for index, row in enumerate(rows)
    ]


def load_docs(paths: LitSearchPaths) -> list[LitSearchDoc]:
    """Corpus documents joined to their external identifiers.

    Titles come from ``corpus_clean`` and identifiers from ``corpus_s2orc``;
    the two configurations describe the same 64,183 documents.
    """

    import pyarrow.parquet as pq

    external: dict[str, dict[str, str | None]] = {}
    for shard in paths.corpus_s2orc:
        table = pq.read_table(shard, columns=["corpusid", "externalids"])
        for row in table.to_pylist():
            ids = row.get("externalids") or {}
            external[str(row["corpusid"])] = {
                "acl": ids.get("acl"),
                "arxiv": ids.get("arxiv"),
                "doi": ids.get("doi"),
            }

    docs: list[LitSearchDoc] = []
    for shard in paths.corpus_clean:
        table = pq.read_table(shard, columns=["corpusid", "title"])
        for row in table.to_pylist():
            corpusid = str(row["corpusid"])
            ids = external.get(corpusid, {})
            docs.append(
                LitSearchDoc(
                    corpusid=corpusid,
                    title=str(row["title"] or ""),
                    acl_id=ids.get("acl"),
                    arxiv_id=ids.get("arxiv"),
                    doi=ids.get("doi"),
                )
            )
    return docs


def match_to_corpus(docs: Iterable[LitSearchDoc], papers: Iterable[CorpusPaper]) -> MatchReport:
    """Map benchmark documents onto our papers, strongest evidence first.

    ACL Anthology id beats normalized title, because the id is evidence and
    the title is a guess that usually works. A normalized title shared by two
    of our papers resolves to neither: guessing between them would put a false
    gold into the evaluation, which is worse than a smaller slice.
    """

    by_acl: dict[str, str] = {}
    by_title: defaultdict[str, list[str]] = defaultdict(list)
    for paper in papers:
        acl = acl_id_from_url(paper.pdf_url)
        if acl:
            by_acl[acl] = paper.paper_id
        normalized = normalize_title(paper.title)
        if normalized:
            by_title[normalized].append(paper.paper_id)

    report = MatchReport()
    for doc in docs:
        if doc.acl_id and doc.acl_id in by_acl:
            report.matches[doc.corpusid] = Match(doc.corpusid, by_acl[doc.acl_id], "acl_id")
            report.matched_by_strong_id += 1
            continue
        candidates = by_title.get(normalize_title(doc.title), [])
        if len(candidates) == 1:
            report.matches[doc.corpusid] = Match(doc.corpusid, candidates[0], "title")
            report.matched_by_title += 1
        else:
            if len(candidates) > 1:
                report.ambiguous_titles += 1
            report.unmatched += 1
    return report


def gold_coverage(queries: Sequence[LitSearchQuery], matched: set[str]) -> GoldCoverage:
    """How many queries our corpus can actually answer.

    A query counts as covered only when **every** gold paper is present.
    Partial presence is reported on its own line rather than folded into
    either side.
    """

    covered = partial = missing = 0
    per_set: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for query in queries:
        golds = set(query.corpusids)
        present = golds & matched
        if golds and present == golds:
            covered += 1
            per_set[query.query_set][0] += 1
        elif present:
            partial += 1
            per_set[query.query_set][1] += 1
        else:
            missing += 1
            per_set[query.query_set][2] += 1
    return GoldCoverage(
        covered=covered,
        partial=partial,
        missing=missing,
        by_query_set={
            name: CoverageCount(covered=c, partial=p, missing=m)
            for name, (c, p, m) in per_set.items()
        },
    )


def litsearch_paper_id(corpusid: str | int) -> str:
    """The UUID a LitSearch document carries in a snapshot and in Qdrant.

    Qdrant point ids are UUIDs or integers, and every release keys its points
    by paper id. A name-based UUID keeps the corpusid recoverable from the
    snapshot and needs no table to map labels onto points.
    """

    return str(uuid.uuid5(LITSEARCH_NAMESPACE, f"litsearch:corpusid:{corpusid}"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_paths(root: Path, checksums: Mapping[str, str]) -> LitSearchPaths:
    """The pinned files E1 captured, checked against their recorded sha256, offline.

    Only the members a conversion reads are hashed: the query file and the six
    ``corpus_clean`` shards. A missing or altered file is refused, not refetched.
    """

    for member in (QUERY_MEMBER, *CORPUS_CLEAN_MEMBERS):
        path = root / member
        if not path.is_file():
            raise LitSearchError("benchmark_file_missing", str(path))
        expected = checksums.get(member)
        if expected is None:
            raise LitSearchError("benchmark_checksum_unpinned", member)
        if _file_sha256(path) != expected:
            raise LitSearchError("benchmark_checksum_mismatch", member)
    return LitSearchPaths(
        root=root,
        query=root / QUERY_MEMBER,
        corpus_clean=tuple(root / member for member in CORPUS_CLEAN_MEMBERS),
        corpus_s2orc=tuple(root / member for member in CORPUS_S2ORC_MEMBERS),
    )


def write_litsearch_snapshot(
    paths: LitSearchPaths,
    destination: Path,
    *,
    run_id: str = "litsearch-v1",
    revision: str = LITSEARCH_REVISION,
    batch_rows: int = 20_000,
) -> dict[str, Any]:
    """``corpus_clean`` as a schema-2 corpus snapshot that ``search build-index`` reads.

    Title and abstract only, the shape the benchmark is packaged in, so the
    headline number never mixes in full text. The chunks table is written
    empty, its digest is that of no rows, and every row is marked
    ``redistribution: unknown``: LitSearch declares no license, so the
    snapshot stays under ``DATA_DIR`` like the rest of the benchmark.
    """

    import pyarrow.parquet as pq

    from ..corpus.export import (
        CHUNKS_SCHEMA,
        DIGEST_RULE,
        PAPERS_SCHEMA,
        SCHEMA_VERSION,
        SHARD_BYTES,
        _ShardWriter,
    )
    from ..search.lexical import paper_text

    target = Path(destination)
    if (target / "manifest.json").exists():
        raise LitSearchError("snapshot_exists", str(target))
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard in paths.corpus_clean:
        source = pq.ParquetFile(shard)
        for batch in source.iter_batches(columns=["corpusid", "title", "abstract"]):
            for row in batch.to_pylist():
                corpusid = str(row["corpusid"])
                if corpusid in seen:
                    raise LitSearchError("duplicate_corpusid", corpusid)
                seen.add(corpusid)
                title = str(row["title"] or "")
                abstract = row["abstract"] or None
                document = paper_text(title, abstract)
                rows.append(
                    {
                        "paper_id": litsearch_paper_id(corpusid),
                        "title": title,
                        "abstract": abstract,
                        "source": "litsearch",
                        "source_revision": revision,
                        "redistribution": "unknown",
                        "identifiers": json.dumps({"litsearch_corpusid": corpusid}),
                        "text_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
                    }
                )
    # The digest rule and the index build both read papers in id order.
    rows.sort(key=lambda row: str(row["paper_id"]))
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['paper_id']}\t{row['text_sha256']}\n".encode())
    papers = _ShardWriter(target, "papers", PAPERS_SCHEMA, SHARD_BYTES)
    for start in range(0, len(rows), batch_rows):
        papers.write(rows[start : start + batch_rows])
    chunks = _ShardWriter(target, "chunks", CHUNKS_SCHEMA, SHARD_BYTES)
    golds = {corpusid for query in load_queries(paths) for corpusid in query.corpusids}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "shards": [*papers.close(), *chunks.close()],
        "counts": {
            "papers": len(rows),
            "chunks_total": 0,
            "chunks_exported": 0,
            "withheld_chunks": 0,
        },
        "digests": {"papers": digest.hexdigest(), "chunks": hashlib.sha256().hexdigest()},
        "digest_rule": DIGEST_RULE,
        "versions": {"parser": [], "chunker": []},
        "rights": {"exported_when": "allowed", "withheld": ["unknown"]},
        "source": {
            "benchmark": "litsearch",
            "repo": LITSEARCH_REPO,
            "revision": revision,
            "members": list(CORPUS_CLEAN_MEMBERS),
            "id_rule": "uuid5(LITSEARCH_NAMESPACE, 'litsearch:corpusid:<corpusid>')",
        },
        "golds": {"distinct": len(golds), "missing_from_corpus": len(golds - seen)},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return manifest
