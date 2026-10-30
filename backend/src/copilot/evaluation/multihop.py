"""MultiHop-RAG (agent follow-ups, 2026-10-28): the planning ablation's multi-part questions.

Tang and Yang's benchmark (ODC-BY 1.0), pinned at Hugging Face ``yixuantt/MultiHopRAG`` revision
``71ac0d0b``: 609 news articles and 2,556 questions whose evidence spans two to four articles.
Only the comparison and temporal questions are used, sampled by sha256 of the question into 15 +
15 development and 75 + 75 test. Files are fetched offline, from the CLI only, into
``DATA_DIR/benchmarks/multihop/<revision>/``; the repository keeps ids, types, splits and the
evidence article URLs, never a question, answer or article text.

Each article becomes one paper in its own database (``copilot_multihop``) and never-activated
release (``multihop-rag-v1``), its body chunked by the live chunker. Every chunk starts with the
article's source and publication date, so a temporal question can be answered from a passage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx
from sqlalchemy import Engine, insert, text

from ..corpus.chunk import ChunkerConfig, Section, TokenSpans, chunk_document

MULTIHOP_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
MULTIHOP_REPOSITORY = "yixuantt/MultiHopRAG"
FILES = {"corpus": "corpus.json", "queries": "MultiHopRAG.json"}
RELEASE = "multihop-rag-v1"
SOURCE = "multihop"
IDENTIFIER = "multihop"
SECTION = "Body"
TYPES = ("comparison_query", "temporal_query")
DEVELOPMENT_PER_TYPE = 15
TEST_PER_TYPE = 75
_PAPER_NAMESPACE = uuid5(
    NAMESPACE_URL, "https://huggingface.co/datasets/yixuantt/MultiHopRAG#paper"
)
_VERSION_NAMESPACE = uuid5(
    NAMESPACE_URL, "https://huggingface.co/datasets/yixuantt/MultiHopRAG#version"
)
_CHUNK_NAMESPACE = uuid5(
    NAMESPACE_URL, "https://huggingface.co/datasets/yixuantt/MultiHopRAG#chunk"
)


class MultihopError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class MultihopQuery:
    id: str
    type: str
    split: str
    question: str
    answer: str
    evidence_urls: tuple[str, ...]


def benchmark_root(data_dir: Path) -> Path:
    return Path(data_dir) / "benchmarks" / "multihop" / MULTIHOP_REVISION


def file_path(data_dir: Path, key: str) -> Path:
    return benchmark_root(data_dir) / FILES[key]


def fetch_multihop(data_dir: Path, *, client: httpx.Client | None = None) -> dict[str, str]:
    """Download the pinned files that are not already here; their sha256 by key."""

    root = benchmark_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{MULTIHOP_REPOSITORY}/resolve/{MULTIHOP_REVISION}"
    owned = client is None
    http = client or httpx.Client(timeout=120.0, follow_redirects=True)
    digests: dict[str, str] = {}
    try:
        for key, name in FILES.items():
            target = root / name
            if not target.exists():
                partial = target.with_suffix(target.suffix + ".part")
                with http.stream("GET", f"{base}/{name}") as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for block in response.iter_bytes():
                            handle.write(block)
                partial.rename(target)
            digests[key] = hashlib.sha256(target.read_bytes()).hexdigest()
    finally:
        if owned:
            http.close()
    return digests


# --- Questions ------------------------------------------------------------------------------


def query_id(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def sample_queries(
    rows: Iterable[Mapping[str, Any]],
    *,
    development: int = DEVELOPMENT_PER_TYPE,
    test: int = TEST_PER_TYPE,
) -> list[MultihopQuery]:
    """Comparison and temporal questions only, ordered by their id within each type: the first
    ``development`` of each type are development, the next ``test`` are test. Independent of
    file order; a question asked twice is kept once."""

    by_type: dict[str, dict[str, Mapping[str, Any]]] = {kind: {} for kind in TYPES}
    for row in rows:
        kind = str(row["question_type"])
        if kind in by_type:
            by_type[kind].setdefault(query_id(str(row["query"])), row)
    sampled: list[MultihopQuery] = []
    for kind in TYPES:
        ordered = sorted(by_type[kind].items())
        if len(ordered) < development + test:
            raise MultihopError("too_few_questions", f"{kind}: {len(ordered)}")
        for index, (qid, row) in enumerate(ordered[: development + test]):
            urls = tuple(dict.fromkeys(str(item["url"]) for item in row["evidence_list"]))
            sampled.append(
                MultihopQuery(
                    id=qid,
                    type=kind,
                    split="development" if index < development else "test",
                    question=str(row["query"]),
                    answer=str(row["answer"]),
                    evidence_urls=urls,
                )
            )
    return sampled


def check_evidence(queries: Sequence[MultihopQuery], known_urls: set[str]) -> None:
    missing = sorted({url for q in queries for url in q.evidence_urls if url not in known_urls})
    if missing:
        raise MultihopError("evidence_outside_corpus", f"{len(missing)} urls, e.g. {missing[:3]}")


def load_queries(data_dir: Path) -> list[MultihopQuery]:
    rows = json.loads(file_path(data_dir, "queries").read_text(encoding="utf-8"))
    return sample_queries(rows)


# --- Articles ---------------------------------------------------------------------------------


def article_header(article: Mapping[str, Any]) -> str:
    date = str(article.get("published_at") or "")[:10]
    source = " ".join(str(article.get("source") or "unknown source").split())
    return f"{source}, published {date}.\n" if date else f"{source}.\n"


def chunk_article(
    article: Mapping[str, Any], spans: TokenSpans, chunker: ChunkerConfig
) -> list[dict[str, Any]]:
    """The body under the live chunker as one section, each chunk led by source and date."""

    body = str(article.get("body") or "").replace("\x00", "")
    chunks = chunk_document([Section(SECTION, body, None, None, 0)], spans, chunker)
    header = article_header(article)
    for chunk in chunks:
        chunk["text"] = header + chunk["text"]
        chunk["token_count"] = len(spans(chunk["text"]))
    return chunks


def paper_id_for(url: str) -> UUID:
    return uuid5(_PAPER_NAMESPACE, url)


def _guard_database(engine: Engine) -> None:
    name = engine.url.database or ""
    if not (name.endswith("multihop") or name.startswith("test_")):
        raise MultihopError("database_refused", f"{name!r} is not a MultiHop-RAG database")


def load_multihop_corpus(
    articles: Sequence[Mapping[str, Any]],
    *,
    engine: Engine,
    spans: TokenSpans,
    chunker: ChunkerConfig,
    queries: Sequence[MultihopQuery],
) -> dict[str, int]:
    """Every article as one paper, one parsed version and its chunks; evidence checked last.

    Refuses a database not named for MultiHop-RAG, and one that already holds papers.
    """

    from ..db.models import Chunk, Paper, PaperIdentifier, PaperVersion

    _guard_database(engine)
    with engine.connect() as connection:
        if connection.execute(text("select exists (select 1 from papers)")).scalar_one():
            raise MultihopError("database_not_empty", str(engine.url.database))
    urls: set[str] = set()
    papers, identifiers, versions, chunks = [], [], [], []
    for article in articles:
        url = str(article["url"])
        if url in urls:
            raise MultihopError("duplicate_article", url)
        urls.add(url)
        paper = paper_id_for(url)
        version = uuid5(_VERSION_NAMESPACE, url)
        body = str(article.get("body") or "")
        title = " ".join(str(article.get("title") or "").split()) or url
        papers.append({"id": paper, "title": title})
        identifiers.append(
            {
                "id": uuid5(paper, IDENTIFIER),
                "paper_id": paper,
                "namespace": IDENTIFIER,
                "value": url,
            }
        )
        versions.append(
            {
                "id": version,
                "paper_id": paper,
                "source": SOURCE,
                "source_url": url,
                "source_revision": MULTIHOP_REVISION,
                "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "redistribution": "unknown",
                "parser_version": chunker.chunker_version,
                "parse_status": "parsed",
            }
        )
        for chunk in chunk_article(article, spans, chunker):
            chunks.append(
                {
                    "id": uuid5(_CHUNK_NAMESPACE, f"{url}#{chunk['ordinal']}"),
                    "paper_version_id": version,
                    "section_path": SECTION,
                    "section_ordinal": 0,
                    "kind": "body",
                    "ordinal": chunk["ordinal"],
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                    "parser_version": chunker.chunker_version,
                    "chunker_version": chunker.chunker_version,
                    "evidence_default": True,
                }
            )
    check_evidence(queries, urls)
    with engine.begin() as connection:
        connection.execute(insert(Paper), papers)
        connection.execute(insert(PaperIdentifier), identifiers)
        connection.execute(insert(PaperVersion), versions)
        for start in range(0, len(chunks), 2_000):
            connection.execute(insert(Chunk), chunks[start : start + 2_000])
        connection.execute(text("analyze"))
    return {
        "articles": len(urls),
        "chunks": len(chunks),
        "max_chunk_tokens": max((c["token_count"] for c in chunks), default=0),
    }


def urls_for(engine: Engine, paper_ids: Sequence[str]) -> dict[str, str]:
    """Paper id to article URL, for mapping a search's papers back to the evidence."""

    if not paper_ids:
        return {}
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "select paper_id::text, value from paper_identifiers"
                " where namespace = :ns and paper_id = any(cast(:ids as uuid[]))"
            ),
            {"ns": IDENTIFIER, "ids": list(paper_ids)},
        ).all()
    return {str(row[0]): str(row[1]) for row in rows}


# --- Fixtures in the repository: ids, types, splits, evidence URLs; no text --------------------


def write_fixtures(
    out: Path, queries: Sequence[MultihopQuery], digests: Mapping[str, str]
) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"id": q.id, "type": q.type, "split": q.split, "evidence_urls": list(q.evidence_urls)},
            sort_keys=True,
        )
        for q in queries
    ]
    (out / "queries.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "source.json").write_text(
        json.dumps(
            {
                "repository": f"huggingface.co/datasets/{MULTIHOP_REPOSITORY}",
                "revision": MULTIHOP_REVISION,
                "license": "ODC-BY-1.0",
                "files": FILES,
                "sha256": dict(digests),
                "types": list(TYPES),
                "per_type": {"development": DEVELOPMENT_PER_TYPE, "test": TEST_PER_TYPE},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for q in queries:
        counts[f"{q.split}.{q.type}"] = counts.get(f"{q.split}.{q.type}", 0) + 1
    return counts


def assignment(fixtures: Path) -> dict[str, str]:
    """Question id to split, from the repository's frozen fixtures."""

    rows = (json.loads(line) for line in (fixtures / "queries.jsonl").read_text().splitlines())
    return {str(row["id"]): str(row["split"]) for row in rows if row}
