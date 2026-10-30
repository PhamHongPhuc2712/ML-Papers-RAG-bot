"""MTRAG ClapNQ (agent experiment, 2026-10-25) and Govt (agent follow-ups, 2026-10-28): IBM's
human-written multi-turn RAG benchmark, one domain at a time.

Pinned at IBM ``mt-rag-benchmark`` commit ``cc5b1d48``, as haiku.rag pins it: 183,408 Wikipedia
passages, 29 conversations, 224 turns, 208 of them with passage-level relevance judgments
(qrels) and a person's standalone rewrite. Files are fetched offline, from the CLI only, into
``DATA_DIR/benchmarks/mtrag/<commit>/``; the repository keeps ids, labels and splits.

The passages go into their own database (``copilot_mtrag``) and never-activated release
(``mtrag-clapnq-v1``), each passage one paper with one chunk, its text verbatim: the search's
top 10 papers are then the top 10 passages, the unit IBM's retrieval numbers are reported in.
They are written in bulk, not through identity resolution, which would merge passages that
share an article title. The passage id is kept as the paper's ``mtrag`` identifier, so a
qrel maps back to a paper.

Govt, read only by the follow-ups' one test scoring, is loaded the same way into
``copilot_mtrag_govt`` and release ``mtrag-govt-v1``; all its conversations are test. FiQA,
read only by the self-check's one test scoring (2026-10-29), likewise into
``copilot_mtrag_fiqa`` and release ``mtrag-fiqa-v1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx
from sqlalchemy import Engine, insert, text

MTRAG_COMMIT = "cc5b1d481b391181b89f7ced860308482e785463"
MTRAG_REPOSITORY = "IBM/mt-rag-benchmark"
CLAPNQ_COLLECTION = "mt-rag-clapnq-elser-512-100-20240503"
FILES = {
    "corpus": "corpora/passage_level/clapnq.jsonl.zip",
    "qrels": "mtrag-human/retrieval_tasks/clapnq/qrels/dev.tsv",
    "lastturn": "mtrag-human/retrieval_tasks/clapnq/clapnq_lastturn.jsonl",
    "rewrite": "mtrag-human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl",
    "reference": "mtrag-human/generation_tasks/reference.jsonl",
}
RELEASE = "mtrag-clapnq-v1"
SOURCE = "mtrag"
IDENTIFIER = "mtrag"
CHUNKER_VERSION = "mtrag-passage-v1"
SECTION = "Passage"
DEVELOPMENT_CONVERSATIONS = 5
# Every id derived from a passage id is a uuid5 under these, so a reload writes the same ids.
_PAPER_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/IBM/mt-rag-benchmark#paper")
_VERSION_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/IBM/mt-rag-benchmark#version")
_CHUNK_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/IBM/mt-rag-benchmark#chunk")
LOAD_BATCH = 2_000


@dataclass(frozen=True)
class MtragDomain:
    """One MTRAG domain: its reference-task collection, its files at the pinned commit, where
    they are kept (``subdir`` under the commit's directory), its release and its fixtures."""

    name: str
    collection: str
    files: Mapping[str, str]
    release: str
    subdir: str
    development: int
    fixtures: Path


def _domain_files(name: str) -> dict[str, str]:
    tasks = f"mtrag-human/retrieval_tasks/{name}"
    return {
        "corpus": f"corpora/passage_level/{name}.jsonl.zip",
        "qrels": f"{tasks}/qrels/dev.tsv",
        "lastturn": f"{tasks}/{name}_lastturn.jsonl",
        "rewrite": f"{tasks}/{name}_rewrite.jsonl",
        "reference": "mtrag-human/generation_tasks/reference.jsonl",
    }


DOMAINS: dict[str, MtragDomain] = {
    # ClapNQ's files sit directly under the commit's directory, as fetched on 2026-10-25.
    "clapnq": MtragDomain(
        "clapnq",
        CLAPNQ_COLLECTION,
        FILES,
        RELEASE,
        "",
        DEVELOPMENT_CONVERSATIONS,
        Path("data/fixtures/mtrag"),
    ),
    "govt": MtragDomain(
        "govt",
        "mt-rag-govt-elser-512-100-20240611",
        _domain_files("govt"),
        "mtrag-govt-v1",
        "govt",
        0,
        Path("data/fixtures/mtrag-govt"),
    ),
    "fiqa": MtragDomain(
        "fiqa",
        "mt-rag-fiqa-beir-elser-512-100-20240501",
        _domain_files("fiqa"),
        "mtrag-fiqa-v1",
        "fiqa",
        0,
        Path("data/fixtures/mtrag-fiqa"),
    ),
}
CLAPNQ = DOMAINS["clapnq"]


def domain_for_release(release: str) -> MtragDomain:
    for domain in DOMAINS.values():
        if domain.release == release:
            return domain
    raise MtragError("unknown_release", release)


class MtragError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class MtragTurn:
    task_id: str
    turn: int
    # The last user message, as written.
    question: str
    # The reference conversation before it: (speaker, text), speaker "user" or "agent".
    history: list[tuple[str, str]]
    reference: str
    answerability: str
    multi_turn_type: str
    relevant: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MtragConversation:
    conversation_id: str
    split: str
    turns: list[MtragTurn]


def benchmark_root(data_dir: Path, domain: MtragDomain = CLAPNQ) -> Path:
    root = Path(data_dir) / "benchmarks" / "mtrag" / MTRAG_COMMIT
    return root / domain.subdir if domain.subdir else root


# --- Parsing ------------------------------------------------------------------------------


def parse_qrels(lines: Iterable[str]) -> dict[str, list[str]]:
    """Corpus ids per query id, in file order; the first line is the header."""

    qrels: dict[str, list[str]] = {}
    rows = iter(lines)
    next(rows, None)
    for line in rows:
        if not line.strip():
            continue
        query_id, corpus_id, _score = line.rstrip("\n").split("\t")
        qrels.setdefault(query_id, []).append(corpus_id)
    return qrels


def check_qrels_resolve(corpus_ids: set[str], qrels: Mapping[str, list[str]]) -> None:
    missing = sorted({cid for ids in qrels.values() for cid in ids if cid not in corpus_ids})
    if missing:
        raise MtragError("qrels_unresolved", f"{len(missing)} ids, e.g. {missing[:3]}")


def assign_splits(
    conversation_ids: Sequence[str], *, development: int = DEVELOPMENT_CONVERSATIONS
) -> dict[str, str]:
    """The first ``development`` conversations by sha256 of their id; the rest are test."""

    ordered = sorted(conversation_ids, key=lambda cid: hashlib.sha256(cid.encode()).hexdigest())
    return {cid: ("development" if i < development else "test") for i, cid in enumerate(ordered)}


def _turn(task: Mapping[str, Any], qrels: Mapping[str, list[str]]) -> MtragTurn:
    messages = [(str(m["speaker"]), str(m["text"])) for m in task["input"]]
    if not messages or messages[-1][0] != "user":
        raise MtragError("task_without_question", str(task["task_id"]))
    return MtragTurn(
        task_id=str(task["task_id"]),
        turn=int(task["turn"]),
        question=messages[-1][1],
        history=messages[:-1],
        reference=str(task["targets"][0]["text"]),
        answerability=str(task["Answerability"][0]),
        multi_turn_type=str(task["Multi-Turn"][0]),
        relevant=list(qrels.get(str(task["task_id"]), [])),
    )


def load_conversations(
    reference_path: Path, qrels: Mapping[str, list[str]], domain: MtragDomain = CLAPNQ
) -> list[MtragConversation]:
    """One domain's conversations from the generation tasks, turns in order, split assigned."""

    grouped: dict[str, list[MtragTurn]] = {}
    for line in Path(reference_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if task.get("Collection") != domain.collection:
            continue
        conversation_id = str(task["task_id"]).split("<::>")[0]
        grouped.setdefault(conversation_id, []).append(_turn(task, qrels))
    splits = assign_splits(list(grouped), development=domain.development)
    return [
        MtragConversation(cid, splits[cid], sorted(turns, key=lambda t: t.turn))
        for cid, turns in sorted(grouped.items())
    ]


def load_queries(path: Path) -> dict[str, str]:
    """``_id`` to text, from a retrieval-task query file (last turn or rewrite)."""

    rows = (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines())
    return {str(row["_id"]): str(row["text"]) for row in rows if row}


def iter_corpus(corpus_zip: Path) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(corpus_zip) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield {"_id": str(row["_id"]), "title": str(row["title"]), "text": str(row["text"])}


# --- Fetching (offline, CLI only) ---------------------------------------------------------


def fetch_mtrag(
    data_dir: Path, *, client: httpx.Client | None = None, domain: MtragDomain = CLAPNQ
) -> dict[str, str]:
    """Download the domain's pinned files that are not already here; their sha256 by key."""

    root = benchmark_root(data_dir, domain)
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{MTRAG_REPOSITORY}/{MTRAG_COMMIT}"
    owned = client is None
    http = client or httpx.Client(timeout=120.0, follow_redirects=True)
    digests: dict[str, str] = {}
    try:
        for key, relative in domain.files.items():
            target = root / Path(relative).name
            if not target.exists():
                partial = target.with_suffix(target.suffix + ".part")
                with http.stream("GET", f"{base}/{relative}") as response:
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


def file_path(data_dir: Path, key: str, domain: MtragDomain = CLAPNQ) -> Path:
    return benchmark_root(data_dir, domain) / Path(domain.files[key]).name


# --- Loading ------------------------------------------------------------------------------


def paper_id_for(passage_id: str) -> UUID:
    return uuid5(_PAPER_NAMESPACE, passage_id)


def _batches(rows: Iterable[dict[str, str]], size: int) -> Iterator[list[dict[str, str]]]:
    batch: list[dict[str, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def strip_nul(value: str) -> str:
    """PostgreSQL text cannot hold NUL; five Govt passages carry one (2026-10-28)."""

    return value.replace("\x00", "")


def _guard_database(engine: Engine) -> None:
    name = engine.url.database or ""
    if not (name.endswith(("mtrag", "mtrag_govt", "mtrag_fiqa")) or name.startswith("test_")):
        raise MtragError("database_refused", f"{name!r} is not an MTRAG database")


def load_mtrag_corpus(
    passages: Iterable[dict[str, str]],
    *,
    engine: Engine,
    count_tokens: Callable[[str], int],
    qrels: Mapping[str, list[str]],
    progress: Callable[[int], None] | None = None,
) -> dict[str, int]:
    """Every passage as one paper, one parsed version and one chunk; qrels checked last.

    Refuses a database not named for MTRAG, and one that already holds papers: a load is
    all or nothing, and a reload starts from a fresh database.
    """

    from ..db.models import Chunk, Paper, PaperIdentifier, PaperVersion

    _guard_database(engine)
    with engine.connect() as connection:
        if connection.execute(text("select exists (select 1 from papers)")).scalar_one():
            raise MtragError("database_not_empty", str(engine.url.database))
    loaded: set[str] = set()
    over = 0
    nul = 0
    with engine.begin() as connection:
        for batch in _batches(passages, LOAD_BATCH):
            papers, identifiers, versions, chunks = [], [], [], []
            for row in batch:
                passage_id = row["_id"]
                if passage_id in loaded:
                    raise MtragError("duplicate_passage", passage_id)
                loaded.add(passage_id)
                paper = paper_id_for(passage_id)
                version = uuid5(_VERSION_NAMESPACE, passage_id)
                body = strip_nul(row["text"])
                nul += int(body != row["text"])
                tokens = count_tokens(body)
                over += int(tokens > 1_200)
                title = " ".join(strip_nul(row["title"]).split())
                papers.append({"id": paper, "title": title or passage_id})
                identifiers.append(
                    {
                        "id": uuid5(paper, IDENTIFIER),
                        "paper_id": paper,
                        "namespace": IDENTIFIER,
                        "value": passage_id,
                    }
                )
                versions.append(
                    {
                        "id": version,
                        "paper_id": paper,
                        "source": SOURCE,
                        "source_url": f"https://github.com/{MTRAG_REPOSITORY}#{passage_id}",
                        "source_revision": MTRAG_COMMIT,
                        "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                        "redistribution": "unknown",
                        "parser_version": CHUNKER_VERSION,
                        "parse_status": "parsed",
                    }
                )
                chunks.append(
                    {
                        "id": uuid5(_CHUNK_NAMESPACE, passage_id),
                        "paper_version_id": version,
                        "section_path": SECTION,
                        "section_ordinal": 0,
                        "kind": "body",
                        "ordinal": 0,
                        "text": body,
                        "token_count": tokens,
                        "parser_version": CHUNKER_VERSION,
                        "chunker_version": CHUNKER_VERSION,
                        "evidence_default": True,
                    }
                )
            connection.execute(insert(Paper), papers)
            connection.execute(insert(PaperIdentifier), identifiers)
            connection.execute(insert(PaperVersion), versions)
            connection.execute(insert(Chunk), chunks)
            if progress is not None:
                progress(len(loaded))
        check_qrels_resolve(loaded, qrels)
    return {
        "passages": len(loaded),
        "over_1200_tokens": over,
        "nul_stripped": nul,
        "qrels": sum(len(ids) for ids in qrels.values()),
    }


def passage_ids_for(engine: Engine, paper_ids: Sequence[str]) -> dict[str, str]:
    """Paper id to passage id, for mapping a search's papers back to qrels."""

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


# --- Fixtures in the repository: ids, labels, splits; no text -------------------------------


def write_fixtures(
    out: Path,
    conversations: Sequence[MtragConversation],
    digests: Mapping[str, str],
    domain: MtragDomain = CLAPNQ,
) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for conversation in conversations:
        for turn in conversation.turns:
            lines.append(
                json.dumps(
                    {
                        "conversation_id": conversation.conversation_id,
                        "task_id": turn.task_id,
                        "turn": turn.turn,
                        "split": conversation.split,
                        "answerability": turn.answerability,
                        "multi_turn_type": turn.multi_turn_type,
                        "relevant": turn.relevant,
                    },
                    sort_keys=True,
                )
            )
    (out / "conversations.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "source.json").write_text(
        json.dumps(
            {
                "repository": MTRAG_REPOSITORY,
                "commit": MTRAG_COMMIT,
                "files": dict(domain.files),
                "sha256": dict(digests),
                "collection": domain.collection,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "conversations": len(conversations),
        "turns": len(lines),
        "with_qrels": sum(1 for c in conversations for t in c.turns if t.relevant),
    }


# --- Passage retrieval metrics ------------------------------------------------------------


def passage_metrics(ranked: Sequence[str], relevant: Sequence[str]) -> dict[str, float]:
    """Recall@5/10 and nDCG@5/10 with binary relevance, as IBM reports them."""

    gold = set(relevant)
    metrics: dict[str, float] = {}
    for k in (5, 10):
        head = list(ranked[:k])
        hits = [1.0 if pid in gold else 0.0 for pid in head]
        metrics[f"recall@{k}"] = sum(hits) / len(gold) if gold else 0.0
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return metrics
