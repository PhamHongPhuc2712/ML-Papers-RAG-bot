"""A section-level query set from our own corpus (release plan S1).

Open RAG Bench asks questions a single passage answers; that is what the chunk-reading
reranker was better at there. This builds the same kind of question over our corpus so the
search page's reranker can be decided on our papers: one body passage per sampled paper,
stratified by venue-year, one model-written question per passage, the passage's paper as
the single gold. A question that names the paper (its distinctive title terms or two
consecutive title words) or copies eight or more consecutive words of the passage is
rejected: the first would be a lookup by name, the second a lookup by string.

S3 (2026-10-15) builds a second set the same way from abstracts: one topic-style question per
paper, the kind typed into a paper search box, as the paper-level guard LitSearch used to be.
Only papers search can return (with at least one chunk) are sampled, and S1's papers are left
out so the two sets stay independent. ``QuerySetSpec`` holds what differs between the two.

Questions are derived from corpus text, so they live under DATA_DIR; the repository keeps
ids, labels and splits.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import Engine, text

from .conversations import distinctive_terms, leaks
from .datasets import Dataset, QrelRecord, QueryRecord, SplitRecord, write_dataset

COPY_LIMIT = 8
_WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


@dataclass(frozen=True)
class ChunkSample:
    chunk_id: str
    paper_id: str
    title: str
    venue: str
    year: int
    section: str
    text: str


def _words(value: str) -> list[str]:
    return _WORD.findall(value.lower())


def longest_copied_run(question: str, passage: str) -> int:
    """The longest run of consecutive words the question shares with the passage."""

    left, right = _words(question), _words(passage)
    best = 0
    previous = [0] * (len(right) + 1)
    for word in left:
        current = [0] * (len(right) + 1)
        for j, other in enumerate(right, start=1):
            if word == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def acceptable(question: str, sample: ChunkSample) -> str | None:
    """None when the question may be used; otherwise why not."""

    if not question.strip():
        return "empty"
    if leaks(question, distinctive_terms(sample.title, "", ""), sample.title):
        return "names_paper"
    if longest_copied_run(question, sample.text) >= COPY_LIMIT:
        return "copies_passage"
    return None


def assign(
    samples: Sequence[ChunkSample], *, development: int, validation: int, test: int, seed: int
) -> dict[str, str]:
    """query id (the chunk id) → split, papers shuffled once by seed."""

    ordered = sorted(samples, key=lambda sample: sample.paper_id)
    random.Random(seed).shuffle(ordered)
    sizes = (("development", development), ("validation", validation), ("test", test))
    splits: dict[str, str] = {}
    start = 0
    for name, size in sizes:
        for sample in ordered[start : start + size]:
            splits[sample.chunk_id] = name
        start += size
    return splits


# Body chunks only; a few bibliographies are parsed as body under a "References" heading.
_SAMPLE_SQL = """
with candidates as (
    select c.id::text as chunk_id, p.id::text as paper_id, p.title, v.name as venue,
           p.publication_year as year, c.section_path as section, c.text,
           row_number() over (partition by p.id order by md5(c.id::text || :seed)) as in_paper
    from chunks c
    join paper_versions pv on pv.id = c.paper_version_id and pv.parse_status = 'parsed'
    join papers p on p.id = pv.paper_id and p.merged_into is null
    join venues v on v.id = p.venue_id
    where c.kind = 'body' and c.token_count >= :min_tokens
      and c.section_path !~* '\\m(references|bibliography)\\M'
), one_per_paper as (
    select *, row_number() over (
        partition by venue, year order by md5(paper_id || :seed)
    ) as in_venue_year
    from candidates where in_paper = 1
)
select chunk_id, paper_id, title, venue, year, section, text
from one_per_paper where in_venue_year <= :per_venue_year
order by venue, year, chunk_id
"""


# One abstract per findable paper: a paper with no chunk cannot be returned by search.
_ABSTRACT_SQL = """
with candidates as (
    select 't-' || p.id::text as chunk_id, p.id::text as paper_id, p.title, v.name as venue,
           p.publication_year as year, 'Abstract' as section, p.abstract as text
    from papers p
    join venues v on v.id = p.venue_id
    where p.merged_into is null and p.abstract is not null
      and length(p.abstract) >= :min_chars
      and not (p.id::text = any(cast(:exclude as text[])))
      and exists (
          select 1 from paper_versions pv join chunks c on c.paper_version_id = pv.id
          where pv.paper_id = p.id and pv.parse_status = 'parsed'
      )
), ranked as (
    select *, row_number() over (
        partition by venue, year order by md5(paper_id || :seed)
    ) as in_venue_year
    from candidates
)
select chunk_id, paper_id, title, venue, year, section, text
from ranked where in_venue_year <= :per_venue_year
order by venue, year, chunk_id
"""


def sample_abstracts(
    engine: Engine,
    *,
    per_venue_year: int,
    seed: int,
    exclude: Sequence[str] = (),
    min_chars: int = 1,
) -> list[ChunkSample]:
    """One abstract per paper search can return, ``per_venue_year`` papers per venue-year."""

    with engine.connect() as connection:
        rows = connection.execute(
            text(_ABSTRACT_SQL),
            {
                "seed": str(seed),
                "per_venue_year": per_venue_year,
                "exclude": list(exclude),
                "min_chars": min_chars,
            },
        ).mappings()
        return [
            ChunkSample(**{key: row[key] for key in ChunkSample.__dataclass_fields__})
            for row in rows
        ]


def sample_chunks(
    engine: Engine, *, per_venue_year: int, seed: int, min_tokens: int = 200
) -> list[ChunkSample]:
    """At most one body passage per paper and ``per_venue_year`` papers per venue-year."""

    with engine.connect() as connection:
        rows = connection.execute(
            text(_SAMPLE_SQL),
            {"seed": str(seed), "min_tokens": min_tokens, "per_venue_year": per_venue_year},
        ).mappings()
        return [
            ChunkSample(**{key: row[key] for key in ChunkSample.__dataclass_fields__})
            for row in rows
        ]


# --- Writing the questions (spends: one call per sample, one retry at most) --------------

PROMPT_PATH = "prompts/eval/section-question-v1.yaml"
TEXTS = "benchmarks/sections-v1/queries.jsonl"
FIXTURE = "data/fixtures/retrieval-sections"
QUERY_SET = "sections_v1"
RETRY = "That question was rejected ({reason}). Write a different one that follows the rules."
QUESTION_MODEL = "openai-gpt-6-luna"
QUESTION_TIMEOUT_S = 60.0
SEED = 42
PER_VENUE_YEAR = 15
MIN_TOKENS = 200
LIMIT = 600
SPLIT_SIZES = (200, 200, 200)


@dataclass(frozen=True)
class QuerySetSpec:
    """What one generated query set is: its prompt, where it is stored, and its sizes."""

    run_prefix: str
    prompt_path: str
    texts: str
    fixture: str
    query_set: str
    workflow: str
    specificity: int
    per_venue_year: int
    limit: int
    split_sizes: tuple[int, int, int]


SECTIONS = QuerySetSpec(
    run_prefix="section-queries",
    prompt_path=PROMPT_PATH,
    texts=TEXTS,
    fixture=FIXTURE,
    query_set=QUERY_SET,
    workflow="grounded_qa",
    specificity=1,
    per_venue_year=PER_VENUE_YEAR,
    limit=LIMIT,
    split_sizes=SPLIT_SIZES,
)
TOPICS = QuerySetSpec(
    run_prefix="topic-queries",
    prompt_path="prompts/eval/topic-question-v1.yaml",
    texts="benchmarks/topics-v1/queries.jsonl",
    fixture="data/fixtures/retrieval-topics",
    query_set="topics_v1",
    workflow="topic_search",
    specificity=0,
    per_venue_year=30,
    limit=1200,
    split_sizes=(400, 400, 400),
)


def load_question_prompt(path: str | Path) -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {"name": str(raw["name"]), "system": str(raw["system"]), "user": str(raw["user"])}


def question_messages(prompt: Mapping[str, str], sample: ChunkSample) -> list[dict[str, str]]:
    values = {"section": sample.section, "passage": sample.text}
    user = re.sub(r"\{(section|passage)\}", lambda match: values[match.group(1)], prompt["user"])
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]


def _question(reply: str) -> str | None:
    """The question in a reply: ``{"question": ...}``, or a bare one-line question.

    The model, asked for JSON, answers with the bare question about half the time (577 of
    1,095 replies in the first build); a single line that is not JSON is taken as the
    question itself. JSON without a ``question`` string, or several lines, is unparsable.
    """

    from ..models.generation import loads_lenient

    body = reply.strip()
    try:
        parsed = loads_lenient(body)
    except json.JSONDecodeError:
        if not body or "\n" in body or body.startswith(("{", "[")):
            return None
        return " ".join(body.strip("\"'“”` ").split())
    if isinstance(parsed, str) and "\n" not in parsed.strip():
        return " ".join(parsed.split()) or None  # a quoted bare question
    if not isinstance(parsed, dict) or not isinstance(parsed.get("question"), str):
        return None
    return " ".join(parsed["question"].split())


@dataclass
class BuiltQueries:
    kept: list[tuple[ChunkSample, str]] = field(default_factory=list)
    rejected: Counter[str] = field(default_factory=Counter)
    dropped: int = 0
    calls: int = 0
    stopped: str | None = None


def build_section_queries(
    samples: Sequence[ChunkSample],
    complete: Callable[[list[dict[str, str]]], Any],
    prompt: Mapping[str, str],
    *,
    spent: Callable[[], float] | None = None,
    max_spend_usd: float | None = None,
) -> BuiltQueries:
    """One question per sample; a rejected one is retried once with the reason, then dropped.

    ``complete(messages) -> ChatResult``. ``rejected`` counts every rejection, first tries and
    retries alike, by reason; ``dropped`` counts samples left without a question. The build
    stops, keeping what it has, once ``spent()`` passes ``max_spend_usd``.
    """

    built = BuiltQueries()
    for sample in samples:
        messages = question_messages(prompt, sample)
        accepted: str | None = None
        for attempt in range(2):
            reply = complete(messages).text
            built.calls += 1
            question = _question(reply)
            reason = "unparsable" if question is None else acceptable(question, sample)
            if reason is None:
                accepted = question
                break
            built.rejected[reason] += 1
            if attempt == 0:
                messages = [
                    *messages,
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": RETRY.format(reason=reason)},
                ]
        if accepted is None:
            built.dropped += 1
        else:
            built.kept.append((sample, accepted))
        if spent is not None and max_spend_usd is not None and spent() > max_spend_usd:
            built.stopped = f"budget: ${spent():.4f} > ${max_spend_usd:.4f}"
            break
    return built


def split_sizes(kept: int, planned: tuple[int, int, int] = SPLIT_SIZES) -> tuple[int, int, int]:
    """The planned sizes (S1: 200 / 200 / 200), or the same thirds of fewer kept questions."""

    if kept >= sum(planned):
        return planned
    third = kept // 3
    return third, third, kept - 2 * third


def write_section_queries(
    kept: Sequence[tuple[ChunkSample, str]],
    splits: Mapping[str, str],
    fixture: Path,
    texts: Path,
    *,
    source: str,
    prompt_name: str = "section-question-v1",
    spec: QuerySetSpec = SECTIONS,
) -> None:
    """ids, labels and splits to ``fixture`` (no text); ``{query_id, query}`` lines to ``texts``."""

    rows = sorted(
        ((sample, question) for sample, question in kept if sample.chunk_id in splits),
        key=lambda row: row[0].chunk_id,
    )
    dataset = Dataset(
        queries=tuple(
            QueryRecord(
                query_id=sample.chunk_id,
                query=question,
                query_set=spec.query_set,
                workflow=spec.workflow,
                specificity=spec.specificity,
                quality=2,
                in_domain=True,
                source=source,
                author=f"generated by {QUESTION_MODEL}, {prompt_name}",
            )
            for sample, question in rows
        ),
        qrels=tuple(
            QrelRecord(
                query_id=sample.chunk_id,
                corpusid=sample.paper_id,
                grade=1,
                paper_id=sample.paper_id,
            )
            for sample, _ in rows
        ),
        splits=tuple(
            SplitRecord(
                query_id=sample.chunk_id, family_id=sample.paper_id, split=splits[sample.chunk_id]
            )
            for sample, _ in rows
        ),
    )
    write_dataset(dataset, fixture, include_query_text=False)
    texts.parent.mkdir(parents=True, exist_ok=True)
    texts.write_text(
        "".join(
            json.dumps({"query_id": sample.chunk_id, "query": question}) + "\n"
            for sample, question in rows
        ),
        "utf-8",
    )


class SectionQueryError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code


def run_section_queries(
    *,
    database_url: str,
    max_spend_usd: float,
    per_venue_year: int = PER_VENUE_YEAR,
    out_fixture: Path = Path(FIXTURE),
    prompt_path: str | Path = PROMPT_PATH,
) -> dict[str, Any]:
    """``eval section-queries`` (S1): one question per sampled body passage."""

    from ..db.session import make_engine

    corpus = make_engine(database_url)
    samples = sample_chunks(
        corpus, per_venue_year=per_venue_year, seed=SEED, min_tokens=MIN_TOKENS
    )[: SECTIONS.limit]
    if not samples:
        raise SectionQueryError("no_samples", "the corpus has no body chunk to sample")
    return _run_generated(
        SECTIONS, corpus, samples, max_spend_usd, out_fixture, load_question_prompt(prompt_path)
    )


def run_topic_queries(
    *,
    database_url: str,
    max_spend_usd: float,
    per_venue_year: int = TOPICS.per_venue_year,
    out_fixture: Path = Path(TOPICS.fixture),
    prompt_path: str | Path = TOPICS.prompt_path,
    exclude_fixture: Path = Path(SECTIONS.fixture),
) -> dict[str, Any]:
    """``eval topic-queries`` (S3): one topic question per sampled abstract, S1's papers out."""

    from ..db.session import make_engine
    from .datasets import read_dataset

    excluded = sorted({str(q.paper_id) for q in read_dataset(exclude_fixture).qrels if q.paper_id})
    corpus = make_engine(database_url)
    samples = sample_abstracts(corpus, per_venue_year=per_venue_year, seed=SEED, exclude=excluded)[
        : TOPICS.limit
    ]
    if not samples:
        raise SectionQueryError("no_samples", "the corpus has no findable abstract to sample")
    report = _run_generated(
        TOPICS, corpus, samples, max_spend_usd, out_fixture, load_question_prompt(prompt_path)
    )
    return report | {"excluded_papers": len(excluded)}


def _run_generated(
    spec: QuerySetSpec,
    corpus: Engine,
    samples: Sequence[ChunkSample],
    max_spend_usd: float,
    out_fixture: Path,
    prompt: Mapping[str, str],
) -> dict[str, Any]:
    """Write and check one question per sample, then store the set as ``spec`` says.

    Refused when its worst case — every sample asked twice at the full output budget —
    exceeds ``max_spend_usd``; stopped once the run's ledger spend does.
    """

    from datetime import UTC, datetime

    from ..config import Settings
    from ..db.session import make_engine
    from ..models.llm import ResponseCache, build_client
    from ..models.spend import SpendLedger, estimate_usd, load_llm_config

    with corpus.connect() as connection:
        release = connection.execute(
            text("select corpus_release_id from active_release")
        ).scalar_one_or_none()
    model = load_llm_config("configs/llm.yaml").models[QUESTION_MODEL]
    retry_chars = model.max_output_tokens * 4 + len(RETRY) + 40
    worst = sum(
        estimate_usd(model.price, len(json.dumps(messages)), model.max_output_tokens)
        + estimate_usd(
            model.price, len(json.dumps(messages)) + retry_chars, model.max_output_tokens
        )
        for messages in (question_messages(prompt, sample) for sample in samples)
    )
    if worst > max_spend_usd:
        raise SectionQueryError(
            "run_budget_exceeded", f"worst case ${worst:.4f} > budget ${max_spend_usd:.4f}"
        )
    settings = Settings()  # type: ignore[call-arg]
    data_dir = Path(settings.data_dir)
    ledger = SpendLedger(make_engine(settings.database_url), settings.llm_daily_spend_cap_usd)
    client = build_client(
        QUESTION_MODEL, settings, ledger, cache=ResponseCache(data_dir / "cache" / "llm")
    )
    run_id = f"{spec.run_prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    def complete(messages: list[dict[str, str]]) -> Any:
        return client.complete(
            messages, timeout=QUESTION_TIMEOUT_S, purpose="evaluation", run_id=run_id
        )

    try:
        built = build_section_queries(
            samples,
            complete,
            prompt,
            spent=lambda: ledger.run_spend(run_id),
            max_spend_usd=max_spend_usd,
        )
    finally:
        client.close()
    development, validation, test = split_sizes(len(built.kept), spec.split_sizes)
    splits = assign(
        [sample for sample, _ in built.kept],
        development=development,
        validation=validation,
        test=test,
        seed=SEED,
    )
    texts = data_dir / spec.texts
    write_section_queries(
        built.kept,
        splits,
        out_fixture,
        texts,
        source=f"copilot_v2@{release}",
        prompt_name=prompt["name"],
        spec=spec,
    )
    return {
        "run_id": run_id,
        "samples": len(samples),
        "venue_years": len({(s.venue, s.year) for s in samples}),
        "kept": len(built.kept),
        "dropped": built.dropped,
        "rejected": dict(built.rejected),
        "calls": built.calls,
        "splits": dict(Counter(splits.values())),
        "stopped": built.stopped,
        "fixture": str(out_fixture),
        "texts": str(texts),
        "worst": worst,
        "spent_usd": ledger.run_spend(run_id),
    }
