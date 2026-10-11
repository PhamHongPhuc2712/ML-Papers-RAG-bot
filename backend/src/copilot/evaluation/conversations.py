"""Follow-up conversations from ORB questions that no answering split has used.

Pairs are chosen per gold paper and split by paper, so no paper is in development and frozen
both. A follow-up is B rewritten to depend on turn 1; it is rejected if it still names what
only the paper names — an acronym, a hyphenated or digit-bearing term from the title or from
B, or two consecutive content words of the title — because then it would not need the
conversation at all. Only tokens B shares with A count from B: B's own subject (a metric, a
dataset) may stay. Rewrites are ORB-derived text: they live under DATA_DIR, and the
repository keeps only ids, splits and each rewrite's sha256.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]*[A-Za-z0-9]|[A-Za-z0-9]")
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
    "without",
    "towards",
    "toward",
    "using",
    "through",
}


def pair_by_paper(questions: Sequence[tuple[str, str]], *, seed: int) -> list[tuple[str, str, str]]:
    by_paper: dict[str, list[str]] = defaultdict(list)
    for paper, query_id in questions:
        by_paper[paper].append(query_id)
    rng = random.Random(seed)
    pairs = []
    for paper in sorted(by_paper):
        ids = sorted(by_paper[paper])
        if len(ids) < 2:
            continue
        first, second = rng.sample(ids, 2)
        pairs.append((paper, first, second))
    return pairs


def split_by_paper(
    pairs: Sequence[tuple[str, str, str]], *, development: int, frozen: int, seed: int
) -> dict[str, list[tuple[str, str, str]]]:
    ordered = sorted(pairs)
    random.Random(seed).shuffle(ordered)
    return {
        "development": ordered[:development],
        "frozen": ordered[development : development + frozen],
    }


def _distinctive(token: str) -> bool:
    return (
        sum(char.isupper() for char in token) >= 2
        or any(char.isdigit() for char in token)
        or "-" in token
    )


def _content(title: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(title) if t.lower() not in _STOP and len(t) > 2]


def _marked(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text) if _distinctive(token)}


def distinctive_terms(title: str, first: str, second: str) -> set[str]:
    """What names the paper: the title's distinctive tokens, and those B shares with A."""

    shared = _marked(second) & {token.lower() for token in _TOKEN.findall(first)}
    return _marked(title) | shared


def leaks(rewrite: str, terms: set[str], title: str) -> bool:
    words = [t.lower() for t in _TOKEN.findall(rewrite)]
    if terms & set(words):
        return True
    content = _content(title)
    bigrams = {(a, b) for a, b in zip(content, content[1:], strict=False)}
    return any((a, b) in bigrams for a, b in zip(words, words[1:], strict=False))


# --- Building the conversations (spends: one rewrite per kept pair, one retry at most) -----

REWRITE_PROMPT = "prompts/chat/followup-rewrite-v2.yaml"
TEXTS = "eval/chat/conversations-v2.jsonl"
FIXTURE = "data/fixtures/chat/conversations.jsonl"
RETRY = "Your follow-up still names {term}; refer to it instead."
KEEP = (
    "Your follow-up drops {term}, which the previous question does not mention;"
    " keep it so the follow-up still says what it asks about."
)
DEFAULT_DEVELOPMENT = 100
DEFAULT_FROZEN = 200


def leaked_term(rewrite: str, terms: set[str], title: str) -> str | None:
    words: list[str] = [str(t).lower() for t in _TOKEN.findall(rewrite)]

    words = [t.lower() for t in _TOKEN.findall(rewrite)]
    for word in words:
        if word in terms:
            return word
    content = _content(title)
    bigrams = {(a, b) for a, b in zip(content, content[1:], strict=False)}
    for a, b in zip(words, words[1:], strict=False):
        if (a, b) in bigrams:
            return f"{a} {b}"
    return None


def dropped_subject(rewrite: str, first: str, second: str) -> str | None:
    """A distinctive term of B that A never uses and the rewrite lost, or None.

    Such a term is B's own subject: without it the follow-up cannot say what it asks about,
    and the history cannot supply it.
    """

    # One character (a bare digit) is never a subject on its own.
    own = sorted(
        t for t in _marked(second) - {t.lower() for t in _TOKEN.findall(first)} if len(t) > 1
    )
    kept = {t.lower() for t in _TOKEN.findall(rewrite)}
    for term in own:
        if term not in kept:
            return term
    return None


def _one_line(text: str) -> str:
    lines = text.strip().splitlines()
    return " ".join(lines[0].split()).strip("\"'“”` ") if lines else ""


def _fill(template: str, values: dict[str, str]) -> str:
    return re.sub(r"\{(first|second)\}", lambda match: values[match.group(1)], template)


def rewrite_followup(
    complete: Any, prompt: Mapping[str, str], title: str, first: str, second: str
) -> tuple[str | None, int]:
    """B rewritten to lean on A, or None when after one retry it still names what A names or
    drops B's own subject.

    ``complete(messages) -> ChatResult``. Returns the rewrite and how many calls it took.
    """

    terms = distinctive_terms(title, first, second)
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": _fill(prompt["user"], {"first": first, "second": second})},
    ]
    reply = _one_line(complete(messages).text)
    problem = _problem(reply, terms, title, first, second)
    if problem is None:
        return reply, 1
    retry = [
        *messages,
        {"role": "assistant", "content": reply},
        {"role": "user", "content": problem},
    ]
    second_reply = _one_line(complete(retry).text)
    if _problem(second_reply, terms, title, first, second) is None:
        return second_reply, 2
    return None, 2


def _problem(reply: str, terms: set[str], title: str, first: str, second: str) -> str | None:
    """The retry message a rewrite earns, or None when it is usable."""

    if not reply:
        return RETRY.format(term="the paper")
    leaked = leaked_term(reply, terms, title)
    if leaked is not None:
        return RETRY.format(term=leaked)
    dropped = dropped_subject(reply, first, second)
    if dropped is not None:
        return KEEP.format(term=dropped)
    return None


def build_conversations(
    *,
    data_dir: Path,
    dataset: Path,
    paper_ids: Mapping[str, str],
    complete: Any,
    out_fixture: Path,
    prompt_path: str | Path = REWRITE_PROMPT,
    seed: int = 42,
    development: int = DEFAULT_DEVELOPMENT,
    frozen: int = DEFAULT_FROZEN,
    spent: Any = None,
    max_spend_usd: float | None = None,
) -> dict[str, Any]:
    """Pair ``retrieval_only`` questions by gold paper, split, rewrite, keep what does not leak.

    ``paper_ids`` maps an ORB document id to the loaded corpus's paper id; a question whose
    gold paper is not loaded is left out. Texts go to ``data_dir/TEXTS``; the repository's
    fixture gets ids, the split and each rewrite's sha256. ``spent()`` is the run's ledger
    spend so far; the build stops, keeping what it has, once it passes ``max_spend_usd``.
    """

    from .orb import OrbPaths, load_orb, read_qa_assignment, text_slice

    raw = yaml.safe_load(Path(prompt_path).read_text(encoding="utf-8")) or {}
    prompt = {"system": str(raw["system"]), "user": str(raw["user"])}
    paths = OrbPaths(data_dir / "benchmarks" / "orb")
    assignment = read_qa_assignment(dataset)
    queries = {
        q.query_id: q
        for q in text_slice(load_orb(paths))
        if assignment.get(q.query_id) == "retrieval_only" and q.doc_id in paper_ids
    }
    questions = [(q.doc_id, q.query_id) for q in queries.values()]
    splits = split_by_paper(
        pair_by_paper(questions, seed=seed), development=development, frozen=frozen, seed=seed
    )
    titles: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    calls = 0
    stopped: str | None = None
    for split in ("development", "frozen"):
        counts[split] = {"pairs": len(splits[split]), "kept": 0, "dropped": 0}
        for doc_id, first_id, second_id in splits[split]:
            if stopped is not None:
                break
            if doc_id not in titles:
                corpus = json.loads(paths.corpus_file(doc_id).read_text(encoding="utf-8"))
                titles[doc_id] = " ".join(str(corpus.get("title") or "").split())
            first, second = queries[first_id], queries[second_id]
            followup, used = rewrite_followup(
                complete, prompt, titles[doc_id], first.query, second.query
            )
            calls += used
            if followup is None:
                counts[split]["dropped"] += 1
            else:
                counts[split]["kept"] += 1
                kept.append(
                    {
                        "conversation_id": f"{first_id}+{second_id}",
                        "split": split,
                        "paper_id": paper_ids[doc_id],
                        "doc_id": doc_id,
                        "first_id": first_id,
                        "first": first.query,
                        "second_id": second_id,
                        "second": second.query,
                        "followup": followup,
                        "reference": second.answer,
                    }
                )
            if spent is not None and max_spend_usd is not None and spent() > max_spend_usd:
                stopped = f"budget: ${spent():.4f} > ${max_spend_usd:.4f}"
    texts = data_dir / TEXTS
    texts.parent.mkdir(parents=True, exist_ok=True)
    texts.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in kept), "utf-8")
    out_fixture.parent.mkdir(parents=True, exist_ok=True)
    out_fixture.write_text(
        "".join(
            json.dumps(
                {
                    "conversation_id": row["conversation_id"],
                    "split": row["split"],
                    "paper_id": row["paper_id"],
                    "first_id": row["first_id"],
                    "second_id": row["second_id"],
                    "rewrite_sha256": hashlib.sha256(row["followup"].encode()).hexdigest(),
                },
                sort_keys=True,
            )
            + "\n"
            for row in kept
        ),
        "utf-8",
    )
    return {
        "texts": str(texts),
        "fixture": str(out_fixture),
        "counts": counts,
        "calls": calls,
        "stopped": stopped,
    }


def read_conversations(fixture: Path, texts: Path, split: str) -> list[dict[str, Any]]:
    """One split's conversations: the repository's ids joined to the texts under DATA_DIR.

    A text whose rewrite no longer matches the recorded sha256 is refused: the conversations
    scored are exactly the ones registered.
    """

    by_id = {
        row["conversation_id"]: row
        for row in (json.loads(line) for line in texts.read_text("utf-8").splitlines() if line)
    }
    out = []
    for line in fixture.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["split"] != split:
            continue
        row = by_id.get(entry["conversation_id"])
        if row is None:
            raise ValueError(f"conversation_text_missing:{entry['conversation_id']}")
        if hashlib.sha256(row["followup"].encode()).hexdigest() != entry["rewrite_sha256"]:
            raise ValueError(f"conversation_text_changed:{entry['conversation_id']}")
        out.append(row)
    return sorted(out, key=lambda row: row["conversation_id"])


REWRITE_MODEL = "openai-gpt-6-luna"
REWRITE_PROMPT_CHARS = 1_500
REWRITE_TIMEOUT_S = 60.0


class ConversationBuildError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code


def run_build_conversations(
    *,
    database_url: str,
    max_spend_usd: float,
    out_fixture: Path = Path(FIXTURE),
    dataset: Path = Path("data/fixtures/orb"),
    development: int = DEFAULT_DEVELOPMENT,
    frozen: int = DEFAULT_FROZEN,
) -> dict[str, Any]:
    """``eval chat-conversations``: the builder over the real ORB files, model and ledger.

    Refused when its worst case — every pair rewritten twice at the full output budget —
    exceeds ``max_spend_usd``; stopped once the run's ledger spend does.
    """

    from datetime import UTC, datetime

    from ..config import Settings
    from ..db.session import make_engine
    from ..models.llm import ResponseCache, build_client
    from ..models.spend import SpendLedger, estimate_usd, load_llm_config
    from .orb import OrbPaths, load_orb, paper_ids_for, text_slice

    model = load_llm_config("configs/llm.yaml").models[REWRITE_MODEL]
    worst = (
        (development + frozen)
        * 2
        * estimate_usd(model.price, REWRITE_PROMPT_CHARS, model.max_output_tokens)
    )
    if worst > max_spend_usd:
        raise ConversationBuildError(
            "run_budget_exceeded", f"worst case ${worst:.4f} > budget ${max_spend_usd:.4f}"
        )
    settings = Settings()  # type: ignore[call-arg]
    data_dir = Path(settings.data_dir)
    doc_ids = {q.doc_id for q in text_slice(load_orb(OrbPaths(data_dir / "benchmarks" / "orb")))}
    paper_ids = paper_ids_for(doc_ids, make_engine(database_url))
    ledger = SpendLedger(make_engine(settings.database_url), settings.llm_daily_spend_cap_usd)
    client = build_client(
        REWRITE_MODEL, settings, ledger, cache=ResponseCache(data_dir / "cache" / "llm")
    )
    run_id = f"chat-followups-build-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    def complete(messages: list[dict[str, str]]) -> Any:
        return client.complete(
            messages, timeout=REWRITE_TIMEOUT_S, purpose="evaluation", run_id=run_id
        )

    try:
        result = build_conversations(
            data_dir=data_dir,
            dataset=dataset,
            paper_ids=paper_ids,
            complete=complete,
            out_fixture=out_fixture,
            development=development,
            frozen=frozen,
            spent=lambda: ledger.run_spend(run_id),
            max_spend_usd=max_spend_usd,
        )
    finally:
        client.close()
    return {**result, "run_id": run_id, "spent_usd": ledger.run_spend(run_id), "worst": worst}
