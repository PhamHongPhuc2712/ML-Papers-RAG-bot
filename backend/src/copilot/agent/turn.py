"""One research-agent turn: the graph chooses passages, the existing generator answers.

The chosen passages are renumbered E1..En before the generator sees them, the labels the
grounded-claims prompt and its validation use everywhere else; `chosen_ids` keeps the
agent's own labels (S1E2, H1) for the trace.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..chat.retriever import CorpusRetriever
from ..chat.turns import HistoryItem
from ..contracts import Evidence, Reference
from ..evidence.retrieve import ChunkRecord, build_context
from ..models.generation import GenerationOutcome, RewritePrompt
from ..models.llm import LlmError
from .check import LedgerChecker, SelfCheck, self_check
from .graph import ToolCallingModel, build_agent_graph
from .plan import PlanError, Planner
from .tools import AgentLimits, AgentTools


@dataclass(frozen=True)
class AgentPrompt:
    name: str
    sha256: str
    system: str
    user: str


def load_agent_prompt(path: str | Path) -> AgentPrompt:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    return AgentPrompt(
        name=str(data["name"]),
        sha256=hashlib.sha256(raw).hexdigest(),
        system=str(data["system"]),
        user=str(data["user"]),
    )


@dataclass
class AgentResult:
    standalone: str
    # What the answer was written from, renumbered E1..En as the generator saw it.
    evidence: list[Evidence]
    chosen_ids: list[str]
    seen: list[Evidence]
    paper_ids: list[str]
    references: list[Reference]
    release_id: str | None
    outcome: GenerationOutcome
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    searches: int = 0
    queries: list[str] = field(default_factory=list)
    sections: int = 0
    model_calls: int = 0
    stop: str = "answered"
    # The optional planning step's sub-questions; empty when off, single-part or failed.
    plan: tuple[str, ...] = ()
    # The optional self-check's record; None when off.
    check: SelfCheck | None = None


@dataclass(frozen=True)
class AgentUpdate:
    """One step of a streamed turn (agent in the chat, 2026-10-30)."""

    stage: Literal["search", "retrieve", "generate"]
    query: str | None = None
    queries: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    result: AgentResult | None = None


@dataclass
class AgentTurn:
    model: ToolCallingModel
    retriever: CorpusRetriever
    section_chunks: Callable[[Evidence], list[ChunkRecord]]
    generator: Any
    prompt: AgentPrompt
    limits: AgentLimits
    max_chars: int
    token_budget: int
    count_tokens: Callable[[str], int]
    # What the answer is written from: the passages the agent named ("chosen"), or those first
    # and then every other passage it saw, in the order found ("seen"); either way within
    # the token budget, and the generator decides whether they suffice.
    answer_from: str = "chosen"
    # On a question with no conversation before it: search the question as written first, and
    # answer the question as written. A standalone question is its own best query (ORB
    # development, 2026-10-25); the agent still rewrites follow-ups and may search again.
    verbatim_first: bool = False
    # The optional planning step (2026-10-28): with a planner, one call splits the question into
    # at most `max_subquestions` parts shown to the agent as parts to cover.
    planner: Planner | None = None
    max_subquestions: int = 3
    # The optional self-check (2026-10-29): with a checker, every claim is checked against its
    # cited passages after the answer is written, and the answer rewritten once if any fails.
    checker: LedgerChecker | None = None
    rewrite_prompt: RewritePrompt | None = None
    clock: Callable[[], float] = time.monotonic

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "agent_prompt": f"{self.prompt.name}:{self.prompt.sha256[:12]}",
            "limits": {
                "max_searches": self.limits.max_searches,
                "max_model_calls": self.limits.max_model_calls,
                "max_sections": self.limits.max_sections,
            },
            "model": str(getattr(self.model, "identity", "unknown")),
            "generator": str(getattr(self.generator, "identity", "unknown")),
            "answer_from": self.answer_from,
            "verbatim_first": self.verbatim_first,
            "plan": None
            if self.planner is None
            else {"prompt": self.planner.identity, "max_subquestions": self.max_subquestions},
            "check": None
            if self.checker is None
            else {
                "checker": self.checker.identity,
                "rewrite": None
                if self.rewrite_prompt is None
                else f"{self.rewrite_prompt.name}:{self.rewrite_prompt.sha256[:12]}",
            },
        }

    def run(
        self,
        message: str,
        history: Sequence[HistoryItem],
        carried: Sequence[Evidence],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> AgentResult:
        result: AgentResult | None = None
        for update in self.iter_steps(
            message, history, carried, request_id=request_id, run_id=run_id
        ):
            result = update.result or result
        assert result is not None
        return result

    def iter_steps(
        self,
        message: str,
        history: Sequence[HistoryItem],
        carried: Sequence[Evidence],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> Iterator[AgentUpdate]:
        """The turn as it goes: ``search`` once per search (its query), ``retrieve`` once the
        passages to answer from are chosen, ``generate`` with the finished result."""

        tools = AgentTools(
            self.retriever, self.section_chunks, limits=self.limits, max_chars=self.max_chars
        )
        system = self.prompt.system.replace("{max_searches}", str(self.limits.max_searches))
        messages: list[BaseMessage] = [SystemMessage(system)]
        for item in history:
            messages.append(
                HumanMessage(item.text) if item.role == "user" else AIMessage(item.text)
            )
        user = self.prompt.user.replace("{carried}", tools.carry(carried))
        user = user.replace("{message}", message)
        started = self.clock()
        warnings: list[str] = []
        plan: tuple[str, ...] = ()
        plan_cost = 0.0
        if self.planner is not None:
            try:
                made = self.planner.plan(
                    message,
                    max_subquestions=self.max_subquestions,
                    request_id=request_id,
                    run_id=run_id,
                )
                plan, plan_cost = made.subquestions, made.cost_usd
            except (LlmError, PlanError):
                warnings.append("agent_plan_failed")
        plan_seconds = self.clock() - started
        if len(plan) > 1:
            parts = "\n".join(f"{i}. {part}" for i, part in enumerate(plan, 1))
            user += (
                f"\n\nParts to cover:\n{parts}\nSearch for each part the passages you have do"
                " not already answer."
            )
        started = self.clock()
        verbatim = self.verbatim_first and not history
        if verbatim:
            found = tools.search_papers(message)
            user += f"\n\nA first search was run with the question as written:\n{found}"
        messages.append(HumanMessage(user))
        graph = build_agent_graph(
            self.model, tools, self.limits, request_id=request_id, run_id=run_id
        )
        reported = 0
        for query in tools.queries[reported:]:
            yield AgentUpdate("search", query=query)
        reported = len(tools.queries)
        state: dict[str, Any] = {}
        for state in graph.stream(  # noqa: B007 - the last value is the final state
            {"messages": messages, "model_calls": 0, "cost_usd": 0.0, "final": None},
            stream_mode="values",
        ):
            for query in tools.queries[reported:]:
                yield AgentUpdate("search", query=query)
            reported = len(tools.queries)
        agent_seconds = self.clock() - started
        warnings += tools.warnings
        final = state["final"]
        if final is not None:
            question = message if verbatim else final["question"] or message
            if any(label not in tools.seen for label in final["evidence_ids"]):
                warnings.append("agent_unknown_evidence")
            chosen_ids = list(
                dict.fromkeys(label for label in final["evidence_ids"] if label in tools.seen)
            )
            if self.answer_from == "seen":
                chosen_ids += [label for label in tools.seen if label not in chosen_ids]
            stop = "answered"
        else:
            question = message
            chosen_ids = list(tools.seen)
            capped = state["model_calls"] >= self.limits.max_model_calls
            stop = "model_call_limit" if capped else "no_answer_call"
            warnings.append(f"agent_{stop}")
        kept, _ = build_context(
            [tools.seen[label] for label in chosen_ids],
            token_budget=self.token_budget,
            count_tokens=self.count_tokens,
        )
        kept_ids = {id(item) for item in kept}
        chosen_ids = [label for label in chosen_ids if id(tools.seen[label]) in kept_ids]
        evidence = [item.model_copy(update={"id": f"E{i}"}) for i, item in enumerate(kept, 1)]
        yield AgentUpdate(
            "retrieve",
            queries=list(tools.queries),
            references=list(tools.references),
            evidence=list(evidence),
            warnings=list(warnings),
        )
        started = self.clock()
        outcome: GenerationOutcome = self.generator.generate(
            question, evidence, request_id=request_id, run_id=run_id
        )
        if outcome.reason not in (None, "no_evidence"):
            warnings.append("answer_failed_validation")
        generate_seconds = self.clock() - started
        check: SelfCheck | None = None
        if self.checker is not None:
            if self.rewrite_prompt is None:
                raise ValueError("a checker needs a rewrite prompt")
            started = self.clock()
            outcome, check = self_check(
                question,
                evidence,
                outcome,
                checker=self.checker,
                generator=self.generator,
                rewrite=self.rewrite_prompt,
                request_id=request_id,
                run_id=run_id,
            )
            if check.result == "failed":
                warnings.append("agent_check_failed")
        result = AgentResult(
            standalone=question,
            evidence=evidence,
            chosen_ids=chosen_ids,
            seen=list(tools.seen.values()),
            paper_ids=list(tools.paper_ids),
            references=list(tools.references),
            release_id=tools.release_id,
            outcome=outcome,
            warnings=warnings,
            timings={
                **({"plan": plan_seconds} if self.planner is not None else {}),
                "agent": agent_seconds,
                "generate": generate_seconds,
                **({"check": self.clock() - started} if check is not None else {}),
            },
            cost_usd=plan_cost
            + float(state["cost_usd"])
            + outcome.cost_usd
            + (check.cost_usd if check is not None else 0.0),
            searches=tools.searches,
            queries=list(tools.queries),
            sections=tools.sections,
            model_calls=int(state["model_calls"]),
            stop=stop,
            plan=plan,
            check=check,
        )
        yield AgentUpdate("generate", result=result)
