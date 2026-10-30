"""One chat turn in three explicit steps: condense, retrieve, generate.

Served until 2026-10-30, when the research agent replaced it (``chat/agent.py``). It stays as the
evaluation harness's baseline: ``eval mtrag``'s ``pipeline`` system and ``eval chat``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from uuid import UUID

from ..contracts import Evidence, Reference
from ..models.generation import GenerationOutcome
from .condense import Condenser
from .retriever import CorpusRetriever
from .turns import HistoryItem


@dataclass
class TurnState:
    standalone: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    paper_ids: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    release_id: str | None = None
    outcome: GenerationOutcome | None = None
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0


@dataclass(frozen=True)
class TurnUpdate:
    stage: Literal["condense", "retrieve", "generate"]
    state: TurnState
    warnings: list[str] = field(default_factory=list)


def _snapshot(state: TurnState) -> TurnState:
    return replace(
        state,
        evidence=list(state.evidence),
        paper_ids=list(state.paper_ids),
        references=list(state.references),
        warnings=list(state.warnings),
        timings=dict(state.timings),
    )


@dataclass(frozen=True)
class TurnPipeline:
    condenser: Condenser
    retriever: CorpusRetriever
    generator: Any
    condense: bool
    identity: dict[str, Any] = field(default_factory=dict)
    history_turns: int = 3
    clock: Callable[[], float] = time.monotonic

    def iter_steps(
        self,
        message: str,
        history: list[HistoryItem],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> Iterator[TurnUpdate]:
        state = TurnState()
        started = self.clock()
        if self.condense:
            query, warnings, cost = self.condenser.standalone(
                history, message, request_id=request_id, run_id=run_id
            )
            state.timings["condense"] = self.clock() - started
        else:
            query, warnings, cost = message, [], 0.0
            state.timings["condense"] = 0.0
        state.standalone = query
        state.warnings.extend(warnings)
        state.cost_usd += cost
        yield TurnUpdate("condense", _snapshot(state), warnings)

        started = self.clock()
        evidence, hits = self.retriever.search_evidence(query)
        state.evidence = evidence
        state.paper_ids = list(hits.paper_ids)
        state.references = list(hits.references)
        state.release_id = hits.release_id
        state.warnings.extend(hits.warnings)
        state.timings["retrieve"] = self.clock() - started
        yield TurnUpdate("retrieve", _snapshot(state), list(hits.warnings))

        started = self.clock()
        outcome: GenerationOutcome = self.generator.generate(
            query, evidence, request_id=request_id, run_id=run_id
        )
        warnings = (
            ["answer_failed_validation"] if outcome.reason not in (None, "no_evidence") else []
        )
        state.outcome = outcome
        state.warnings.extend(warnings)
        state.cost_usd += outcome.cost_usd
        state.timings["generate"] = self.clock() - started
        yield TurnUpdate("generate", _snapshot(state), warnings)

    def run(
        self,
        message: str,
        history: list[HistoryItem],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> TurnState:
        result: TurnState | None = None
        for update in self.iter_steps(message, history, request_id=request_id, run_id=run_id):
            result = update.state
        assert result is not None and result.outcome is not None
        return result
