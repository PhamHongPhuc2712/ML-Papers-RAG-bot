"""One chat turn in three explicit steps: condense, retrieve, generate."""

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


def build_default_turn_pipeline(
    settings: Any, engine: Any, search: Any, config_path: str = "configs/chat.yaml"
) -> TurnPipeline:
    """Build the serving turn from the configured search and model adapters."""

    from ..evidence.retrieve import EvidenceRetriever, load_evidence_config
    from ..models.generation import (
        ChatGenerator,
        FakeGenerator,
        StructuredGenerator,
        load_generation_prompt,
    )
    from ..models.lc_chat import build_ledger_chat_model
    from ..models.llm import LlmError, ResponseCache, build_client
    from ..models.spend import SpendLedger
    from ..search.api import default_search_service
    from .condense import load_condense_prompt
    from .config import chat_search_config, load_chat_config
    from .retriever import ServicePaperSearch

    config = load_chat_config(config_path)
    client, models, runner = search.serving()
    if models.embedder is None:
        raise LlmError("models_unavailable")
    service = default_search_service(
        settings,
        engine,
        client,
        models=models,
        config=chat_search_config(config),
        runner=runner,
        listwise=False,
    )
    evidence_config = load_evidence_config(config.evidence)
    evidence = EvidenceRetriever(
        engine=engine,
        client=client,
        embedder=models.embedder,
        reranker=models.reranker,
        config=evidence_config,
        data_dir=settings.data_dir,
        max_tokens=models.max_tokens,
    )
    retriever = CorpusRetriever(
        search=ServicePaperSearch(
            service, search.active_release, config.papers, references=config.references
        ),
        evidence=evidence.retrieve,
    )
    prompt = load_generation_prompt(config.generate_prompt)
    condense_prompt = load_condense_prompt(config.condense_prompt)
    mock = settings.model_mode == "mock" or settings.mock_mode
    generator: Any
    if mock:
        condenser = Condenser(None, condense_prompt, max_chars=config.history_max_chars)
        generator = FakeGenerator(prompt)
    else:
        ledger = SpendLedger(engine, settings.llm_daily_spend_cap_usd)
        cache = ResponseCache(settings.data_dir / "cache" / "llm")
        condenser = Condenser(
            build_ledger_chat_model(
                config.model, settings, ledger, timeout=config.condense_timeout, cache=cache
            ),
            condense_prompt,
            max_chars=config.history_max_chars,
        )
        if config.generator == "structured":
            generator = StructuredGenerator(
                build_ledger_chat_model(
                    config.model, settings, ledger, timeout=config.generate_timeout, cache=cache
                ),
                prompt,
                purpose="chat",
            )
        else:
            generator = ChatGenerator(
                build_client(config.model, settings, ledger, cache=cache),
                prompt,
                timeout=config.generate_timeout,
                purpose="chat",
            )
    identity = {
        "chat_config_sha256": config.sha256,
        "generator": generator.identity,
        "condense_prompt": f"{condense_prompt.name}:{condense_prompt.sha256[:12]}",
        "search": service.identity("hybrid_rerank"),
        "evidence": evidence_config.identity(),
    }
    return TurnPipeline(
        condenser=condenser,
        retriever=retriever,
        generator=generator,
        condense=config.condense,
        identity=identity,
        history_turns=config.history_turns,
    )
