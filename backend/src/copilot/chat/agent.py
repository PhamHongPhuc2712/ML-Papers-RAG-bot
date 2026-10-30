"""The served chat turn since 2026-10-30: the research agent (docs/results.md).

The agent searches the shipped first stage up to ``agent.max_searches`` times, may answer a
follow-up from the passages earlier answers cited, and hands the passages it chose to the grounded
generator. Every model call is under the spend ledger with purpose ``chat``. In mock mode a
fixture model searches the question once and answers from everything it saw, so the web app and
its tests run without a provider.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..agent.turn import AgentTurn, AgentUpdate
from ..contracts import Evidence
from .turns import HistoryItem

_QUESTION = re.compile(r"^Current question: (.*)$", re.MULTILINE)
_LABEL = re.compile(r"^\[((?:S\d+E\d+|H\d+)\+?)\]", re.MULTILINE)


class FixtureToolModel:
    """Mock mode's agent model: search the current question unless a search already ran, then
    answer from every passage shown. Deterministic and free."""

    identity = "fixture/research-agent"

    def step(
        self,
        messages: list[BaseMessage],
        tools: Sequence[dict[str, Any]],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> tuple[AIMessage, float]:
        user = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        text = str(user.content)
        match = _QUESTION.search(text)
        question = match.group(1).strip() if match else text.strip()
        searched = "A first search was run" in text or any(
            isinstance(m, ToolMessage) for m in messages
        )
        if not searched:
            call = {"name": "search_papers", "args": {"query": question}, "id": "fixture-search"}
        else:
            shown = "\n".join(str(m.content) for m in messages)
            ids = list(dict.fromkeys(_LABEL.findall(shown)))
            call = {
                "name": "answer",
                "args": {"question": question, "evidence_ids": ids},
                "id": "fixture-answer",
            }
        return AIMessage(content="", tool_calls=[call]), 0.0


@dataclass
class ChatAgent:
    """What the chat service needs from a turn: its steps, history depth and identity."""

    turn: AgentTurn
    history_turns: int = 3
    carried: int = 12
    identity: dict[str, Any] = field(default_factory=dict)

    def iter_steps(
        self,
        message: str,
        history: list[HistoryItem],
        carried: Sequence[Evidence],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> Iterator[AgentUpdate]:
        return self.turn.iter_steps(message, history, carried, request_id=request_id, run_id=run_id)


def build_chat_agent(
    settings: Any, engine: Any, search: Any, config_path: str = "configs/chat.yaml"
) -> ChatAgent:
    """Build the serving turn from the configured search, evidence and model adapters."""

    from ..agent.graph import LedgerToolModel
    from ..agent.tools import AgentLimits
    from ..agent.turn import load_agent_prompt
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
    from .config import chat_search_config, load_chat_config
    from .retriever import CorpusRetriever, ServicePaperSearch

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
    mock = settings.model_mode == "mock" or settings.mock_mode
    generator: Any
    model: Any
    if mock:
        model = FixtureToolModel()
        generator = FakeGenerator(prompt)
    else:
        ledger = SpendLedger(engine, settings.llm_daily_spend_cap_usd)
        cache = ResponseCache(settings.data_dir / "cache" / "llm")
        # The agent's own steps are not cached: each depends on the searches before it.
        model = LedgerToolModel(
            build_ledger_chat_model(
                config.model, settings, ledger, timeout=config.generate_timeout, cache=None
            ),
            "chat",
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
    turn = AgentTurn(
        model,
        retriever,
        evidence.section_chunks,
        generator,
        load_agent_prompt(config.agent_prompt),
        limits=AgentLimits(
            max_searches=config.max_searches,
            max_model_calls=config.max_model_calls,
            max_sections=config.max_sections,
        ),
        max_chars=evidence_config.expansion_max_chars,
        token_budget=evidence_config.token_budget,
        count_tokens=evidence.count_tokens,
        answer_from=config.answer_from,
        verbatim_first=config.verbatim_first,
    )
    identity = {
        "chat_config_sha256": config.sha256,
        "agent": turn.identity,
        "search": service.identity("hybrid_rerank"),
        "evidence": evidence_config.identity(),
    }
    return ChatAgent(
        turn, history_turns=config.history_turns, carried=config.carried, identity=identity
    )
