"""The research agent's loop as a LangGraph graph: agent ⇄ tools, until it answers.

Two nodes. `agent` asks the model for its next step; `tools` runs the calls it made. The
loop ends when the model calls `answer`, replies without a tool, or reaches the
model-call cap. Answer text is not written here: the turn hands the chosen passages to the
existing generator, so the agent and the pipeline differ only in how evidence is gathered.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, Any, Protocol, TypedDict
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from ..models.lc_chat import LedgerChatModel
from .tools import TOOL_SCHEMAS, AgentLimits, AgentTools


class ToolCallingModel(Protocol):
    def step(
        self,
        messages: list[BaseMessage],
        tools: Sequence[dict[str, Any]],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> tuple[AIMessage, float]: ...


class LedgerToolModel:
    """The chat model's tool calls, every one under the ledger with one purpose."""

    def __init__(self, chat: LedgerChatModel, purpose: str) -> None:
        self._chat = chat
        self._purpose = purpose
        self.identity = f"{chat.model.key}={chat.model.model}"

    def step(
        self,
        messages: list[BaseMessage],
        tools: Sequence[dict[str, Any]],
        *,
        request_id: UUID | None = None,
        run_id: str | None = None,
    ) -> tuple[AIMessage, float]:
        return self._chat.invoke_tools(
            messages, tools, purpose=self._purpose, request_id=request_id, run_id=run_id
        )


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    model_calls: int
    cost_usd: float
    final: dict[str, Any] | None


def _run_calls(
    tools: AgentTools, message: AIMessage
) -> tuple[list[BaseMessage], dict[str, Any] | None]:
    replies: list[BaseMessage] = []
    final: dict[str, Any] | None = None
    for bad in message.invalid_tool_calls:
        replies.append(
            ToolMessage(
                content=f"Malformed call to {bad.get('name')}: send valid JSON arguments.",
                tool_call_id=str(bad.get("id") or "invalid"),
            )
        )
    for call in message.tool_calls:
        args = call["args"]
        if call["name"] == "search_papers":
            text = tools.search_papers(str(args.get("query", "")).strip())
        elif call["name"] == "read_section":
            text = tools.read_section(str(args.get("evidence_id", "")).strip())
        elif call["name"] == "answer":
            final = {
                "question": str(args.get("question", "")).strip(),
                "evidence_ids": [str(label) for label in args.get("evidence_ids", [])],
            }
            text = "Recorded."
        else:
            text = f"Unknown tool {call['name']}."
        replies.append(ToolMessage(content=text, tool_call_id=str(call["id"])))
    return replies, final


def build_agent_graph(
    model: ToolCallingModel,
    tools: AgentTools,
    limits: AgentLimits,
    *,
    request_id: UUID | None,
    run_id: str | None,
) -> Any:
    def agent(state: AgentState) -> dict[str, Any]:
        message, cost = model.step(
            state["messages"], TOOL_SCHEMAS, request_id=request_id, run_id=run_id
        )
        return {
            "messages": [message],
            "model_calls": state["model_calls"] + 1,
            "cost_usd": state["cost_usd"] + cost,
        }

    def run_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        replies, final = _run_calls(tools, last)
        return {"messages": replies, "final": final}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        return "tools" if last.tool_calls or last.invalid_tool_calls else END

    def after_tools(state: AgentState) -> str:
        if state["final"] is not None or state["model_calls"] >= limits.max_model_calls:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()
