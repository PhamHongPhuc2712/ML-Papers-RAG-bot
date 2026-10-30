"""What the research agent can do in one turn: search, widen a passage, and answer.

Ids are unique within a turn: search n's k-th new passage is S{n}E{k}, a passage carried
from earlier turns is H{k}, a widened passage is its id plus "+". A passage already shown
is referred to by its first id instead of being sent again.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..chat.retriever import CorpusRetriever
from ..contracts import Evidence, Reference
from ..evidence.retrieve import ChunkRecord, expand_to_section
from ..models.generation import evidence_block


@dataclass(frozen=True)
class AgentLimits:
    max_searches: int = 3
    max_model_calls: int = 8
    max_sections: int = 4


def _function(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _function(
        "search_papers",
        "Search the corpus. Returns passages with ids. Write a standalone query.",
        {"query": {"type": "string"}},
    ),
    _function(
        "read_section",
        "Read the whole section around a passage you have already seen, by its id.",
        {"evidence_id": {"type": "string"}},
    ),
    _function(
        "answer",
        "Finish: the standalone question being answered and the ids of the passages that "
        "answer it. An empty list means the passages do not answer it.",
        {
            "question": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
    ),
]


@dataclass
class AgentTools:
    retriever: CorpusRetriever
    section_chunks: Callable[[Evidence], list[ChunkRecord]]
    limits: AgentLimits
    max_chars: int
    seen: dict[str, Evidence] = field(default_factory=dict)
    searches: int = 0
    sections: int = 0
    queries: list[str] = field(default_factory=list)
    paper_ids: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    release_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    _by_chunk: dict[str, str] = field(default_factory=dict)

    def carry(self, evidence: Sequence[Evidence]) -> str:
        """Label passages from earlier turns H1..Hn; "none" when there are none."""

        if not evidence:
            return "none"
        blocks = []
        for index, item in enumerate(evidence, 1):
            labelled = item.model_copy(update={"id": f"H{index}"})
            self._remember(labelled)
            blocks.append(evidence_block(labelled))
        return "\n\n".join(blocks)

    def search_papers(self, query: str) -> str:
        if self.searches >= self.limits.max_searches:
            return (
                f"Search limit reached ({self.searches} of {self.limits.max_searches}). "
                "Answer from the passages you already have."
            )
        self.searches += 1
        self.queries.append(query)
        evidence, hits = self.retriever.search_evidence(query)
        self.release_id = hits.release_id
        self.warnings.extend(w for w in hits.warnings if w not in self.warnings)
        for paper_id in hits.paper_ids:
            if paper_id not in self.paper_ids:
                self.paper_ids.append(paper_id)
        self._merge_references(hits.references)
        if not evidence:
            return f"Search {self.searches} for “{query}”: no passages found."
        blocks = []
        fresh = 0
        for item in evidence:
            known = self._by_chunk.get(str(item.chunk_id)) if item.chunk_id else None
            if known is not None:
                blocks.append(f"[{known}] already shown as {known}.")
                continue
            fresh += 1
            labelled = item.model_copy(update={"id": f"S{self.searches}E{fresh}"})
            self._remember(labelled)
            blocks.append(evidence_block(labelled))
        header = f"Search {self.searches} for “{query}”: {len(evidence)} passages."
        return "\n\n".join([header, *blocks])

    def read_section(self, evidence_id: str) -> str:
        item = self.seen.get(evidence_id)
        if item is None:
            return f"Unknown passage id {evidence_id}. Use an id shown above."
        if item.chunk_id is None or item.abstract_only:
            return f"No section to read for {evidence_id}: it is an abstract."
        if self.sections >= self.limits.max_sections:
            return "Section limit reached. Answer from the passages you already have."
        grown = expand_to_section(
            item, chunks=self.section_chunks(item), max_chars=self.max_chars, min_share=0.0
        )
        if not grown.expanded:
            return f"{evidence_id} already shows all of its section that fits."
        self.sections += 1
        widened = grown.model_copy(update={"id": f"{evidence_id}+"})
        self.seen[widened.id] = widened
        return evidence_block(widened)

    def _merge_references(self, references: Sequence[Reference]) -> None:
        """Every search's ranked papers, in the order first seen, ranks renumbered; a paper
        read by any search is marked read."""

        by_id = {str(r.paper_id): i for i, r in enumerate(self.references)}
        for reference in references:
            index = by_id.get(str(reference.paper_id))
            if index is None:
                by_id[str(reference.paper_id)] = len(self.references)
                self.references.append(reference)
            elif reference.read and not self.references[index].read:
                self.references[index] = self.references[index].model_copy(update={"read": True})
        self.references = [
            r.model_copy(update={"rank": rank}) for rank, r in enumerate(self.references, 1)
        ]

    def _remember(self, item: Evidence) -> None:
        self.seen[item.id] = item
        if item.chunk_id is not None:
            self._by_chunk.setdefault(str(item.chunk_id), item.id)
