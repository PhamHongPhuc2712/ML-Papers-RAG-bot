"""The agent's tools: unique passage ids, a search budget, sections on demand."""

from __future__ import annotations

from uuid import uuid4

from copilot.agent.tools import TOOL_SCHEMAS, AgentLimits, AgentTools
from copilot.chat.retriever import CorpusRetriever, PaperHits
from copilot.contracts import Evidence
from copilot.evidence.retrieve import ChunkRecord


def _evidence(label: str, text: str = "Dropout noise as augmentation.") -> Evidence:
    return Evidence(
        id=label,
        source_kind="paper",
        source_id=uuid4(),
        version_id=str(uuid4()),
        chunk_id=uuid4(),
        text=text,
        section="Method",
        page_start=3,
        page_end=3,
    )


def _chunk(evidence: Evidence, chunk_id: str, ordinal: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        paper_id=str(evidence.source_id),
        version_id=evidence.version_id,
        section="Method",
        kind="body",
        ordinal=ordinal,
        page_start=3,
        page_end=3,
        text=text,
        evidence_default=True,
    )


class Corpus:
    def __init__(self, batches: list[list[Evidence]]) -> None:
        self.batches = list(batches)
        self.queries: list[str] = []

    def search(self, query: str) -> PaperHits:
        self.queries.append(query)
        return PaperHits("rel-1", [str(e.source_id) for e in self.batches[0]], ["rerank_degraded"])

    def evidence(self, query: str, papers: list[str], release: str) -> list[Evidence]:
        return self.batches.pop(0) if len(self.batches) > 1 else list(self.batches[0])


def _tools(corpus: Corpus, chunks=None, **limits) -> AgentTools:
    return AgentTools(
        CorpusRetriever(search=corpus.search, evidence=corpus.evidence),
        chunks or (lambda item: []),
        limits=AgentLimits(**limits),
        max_chars=6000,
    )


def test_each_search_gets_its_own_ids():
    corpus = Corpus([[_evidence("E1"), _evidence("E2")], [_evidence("E1", "Other.")]])
    tools = _tools(corpus)
    first = tools.search_papers("dropout")
    second = tools.search_papers("dropout augmentation")
    assert "[S1E1]" in first and "[S1E2]" in first and "[S2E1]" in second
    assert set(tools.seen) == {"S1E1", "S1E2", "S2E1"}
    assert tools.searches == 2 and tools.queries == ["dropout", "dropout augmentation"]
    assert tools.release_id == "rel-1" and tools.warnings == ["rerank_degraded"]
    # The union over both searches: two papers from the first, one more from the second.
    assert len(tools.paper_ids) == 3


def test_a_repeated_passage_is_referenced_not_resent():
    same = _evidence("E1")
    corpus = Corpus([[same], [same.model_copy(update={"id": "E1"})]])
    tools = _tools(corpus)
    tools.search_papers("dropout")
    again = tools.search_papers("dropout")
    assert "already shown as S1E1" in again and same.text not in again
    assert tools.searches == 2 and set(tools.seen) == {"S1E1"}


def test_the_fourth_search_is_refused_and_not_run():
    corpus = Corpus([[_evidence("E1")]])
    tools = _tools(corpus, max_searches=3)
    for query in ("a", "b", "c"):
        tools.search_papers(query)
    refused = tools.search_papers("d")
    assert "Search limit reached" in refused
    assert corpus.queries == ["a", "b", "c"] and tools.searches == 3


def test_a_search_with_no_passages_says_so():
    tools = _tools(Corpus([[]]))
    assert "no passages found" in tools.search_papers("nothing")
    assert tools.searches == 1 and tools.seen == {}


def test_carried_passages_are_labelled_h():
    tools = _tools(Corpus([[_evidence("E1")]]))
    block = tools.carry([_evidence("E3"), _evidence("E7")])
    assert "[H1]" in block and "[H2]" in block and set(tools.seen) == {"H1", "H2"}
    assert tools.carry([]) == "none"


def test_a_carried_passage_found_again_is_referenced_by_its_h_id():
    carried = _evidence("E3")
    tools = _tools(Corpus([[carried.model_copy(update={"id": "E1"})]]))
    tools.carry([carried])
    assert "already shown as H1" in tools.search_papers("q")


def test_read_section_widens_a_seen_passage_under_a_new_id():
    hit = _evidence("E1", "Hit.")
    rows = [
        _chunk(hit, str(uuid4()), 0, "Before."),
        _chunk(hit, str(hit.chunk_id), 1, "Hit."),
    ]
    tools = _tools(Corpus([[hit]]), chunks=lambda item: rows)
    tools.search_papers("q")
    widened = tools.read_section("S1E1")
    assert "[S1E1+]" in widened and "Before." in widened and tools.sections == 1
    assert tools.seen["S1E1+"].expanded


def test_read_section_on_a_whole_section_says_there_is_nothing_more():
    hit = _evidence("E1", "Hit.")
    tools = _tools(Corpus([[hit]]), chunks=lambda item: [_chunk(hit, str(hit.chunk_id), 0, "Hit.")])
    tools.search_papers("q")
    assert "already shows" in tools.read_section("S1E1") and tools.sections == 0


def test_read_section_refuses_unknown_and_abstract_only_ids():
    tools = _tools(Corpus([[_evidence("E1")]]))
    assert "Unknown passage id" in tools.read_section("S9E9")
    tools.seen["A1"] = _evidence("A1").model_copy(update={"chunk_id": None, "abstract_only": True})
    assert "No section to read" in tools.read_section("A1")


def test_the_schemas_name_three_tools():
    names = [tool["function"]["name"] for tool in TOOL_SCHEMAS]
    assert names == ["search_papers", "read_section", "answer"]
