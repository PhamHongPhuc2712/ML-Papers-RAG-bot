import { describe, expect, it } from "vitest";
import type { ChatMessage, Reference, StoredEvidence } from "./api";
import { answerAsText, citedByPaper, filterReferences, partialSearch, passagesOf, referenceFacets, stepsSummary, where } from "./references";

const ref = (rank: number, venue: string | null, year: number | null, read = rank <= 2): Reference => ({
  paper_id: `p${rank}`, rank, read, title: `Paper ${rank}`, venue, year,
});
const REFS = [ref(1, "ICLR", 2025), ref(2, "ACL", 2024), ref(3, "ICLR", 2024), ref(4, null, null)];
const passage = (id: string, paper: string, extra: Partial<StoredEvidence> = {}): StoredEvidence => ({
  id, paper_id: paper, title: `Title ${paper}`, venue: "ACL", year: 2024, section: "Method", page_start: 3, page_end: 3,
  text: "t", abstract_only: false, cited: false, ...extra,
});
const MESSAGE = {
  standalone_query: "graph transformers", ordinal: 1, references: REFS, warnings: [],
  claims: [{ text: "a.", evidence_ids: ["E2"] }, { text: "b.", evidence_ids: ["E1", "E3"] }],
  evidence: [passage("E1", "p1"), passage("E2", "p2"), passage("E3", "p1"), passage("E4", "p1")],
} as unknown as ChatMessage;

describe("steps and warnings", () => {
  it("summarizes what was searched and read", () => {
    expect(stepsSummary(MESSAGE)).toBe("Searched for “graph transformers” and read 2 of 4 papers");
    expect(stepsSummary({ ...MESSAGE, references: [] })).toBe("Searched for “graph transformers”");
    expect(stepsSummary({ ...MESSAGE, standalone_query: null, evidence: [] })).toBeNull();
  });
  it("lists every search the agent ran, and says so when it searched nothing", () => {
    expect(stepsSummary({ ...MESSAGE, search_queries: ["a", "b"] })).toBe("Searched for “a” and “b”, then read 2 of 4 papers");
    expect(stepsSummary({ ...MESSAGE, search_queries: ["a", "b", "c"], references: [] })).toBe("Searched for “a”, “b” and “c”");
    expect(stepsSummary({ ...MESSAGE, search_queries: [], standalone_query: null })).toBe(
      "Answered from passages cited earlier in this chat",
    );
  });
  it("calls a search partial when a stage timed out or was missing, not for a rewritten follow-up", () => {
    expect(partialSearch({ ...MESSAGE, warnings: ["lexical_timeout"] })).toBe(true);
    expect(partialSearch({ ...MESSAGE, warnings: ["rerank_unavailable"] })).toBe(true);
    expect(partialSearch({ ...MESSAGE, warnings: ["condense_failed"] })).toBe(false);
    expect(partialSearch(MESSAGE)).toBe(false);
  });
});

describe("references", () => {
  it("offers each venue and year once, newest year first", () => {
    expect(referenceFacets(REFS)).toEqual({ venues: ["ACL", "ICLR"], years: [2025, 2024] });
  });
  it("filters by venue, year and cited papers, keeping rank order", () => {
    const cited = citedByPaper(MESSAGE);
    const all = { venue: null, year: null, citedOnly: false };
    expect(filterReferences(REFS, { ...all, venue: "ICLR" }, cited).map((r) => r.rank)).toEqual([1, 3]);
    expect(filterReferences(REFS, { ...all, venue: "ICLR", year: 2024 }, cited).map((r) => r.rank)).toEqual([3]);
    expect(filterReferences(REFS, { ...all, citedOnly: true }, cited).map((r) => r.rank)).toEqual([1, 2]);
  });
  it("maps each paper to its citation numbers", () => {
    expect([...citedByPaper(MESSAGE)]).toEqual([["p2", [1]], ["p1", [2, 3]]]);
  });
  it("splits a paper's passages into cited, in citation order, and only read", () => {
    const { cited, read } = passagesOf(MESSAGE, "p1");
    expect(cited.map((p) => p.id)).toEqual(["E1", "E3"]);
    expect(read.map((p) => p.id)).toEqual(["E4"]);
  });
  it("says where a passage is", () => {
    expect(where(passage("E", "p"))).toBe("Method, page 3");
    expect(where(passage("E", "p", { page_end: 4 }))).toBe("Method, pages 3–4");
    expect(where(passage("E", "p", { page_start: null, section: "Abstract" }))).toBe("Abstract");
  });
  it("copies the answer with its numbered sources", () => {
    expect(answerAsText(MESSAGE)).toBe("a. [1]\nb. [2][3]\n\n[1] Title p2, ACL 2024\n[2] Title p1, ACL 2024\n[3] Title p1, ACL 2024");
  });
});
