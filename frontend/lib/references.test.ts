import { describe, expect, it } from "vitest";
import type { ChatMessage, Reference } from "./api";
import { citedByPaper, filterReferences, referenceFacets, searchLine } from "./references";

const ref = (rank: number, venue: string | null, year: number | null, read = rank <= 2): Reference => ({
  paper_id: `p${rank}`, rank, read, title: `Paper ${rank}`, venue, year,
});
const REFS = [ref(1, "ICLR", 2025), ref(2, "ACL", 2024), ref(3, "ICLR", 2024), ref(4, null, null)];

describe("search line", () => {
  it("names the query and how many papers were read", () => {
    const message = { standalone_query: "graph transformers", ordinal: 1, references: REFS } as unknown as ChatMessage;
    expect(searchLine(message)).toBe("Searched for “graph transformers” · read 2 papers");
  });
  it("says one paper in the singular", () => {
    const message = { standalone_query: "q", ordinal: 1, references: [ref(1, "ACL", 2024)] } as unknown as ChatMessage;
    expect(searchLine(message)).toBe("Searched for “q” · read 1 paper");
  });
  it("keeps an answer stored before References showing only a follow-up's query", () => {
    const first = { standalone_query: "q", ordinal: 1 } as unknown as ChatMessage;
    const follow = { standalone_query: "q", ordinal: 3, references: [] } as unknown as ChatMessage;
    expect(searchLine(first)).toBeNull();
    expect(searchLine(follow)).toBe("Searched for “q”");
  });
});

describe("references", () => {
  it("offers each venue and year once, newest year first", () => {
    expect(referenceFacets(REFS)).toEqual({ venues: ["ACL", "ICLR"], years: [2025, 2024] });
  });
  it("filters by venue and year together, keeping rank order", () => {
    expect(filterReferences(REFS, { venue: "ICLR", year: null }).map((r) => r.rank)).toEqual([1, 3]);
    expect(filterReferences(REFS, { venue: "ICLR", year: 2024 }).map((r) => r.rank)).toEqual([3]);
    expect(filterReferences(REFS, { venue: null, year: null })).toHaveLength(4);
  });
  it("maps each paper to the citation numbers its passages carry", () => {
    const message = {
      claims: [{ text: "a", evidence_ids: ["E2"] }, { text: "b", evidence_ids: ["E1", "E3"] }],
      evidence: [
        { id: "E1", paper_id: "p1" }, { id: "E2", paper_id: "p2" }, { id: "E3", paper_id: "p1" }, { id: "E4", paper_id: "p1" },
      ],
    } as unknown as ChatMessage;
    expect([...citedByPaper(message)]).toEqual([["p2", [1]], ["p1", [2, 3]]]);
  });
});
