import { describe, expect, it } from "vitest";
import { corpusLine, filterLabel, listInWords, summarizeCoverage, withFilter } from "./corpus";

const COVERAGE = {
  as_of: "2026-10-11T00:00:00Z",
  venue_years: [
    { venue: "ICLR", year: 2025, papers: 1 },
    { venue: "ACL", year: 2024, papers: 1 },
    { venue: "ACL", year: 2025, papers: 1 },
  ],
  totals: { papers: 3, venue_years: 3 },
};

describe("corpus", () => {
  it("lists each venue once, alphabetically, and years newest first", () => {
    expect(summarizeCoverage(COVERAGE)).toEqual({ papers: 3, venues: ["ACL", "ICLR"], years: [2025, 2024] });
  });
  it("says the venues and the span in words", () => {
    expect(listInWords(["A"])).toBe("A");
    expect(listInWords(["A", "B", "C"])).toBe("A, B and C");
    expect(corpusLine(summarizeCoverage(COVERAGE))).toBe("Searches the main-conference papers of ACL and ICLR from 2024 to 2025.");
    expect(corpusLine(null)).toBeNull();
  });
  it("writes a chosen filter into the question and leaves it alone without one", () => {
    expect(withFilter("Find papers on LoRA", { venue: null, year: null })).toBe("Find papers on LoRA");
    expect(withFilter("Find papers on LoRA", { venue: "ACL", year: 2024 })).toBe("Find papers on LoRA\n\nOnly use papers from ACL 2024.");
    expect(filterLabel({ venue: "ACL", year: null })).toBe("ACL");
    expect(filterLabel({ venue: null, year: null })).toBeNull();
  });
});
