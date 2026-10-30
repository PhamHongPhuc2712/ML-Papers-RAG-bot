import { describe, expect, it } from "vitest";
import { nextRating } from "./feedback";

describe("nextRating", () => {
  it("sets, replaces and clears", () => {
    expect(nextRating(null, "helpful")).toBe("helpful");
    expect(nextRating("helpful", "not_helpful")).toBe("not_helpful");
    expect(nextRating("not_helpful", "not_helpful")).toBeNull();
  });
});
