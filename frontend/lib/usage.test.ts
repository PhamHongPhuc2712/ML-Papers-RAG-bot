import { describe, expect, it } from "vitest";
import { answersLeft, limitReached } from "./usage";

const usage = (today: number, limit = 20) => ({ today, month: today, limit, resets_at: "2026-10-22T00:00:00Z" });

describe("answersLeft", () => {
  it("says nothing while more than 5 are left", () => {
    expect(answersLeft(usage(0))).toBeNull();
    expect(answersLeft(usage(14))).toBeNull();
    expect(answersLeft(null)).toBeNull();
  });

  it("counts down the last 5, then says none are left", () => {
    expect(answersLeft(usage(15))).toBe("5 answers left today");
    expect(answersLeft(usage(19))).toBe("1 answer left today");
    expect(answersLeft(usage(20))).toBe("No answers left today");
    expect(answersLeft(usage(23))).toBe("No answers left today");
  });
});

describe("limitReached", () => {
  it("names the limit and when asking works again, in the reader's terms", () => {
    const now = new Date("2026-10-21T21:20:00Z");
    expect(limitReached(20, "2026-10-22T00:00:00Z", now)).toBe(
      "You have used today's 20 answers. You can ask again after 00:00 UTC, in about 3 hours.",
    );
    expect(limitReached(20, "2026-10-22T00:00:00Z", new Date("2026-10-21T23:40:00Z"))).toBe(
      "You have used today's 20 answers. You can ask again after 00:00 UTC, in about 20 minutes.",
    );
  });
});
