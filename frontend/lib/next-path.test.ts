import { describe, expect, it } from "vitest";
import { safeNext } from "./next-path";

describe("safeNext", () => {
  it("keeps same-site paths", () => {
    expect(safeNext("/chat/abc")).toBe("/chat/abc");
    expect(safeNext("/?q=a%20b")).toBe("/?q=a%20b");
  });
  it("refuses anything that leaves the site, and nothing at all", () => {
    for (const bad of ["https://evil.example.org", "//evil.example.org", "/\\evil.example.org", "chat", "", null, undefined]) {
      expect(safeNext(bad)).toBe("/");
    }
  });
});
