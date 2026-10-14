import { describe, expect, it } from "vitest";
import { endsSession } from "./session-errors";

const body = (code: string) => JSON.stringify({ error: { code, message: "", retryable: false } });

describe("endsSession", () => {
  it("ends the session for token errors", () => {
    expect(endsSession(401, body("token_expired"))).toBe(true);
    expect(endsSession(401, body("token_invalid"))).toBe(true);
    expect(endsSession(401, "not json")).toBe(true);
  });

  it("keeps it for a wrong password and for anything but 401", () => {
    expect(endsSession(401, body("invalid_credentials"))).toBe(false);
    expect(endsSession(403, body("token_invalid"))).toBe(false);
    expect(endsSession(204, "")).toBe(false);
  });
});
