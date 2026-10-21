import { describe, expect, it } from "vitest";
import { LOGIN_MESSAGES, isLoginError } from "./login-messages";

describe("isLoginError", () => {
  it("knows the API's code errors and the page's own", () => {
    for (const code of ["invalid_code", "code_expired", "code_exhausted", "mail_unavailable", "email", "unreachable"]) {
      expect(isLoginError(code)).toBe(true);
    }
  });

  it("refuses anything else, including inherited names", () => {
    for (const value of [undefined, "", "credentials", "toString", "__proto__"]) expect(isLoginError(value)).toBe(false);
    expect(Object.keys(LOGIN_MESSAGES)).toHaveLength(6);
  });
});
