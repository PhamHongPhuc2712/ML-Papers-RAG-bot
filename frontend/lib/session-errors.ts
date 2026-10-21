// A 401 ends the browser session only when the token itself is the problem. A wrong, used-up or
// expired code confirming a delete (DELETE /v1/me, DELETE /v1/chats) is also a 401, and must not
// sign the user out (sign-up plan, A4).
const KEEPS_SESSION = new Set(["invalid_code", "code_expired", "code_exhausted"]);

export function endsSession(status: number, body: string): boolean {
  if (status !== 401) return false;
  try {
    const code = (JSON.parse(body) as { error?: { code?: unknown } }).error?.code;
    return !(typeof code === "string" && KEEPS_SESSION.has(code));
  } catch {
    return true;
  }
}
