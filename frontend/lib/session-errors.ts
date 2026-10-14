// A 401 ends the browser session only when the token itself is the problem. A wrong password
// on a re-authenticated action (DELETE /v1/me) is also a 401, and must not sign the user out.
const KEEPS_SESSION = new Set(["invalid_credentials"]);

export function endsSession(status: number, body: string): boolean {
  if (status !== 401) return false;
  try {
    const code = (JSON.parse(body) as { error?: { code?: unknown } }).error?.code;
    return !(typeof code === "string" && KEEPS_SESSION.has(code));
  } catch {
    return true;
  }
}
