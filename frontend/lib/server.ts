import "server-only";

export const SESSION_COOKIE = "copilot_session";

/** Where the Next server reaches the API; the browser never sees it. */
export function backendUrl(): string {
  return (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
}

export function cookieSecure(): boolean {
  return process.env.SESSION_COOKIE_SECURE === "true";
}
