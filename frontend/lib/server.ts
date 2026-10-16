import "server-only";
import { cookies } from "next/headers";

export const SESSION_COOKIE = "copilot_session";

/** Where the Next server reaches the API; the browser never sees it. */
export function backendUrl(): string {
  return (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
}

export function cookieSecure(): boolean {
  return process.env.SESSION_COOKIE_SECURE === "true";
}

/** Who the session cookie belongs to, for the account menu; null when it is absent or not accepted. */
export async function currentUser(): Promise<{ email: string } | null> {
  const token = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!token) return null;
  try {
    const response = await fetch(`${backendUrl()}/v1/me`, {
      headers: { authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    return response.ok ? ((await response.json()) as { email: string }) : null;
  } catch {
    return null;
  }
}
