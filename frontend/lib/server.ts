import "server-only";
import { cookies } from "next/headers";
import { summarizeCoverage, type Corpus, type Coverage } from "./corpus";

export const SESSION_COOKIE = "copilot_session";
/** The email a sign-in code was sent to, for the code step (sign-up plan, A2); 15 minutes. */
export const LOGIN_EMAIL_COOKIE = "copilot_login_email";

/** Where the Next server reaches the API; the browser never sees it. */
export function backendUrl(): string {
  return (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
}

export function cookieSecure(): boolean {
  return process.env.SESSION_COOKIE_SECURE === "true";
}

export type CurrentUser = {
  email: string;
  usage?: { today: number; month: number; limit: number; resets_at: string } | null;
};

/** Who the session cookie belongs to, and their answers used (Settings); null when it is absent or not accepted. */
export async function currentUser(): Promise<CurrentUser | null> {
  const token = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!token) return null;
  try {
    const response = await fetch(`${backendUrl()}/v1/me`, {
      headers: { authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    return response.ok ? ((await response.json()) as CurrentUser) : null;
  } catch {
    return null;
  }
}

/** The corpus's venues, years and size, cached for five minutes; null when the API is unreachable. */
export async function corpus(): Promise<Corpus | null> {
  try {
    const response = await fetch(`${backendUrl()}/v1/corpus/coverage`, { next: { revalidate: 300 } });
    return response.ok ? summarizeCoverage((await response.json()) as Coverage) : null;
  } catch {
    return null;
  }
}
