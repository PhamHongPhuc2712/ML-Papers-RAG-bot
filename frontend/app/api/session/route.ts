import { cookies } from "next/headers";
import { safeNext } from "@/lib/next-path";
import { SESSION_COOKIE, backendUrl, cookieSecure } from "@/lib/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const unreachable = {
  error: { code: "backend_unreachable", message: "The service is not reachable.", retryable: true },
};

type Credentials = { email?: unknown; password?: unknown };

/** A form posted before the page's script ran (no JavaScript yet, or none at all): log in, then
 *  redirect, so the password never travels in a URL. The script's own path sends JSON. */
async function fromForm(request: Request): Promise<Response> {
  const form = await request.formData();
  const next = safeNext(typeof form.get("next") === "string" ? (form.get("next") as string) : null);
  const outcome = await logIn({ email: form.get("email"), password: form.get("password") });
  const target = outcome.ok ? next : `/login?${new URLSearchParams({ next, error: outcome.status === 401 ? "credentials" : "unreachable" })}`;
  return new Response(null, { status: 303, headers: { location: target } });
}

export async function POST(request: Request): Promise<Response> {
  if (request.headers.get("content-type")?.startsWith("application/x-www-form-urlencoded")) return fromForm(request);
  const body = (await request.json().catch(() => null)) as Credentials | null;
  const outcome = await logIn(body ?? {});
  return outcome.ok ? Response.json({ ok: true }) : Response.json(outcome.body, { status: outcome.status });
}

type Outcome = { ok: true } | { ok: false; status: number; body: unknown };

/** Exchange credentials for a token and keep it in the httpOnly cookie. */
async function logIn(body: Credentials): Promise<Outcome> {
  if (typeof body.email !== "string" || typeof body.password !== "string") {
    return { ok: false, status: 422, body: { error: { code: "validation_error", message: "Email and password are required.", retryable: false } } };
  }
  let upstream: Response;
  try {
    upstream = await fetch(`${backendUrl()}/v1/auth/token`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: body.email, password: body.password }),
      cache: "no-store",
    });
  } catch {
    return { ok: false, status: 503, body: unreachable };
  }
  const payload = (await upstream.json().catch(() => null)) as
    | { access_token?: string; expires_in?: number }
    | null;
  if (!upstream.ok || !payload?.access_token) {
    return { ok: false, status: upstream.ok ? 502 : upstream.status, body: payload ?? unreachable };
  }
  (await cookies()).set(SESSION_COOKIE, payload.access_token, {
    httpOnly: true,
    sameSite: "lax",
    secure: cookieSecure(),
    path: "/",
    maxAge: payload.expires_in ?? 3600,
  });
  return { ok: true };
}

export async function DELETE(): Promise<Response> {
  (await cookies()).delete(SESSION_COOKIE);
  return new Response(null, { status: 204 });
}
